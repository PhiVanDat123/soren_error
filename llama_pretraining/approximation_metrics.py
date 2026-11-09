"""Diagnostics on the actual momentum input and polynomial output, before LR/shape scaling."""
import json
import re
from pathlib import Path

import torch
import torch.distributed as dist
from summarize_approximation import HEADER, summarize


@torch.no_grad()
def approximation_metrics(matrix, approximate, kind):
    """Compare actual output with SVD references; ratios are fractions, logs are base 2.

    Soren's primary reference matches exact_matrix_sigmoid (raw input). The
    additional normalized reference diagnoses the normalization used by SNS's T
    stream. No reference computation changes the update used for training.
    """
    x, y = matrix.detach().double(), approximate.detach().double()
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    norm = x.norm()
    references = {"svd_polar": torch.ones_like(s)} if kind == "muon" else {
        "svd_sigmoid_raw": s.sigmoid(),
        "svd_sigmoid_fro_normalized": (s / (norm + 1e-8)).sigmoid(),
    }
    # Retain signs and the input's basis: sorting output singular values would
    # hide sign flips and mode permutations in the per-mode relative error.
    projected = u.mT @ y @ vh.mT
    coefficients = projected.diagonal()
    output_s = torch.linalg.svdvals(y)
    floor = 1e-30
    log_y = output_s.clamp_min(floor).log2()
    log_y = log_y - log_y.quantile(0.5)
    rows = []
    for reference, target in references.items():
        exact = (u * target) @ vh
        delta = coefficients - target
        log_exact = target.sort(descending=True).values.clamp_min(floor).log2()
        log_exact = log_exact - log_exact.quantile(0.5)
        width_exact = log_exact.max() - log_exact.min()
        width_approx = log_y.max() - log_y.min()
        rows.append(dict(
            reference=reference,
            exact_frobenius_squared=exact.square().sum().item(),
            error_frobenius_squared=(y - exact).square().sum().item(),
            relative_frobenius_error=((y - exact).norm() / exact.norm()).item(),
            max_per_mode_relative_error=(delta.abs() / target).max().item(),
            mean_per_mode_relative_error=(delta.abs() / target).mean().item(),
            mae=delta.abs().mean().item(),
            max_absolute_error=delta.abs().max().item(),
            log_centered_max_error=(log_y - log_exact).abs().max().item(),
            log_centered_rmse=(log_y - log_exact).square().mean().sqrt().item(),
            width_exact=width_exact.item(), width_approx=width_approx.item(),
            width_difference=(width_approx - width_exact).item(),
            off_diagonal_relative_error=(projected - torch.diag(coefficients)).norm().div(exact.norm()).item(),
            negative_coefficients=int((coefficients < 0).sum().item()),
            log_clamped_modes=int((output_s < floor).sum().item()),
        ))
    rank_tol = max(x.shape) * torch.finfo(torch.float64).eps * s.max()
    common = dict(input_frobenius_norm=norm.item(), input_sigma_min=s.min().item(),
                  input_sigma_max=s.max().item(),
                  numerical_rank=int((s > rank_tol).sum().item()),
                  num_modes=s.numel(), log_floor=floor)
    return [dict(common, **row) for row in rows]


class ApproximationLogger:
    """Append one JSON object per line to a rank-specific .txt file.

    All ranks participate, since distributed optimizers assign matrices to ranks.
    The caller sets step to the absolute optimizer update step (also on resume).
    """
    def __init__(self, directory, named_parameters, optimizer_name, rank=0,
                 every=0, steps=(), parameter_regex=None):
        if every < 0:
            raise ValueError("approximation_log_every must be nonnegative")
        self.names = {id(p): name for name, p in named_parameters}
        self.optimizer_name = optimizer_name
        self.rank, self.every, self.steps = rank, every, set(steps)
        self.pattern = re.compile(parameter_regex) if parameter_regex else None
        self.step = 0
        self.pending = []
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.path = Path(directory) / f"{optimizer_name}_approximation_rank{rank}.txt"

    def callback(self, parameter):
        if self.step not in self.steps and not (self.every and self.step > 0 and self.step % self.every == 0):
            return None
        name = self.names[id(parameter)]
        if self.pattern and not self.pattern.search(name):
            return None

        def record(matrix, output, kind):
            rows = approximation_metrics(matrix, output, kind)
            with self.path.open("a", encoding="utf-8") as stream:
                for row in rows:
                    row.update(step=self.step, rank=self.rank, parameter=name,
                               shape=list(matrix.shape), optimizer=self.optimizer_name,
                               approximation="sns" if kind == "soren" else "ns_quintic_bfloat16",
                               input_dtype=str(matrix.dtype), output_dtype=str(output.dtype),
                               input_kind="nesterov_momentum", ns_steps=5)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    self.pending.append(row)
        return record

    def flush_summary(self):
        """Called by every rank after a sampled optimizer update."""
        if self.step not in self.steps and not (self.every and self.step > 0 and self.step % self.every == 0):
            return
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size() if self.rank == 0 else None
            dist.gather_object(self.pending, gathered, dst=0)
            rows = [row for rank_rows in gathered for row in rank_rows] if self.rank == 0 else []
        else:
            rows = self.pending
        if self.rank == 0 and rows:
            path = self.path.parent / f"{self.optimizer_name}_approximation_summary.txt"
            needs_header = not path.exists() or path.stat().st_size == 0
            with path.open('a', encoding='utf-8') as stream:
                if needs_header:
                    stream.write(HEADER)
                stream.write(summarize(rows))
        self.pending = []


def approximation_callback(optimizer, parameter):
    diagnostic = getattr(optimizer, "approximation_logger", None)
    return diagnostic.callback(parameter) if diagnostic is not None else None
