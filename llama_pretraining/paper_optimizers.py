"""Optimizers from arXiv:2610.10965v1 and arXiv:2605.11181v1.

Targeted Muon: Algorithm 1, convergent fp32 polynomial, absolute threshold.
Kaon: Algorithm 2 (chaotic polynomial, not fresh random SVD coefficients).
Freon: exact SVD spectral definition, mean-Schatten normalization, NOT QDWH.
TruncatedSGD: remove the top fraction and preserve the original Frobenius norm.
"""
import math

import torch
import torch.distributed as dist


def _wide(matrix):
    if matrix.ndim != 2:
        raise ValueError('Spectral transforms require a 2-D matrix')
    transposed = matrix.shape[0] > matrix.shape[1]
    return (matrix.mT if transposed else matrix).float(), transposed


def convergent_sign(matrix, steps=20):
    """p(x)=3x-16/5*x^3+6/5*x^5; works for rectangular polar or symmetric sign."""
    x = matrix.float()
    x = x / x.norm().clamp_min(1e-30)
    for _ in range(steps):
        gram = x @ x.mT
        x = 3 * x + (-3.2 * gram + 1.2 * (gram @ gram)) @ x
    return x


def targeted_muon_transform(matrix, threshold=0.01, target='bottom', steps=20):
    x, transposed = _wide(matrix)
    # Left projectors on the smaller Gram matrix are algebraically equivalent
    # to the paper's right projectors; avoid an n x n Gram for wide matrices.
    identity = torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
    shifted = x @ x.mT - threshold ** 2 * identity
    sign = convergent_sign(shifted, steps)
    top = 0.5 * (identity + sign)
    bottom = identity - top
    polar = convergent_sign(x, steps)
    if target == 'top':
        result = top @ polar + bottom @ x
    elif target == 'bottom':
        result = top @ x + bottom @ polar
    else:
        raise ValueError('target must be top or bottom')
    return result.mT if transposed else result


def kaon_transform(matrix, steps=5):
    x, transposed = _wide(matrix)
    x = x / x.norm().clamp_min(1e-30)
    identity = torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
    for _ in range(steps):
        residual = identity - x @ x.mT
        x = 4.1 * (residual @ residual) @ x
    x = x / 1.175
    return x.mT if transposed else x


def freon_svd_transform(matrix, exponent=2/3, relative_floor=1e-6):
    """U diag((sigma / power_mean_q(sigma))**(q-1)) Vh, q=2-2c.

    Small singular values are regularized relative to sigma_max. Exact zero
    modes stay zero rather than selecting arbitrary null-space SVD directions.
    """
    x, transposed = _wide(matrix)
    u, singular, vh = torch.linalg.svd(x, full_matrices=False)
    safe = singular.clamp_min((singular.max() * relative_floor).clamp_min(1e-30))
    logs = safe.log()
    q = 2 - 2 * exponent
    log_mean = logs.mean() if abs(q) < 1e-8 else (
        torch.logsumexp(q * logs, dim=0) - math.log(singular.numel())) / q
    values = ((q - 1) * (logs - log_mean)).exp()
    values = torch.where(singular > 0, values, torch.zeros_like(values))
    result = (u * values) @ vh
    return result.mT if transposed else result


def truncated_sgd_transform(matrix, fraction=0.05):
    x, transposed = _wide(matrix)
    if fraction == 0:
        return matrix.float().clone()
    u, singular, vh = torch.linalg.svd(x, full_matrices=False)
    removed = math.ceil(fraction * singular.numel())
    kept = singular.clone()
    kept[:removed] = 0
    result = (u * kept) @ vh
    result *= singular.norm() / kept.norm().clamp_min(1e-30)
    return result.mT if transposed else result


class PaperOptimizerWithAuxAdam(torch.optim.Optimizer):
    """Shard spectral updates across DDP ranks; use AdamW for auxiliary groups.

    Same use_muon group flag as the existing pretraining entry point. No
    approximation or spectrum logging. Momentum is fp32 and saved in state_dict.
    """
    def __init__(self, param_groups, method, *, momentum=0.95, steps=5,
                 threshold=0.01, target='bottom', exponent=2/3,
                 relative_floor=1e-6, truncate_fraction=0.05):
        if method not in ('targeted_muon', 'kaon', 'freon', 'truncated_sgd'):
            raise ValueError(f'Unknown spectral method: {method}')
        if not 0 <= momentum < 1 or steps < 1:
            raise ValueError('Require 0 <= momentum < 1 and steps >= 1')
        if not math.isfinite(threshold) or threshold < 0 or target not in ('top', 'bottom'):
            raise ValueError('Require finite threshold >= 0 and target top/bottom')
        if not 0 <= exponent <= 1 or not 0 < relative_floor < 1:
            raise ValueError('Require 0 <= Freon exponent <= 1 and 0 < relative_floor < 1')
        if not 0 <= truncate_fraction < 1:
            raise ValueError('truncate_fraction must be in [0, 1)')
        defaults = dict(lr=0.02, weight_decay=0., momentum=momentum,
                        betas=(0.9, 0.95), eps=1e-8, use_muon=True,
                        method=method, steps=steps, threshold=threshold, target=target,
                        exponent=exponent, relative_floor=relative_floor,
                        truncate_fraction=truncate_fraction)
        super().__init__(param_groups, defaults)

    def _momentum_input(self, parameter, group):
        grad = parameter.grad.detach().float()
        if grad.is_sparse:
            raise RuntimeError('Sparse gradients are not supported')
        state = self.state[parameter]
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(parameter, dtype=torch.float32)
        state['momentum_buffer'] = state['momentum_buffer'].float()
        momentum = state['momentum_buffer']
        # Unnormalized Nesterov convention in targeted Muon's Algorithm 1.
        # Use it consistently for all new methods; normalized spectral methods
        # are invariant to a positive scalar multiplication of the input.
        momentum.mul_(group['momentum']).add_(grad)
        update = grad + group['momentum'] * momentum
        return update

    def _spectral_update(self, parameter, group, update):
        matrix = update.reshape(update.shape[0], -1)
        method = group['method']
        if method == 'targeted_muon':
            output = targeted_muon_transform(matrix, group['threshold'], group['target'], group['steps'])
        elif method == 'kaon':
            output = kaon_transform(matrix, group['steps'])
        elif method == 'freon':
            output = freon_svd_transform(matrix, group['exponent'], group['relative_floor'])
        else:
            output = truncated_sgd_transform(matrix, group['truncate_fraction'])
        # Retain the repository's aspect-ratio LR convention for normalized
        # spectral updates; targeted Muon/SGD follow raw Algorithm 1 scales.
        if method in ('kaon', 'freon'):
            output *= max(1, matrix.shape[0] / matrix.shape[1]) ** 0.5
        return output.reshape(parameter.shape)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        world = dist.get_world_size() if distributed else 1
        for group in self.param_groups:
            if group['use_muon']:
                for index, parameter in enumerate(group['params']):
                    owner = index % world
                    # Maintain all momentum states on every rank so the existing
                    # rank-0-only checkpoint contains the complete optimizer.
                    update = self._momentum_input(parameter, group) if parameter.grad is not None else None
                    if rank == owner and update is not None:
                        output = self._spectral_update(parameter, group, update)
                        parameter.mul_(1 - group['lr'] * group['weight_decay'])
                        parameter.add_(output.to(parameter.dtype), alpha=-group['lr'])
                    if distributed:
                        dist.broadcast(parameter, src=owner)
            else:
                for parameter in group['params']:
                    if parameter.grad is None:
                        continue
                    grad = parameter.grad.detach().float()
                    state = self.state[parameter]
                    if not state:
                        state.update(step=0, exp_avg=torch.zeros_like(grad), exp_avg_sq=torch.zeros_like(grad))
                    state['step'] += 1
                    state['exp_avg'] = state['exp_avg'].float()
                    state['exp_avg_sq'] = state['exp_avg_sq'].float()
                    b1, b2 = group['betas']
                    state['exp_avg'].lerp_(grad, 1 - b1)
                    state['exp_avg_sq'].lerp_(grad.square(), 1 - b2)
                    corrected = state['exp_avg'] / (1 - b1 ** state['step'])
                    denom = (state['exp_avg_sq'] / (1 - b2 ** state['step'])).sqrt() + group['eps']
                    parameter.mul_(1 - group['lr'] * group['weight_decay'])
                    parameter.add_((corrected / denom).to(parameter.dtype), alpha=-group['lr'])
        return loss
