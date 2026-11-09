import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from approximation_metrics import ApproximationLogger, approximation_metrics
from muon import MuonWithAuxAdam
from soren import MuonWithAuxAdam_sigmoid
from summarize_approximation import summarize


class ApproximationTests(unittest.TestCase):
    def test_global_aggregation_uses_squared_norms(self):
        common = dict(optimizer='muon', step=1, reference='svd_polar',
                      relative_frobenius_error=0.5, max_per_mode_relative_error=0.7,
                      num_modes=2, log_centered_rmse=1., width_exact=0., width_approx=2.)
        rows = [dict(common, exact_frobenius_squared=4., error_frobenius_squared=1.),
                dict(common, exact_frobenius_squared=12., error_frobenius_squared=3.)]
        self.assertIn('| 2 | 50.000 | 50.000 | 70.000 |', summarize(rows))

    def test_exact_references_and_signed_mode_error(self):
        matrix = torch.diag(torch.tensor([2., 1.], dtype=torch.float64))
        for kind, target in [("muon", torch.ones(2, dtype=torch.float64)),
                             ("soren", matrix.diagonal().sigmoid())]:
            row = approximation_metrics(matrix, torch.diag(target), kind)[0]
            self.assertLess(row["relative_frobenius_error"], 1e-12)
            self.assertLess(row["max_per_mode_relative_error"], 1e-12)
        # Identical singular values do not imply the same update: retain signs.
        row = approximation_metrics(matrix, torch.diag(torch.tensor([-1., 1.])), "muon")[0]
        self.assertAlmostEqual(row["max_per_mode_relative_error"], 2.)
        self.assertAlmostEqual(row["log_centered_max_error"], 0.)
        self.assertEqual(row["negative_coefficients"], 1)

    def test_log_width_and_zero_handling(self):
        matrix = torch.diag(torch.tensor([2., 1.]))
        row = approximation_metrics(matrix, torch.diag(torch.tensor([1., 0.25])), "muon")[0]
        self.assertAlmostEqual(row["width_exact"], 0.)
        self.assertAlmostEqual(row["width_approx"], 2.)
        self.assertAlmostEqual(row["log_centered_max_error"], 1.)
        row = approximation_metrics(torch.zeros(2, 3), torch.zeros(2, 3), "muon")[0]
        self.assertEqual(row["log_clamped_modes"], 2)
        self.assertEqual(row["numerical_rank"], 0)
        self.assertAlmostEqual(row["relative_frobenius_error"], 1.)
        json.dumps(row, allow_nan=False)

    def test_optimizer_logging_preserves_updates_and_momentum(self):
        torch.manual_seed(17)
        # Distributed optimizers own one parameter per rank; emulate one rank
        # here to exercise the exact class used by the pretraining entry point.
        with tempfile.TemporaryDirectory() as directory, \
                patch("torch.distributed.get_world_size", return_value=1), \
                patch("torch.distributed.get_rank", return_value=0), \
                patch("torch.distributed.all_gather"):
            for kind, optimizer_cls in [("soren", MuonWithAuxAdam_sigmoid), ("muon", MuonWithAuxAdam)]:
                p = torch.nn.Parameter(torch.randn(3, 5))
                q = torch.nn.Parameter(p.detach().clone())
                baseline = optimizer_cls([dict(params=[p], use_muon=True)])
                measured = optimizer_cls([dict(params=[q], use_muon=True)])
                diagnostic = ApproximationLogger(directory, [("model.layers.7.weight", q)],
                                                 kind, every=2, steps={1},
                                                 parameter_regex=r"layers\.7\.")
                measured.approximation_logger = diagnostic
                for step in (1, 2, 3):
                    grad = torch.randn_like(p)
                    p.grad, q.grad = grad.clone(), grad.clone()
                    diagnostic.step = step
                    baseline.step()
                    measured.step()
                    diagnostic.flush_summary()
                    self.assertTrue(torch.equal(p, q))
                    self.assertTrue(torch.equal(baseline.state[p]["momentum_buffer"],
                                                measured.state[q]["momentum_buffer"]))
                rows = [json.loads(line) for line in Path(diagnostic.path).read_text().splitlines()]
                self.assertEqual({row["step"] for row in rows}, {1, 2})
                self.assertEqual(len(rows), 4 if kind == "soren" else 2)
                self.assertEqual(rows[0]["shape"], [3, 5])
                summary = (Path(directory) / f'{kind}_approximation_summary.txt').read_text()
                self.assertIn('Global rel. F (%)', summary)
                self.assertNotIn('N/A', summary)
                diagnostic.step = 4
                diagnostic.pattern = re.compile("missing_parameter")
                self.assertIsNone(diagnostic.callback(q))


if __name__ == "__main__":
    unittest.main()
