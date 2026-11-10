import copy
import unittest
from unittest.mock import patch

import torch

from paper_optimizers import (PaperOptimizerWithAuxAdam, targeted_muon_transform,
                              kaon_transform, freon_svd_transform, truncated_sgd_transform)


class PaperOptimizerTests(unittest.TestCase):
    def test_distributed_owner_keeps_complete_checkpoint_state(self):
        parameters = [torch.nn.Parameter(torch.randn(2, 3)) for _ in range(2)]
        for p in parameters:
            p.grad = torch.randn_like(p)
        optimizer = PaperOptimizerWithAuxAdam([dict(params=parameters)], 'kaon')
        original = [p.detach().clone() for p in parameters]
        with patch('torch.distributed.is_initialized', return_value=True), \
                patch('torch.distributed.get_rank', return_value=0), \
                patch('torch.distributed.get_world_size', return_value=2), \
                patch('torch.distributed.broadcast') as broadcast:
            optimizer.step()
        self.assertEqual(broadcast.call_count, 2)
        self.assertEqual(len(optimizer.state_dict()['state']), 2)
        self.assertFalse(torch.equal(original[0], parameters[0]))
        self.assertTrue(torch.equal(original[1], parameters[1]))

    def test_targeted_modes_and_transpose(self):
        matrix = torch.tensor([[0.2, 0., 0.], [0., 0.8, 0.]])
        for target, values in [('top', (0.2, 1.)), ('bottom', (1., 0.8))]:
            expected = torch.tensor([[values[0], 0., 0.], [0., values[1], 0.]])
            output = targeted_muon_transform(matrix, 0.5, target)
            torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(targeted_muon_transform(matrix.mT, 0.5, target), output.mT)

    def test_kaon_matches_scalar_polynomial(self):
        matrix = torch.diag(torch.tensor([0.2, 0.8]))
        coefficients = matrix.diagonal() / matrix.norm()
        for _ in range(5):
            coefficients = 4.1 * coefficients * (1 - coefficients.square()).square()
        torch.testing.assert_close(kaon_transform(matrix).diagonal(), coefficients / 1.175,
                                   atol=1e-4, rtol=1e-4)

    def test_freon_spectral_definition(self):
        matrix = torch.diag(torch.tensor([4., 1.]))
        for exponent in (0., 0.25, 0.5, 2/3, 0.75, 1.):
            q = 2 - 2 * exponent
            singular = matrix.diagonal()
            mean = singular.log().mean().exp() if q == 0 else singular.pow(q).mean().pow(1/q)
            expected = (singular / mean).pow(q - 1)
            torch.testing.assert_close(freon_svd_transform(matrix, exponent).diagonal(), expected)

    def test_truncation_and_zero_inputs(self):
        matrix = torch.diag(torch.tensor([4., 3., 2., 1.]))
        output = truncated_sgd_transform(matrix, 0.25)
        self.assertEqual(output[0, 0].item(), 0.)
        torch.testing.assert_close(output.norm(), matrix.norm())
        for transform in (targeted_muon_transform, kaon_transform, freon_svd_transform, truncated_sgd_transform):
            output = transform(torch.zeros(3, 5))
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(output.norm().item(), 0.)

    def test_step_aux_adam_and_checkpoint_resume(self):
        torch.manual_seed(4)
        for method in ('targeted_muon', 'kaon', 'freon', 'truncated_sgd'):
            p = torch.nn.Parameter(torch.randn(3, 5))
            bias = torch.nn.Parameter(torch.randn(3))
            adam_bias = torch.nn.Parameter(bias.detach().clone())
            opt = PaperOptimizerWithAuxAdam([
                dict(params=[p], use_muon=True, lr=0.01),
                dict(params=[bias], use_muon=False, lr=0.001)], method)
            adam = torch.optim.AdamW([adam_bias], lr=0.001, betas=(0.9, 0.95), weight_decay=0., eps=1e-8)
            p.grad, bias.grad = torch.randn_like(p), torch.randn_like(bias)
            adam_bias.grad = bias.grad.clone()
            opt.step()
            adam.step()
            torch.testing.assert_close(bias, adam_bias)
            restored_p = torch.nn.Parameter(p.detach().clone())
            restored_bias = torch.nn.Parameter(bias.detach().clone())
            restored = PaperOptimizerWithAuxAdam([
                dict(params=[restored_p], use_muon=True),
                dict(params=[restored_bias], use_muon=False)], method)
            restored.load_state_dict(copy.deepcopy(opt.state_dict()))
            for original, other in ((p, restored_p), (bias, restored_bias)):
                original.grad = torch.randn_like(original)
                other.grad = original.grad.clone()
            opt.step()
            restored.step()
            torch.testing.assert_close(p, restored_p)
            torch.testing.assert_close(bias, restored_bias)
            self.assertTrue(torch.isfinite(p).all())


if __name__ == '__main__':
    unittest.main()
