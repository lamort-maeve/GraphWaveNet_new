"""Small CPU correctness tests; no full training."""
import unittest
import numpy as np
import torch

from causal_data import GraphArchive
from causal_model import lag_message, DynamicGraphWaveNet
from train_dynamic import Metrics


class Correctness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_direction_sign_and_lag(self):
        x = torch.zeros(1, 1, 2, 5)
        x[0, 0, 0, 1] = 3
        a = torch.zeros(1, 5, 2, 2)
        a[:, :, 0, 1] = -2
        expected = torch.zeros_like(x)
        expected[0, 0, 1, 3] = -6
        torch.testing.assert_close(lag_message(x, a, 2), expected)
        self.assertEqual(lag_message(x, a, 5).abs().sum(), 0)

    def test_asof_missing_and_future(self):
        archive = GraphArchive.__new__(GraphArchive)
        archive.times = np.array([10, 20, 40])
        np.testing.assert_array_equal(archive.ids(np.array([-1, 9, 10, 19, 20, 39, 40])),
                                      [0, 0, 1, 1, 2, 2, 3])

    def test_modes_gradients_singleton_and_permutation(self):
        rng = np.random.default_rng(9)
        bank = rng.normal(size=(4, 3, 4, 4)).astype(np.float32)
        bank[0] = 0
        x = torch.randn(3, 1, 4, 5)
        ids = torch.tensor([[1]*5, [2]*5, [3]*5])
        for mode in ('causal', 'adaptive', 'combined'):
            torch.manual_seed(4)
            model = DynamicGraphWaveNet(4, 1, [0, 2], bank, mode, hidden=4, blocks=1, layers=2, dropout=0)
            output = model(x, ids)
            self.assertEqual(output.shape, (3, 1, 2))
            output.square().mean().backward()
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
            model.eval()
            perm = torch.tensor([2, 0, 1])
            torch.testing.assert_close(model(x[perm], ids[perm]), model(x, ids)[perm])
            self.assertEqual(model(x[:1], ids[:1]).shape, (1, 1, 2))
            altered = model(x, torch.zeros_like(ids))
            if mode == 'adaptive':
                torch.testing.assert_close(altered, model(x, ids))
            else:
                self.assertGreater((altered-model(x, ids)).abs().max().item(), 1e-7)
            if mode == 'causal':
                self.assertFalse(any('nodevec' in n for n, _ in model.named_parameters()))

    def test_future_input_cannot_change_past_message(self):
        x = torch.randn(1, 2, 3, 6)
        a = torch.randn(1, 6, 3, 3)
        changed = x.clone(); changed[..., 4:] += 100
        for lag in range(4):
            torch.testing.assert_close(lag_message(x, a, lag)[..., :4],
                                       lag_message(changed, a, lag)[..., :4])

    def test_every_lag_is_used(self):
        x = torch.zeros(1, 1, 2, 13)
        x[0, 0, 0, :] = torch.arange(1., 14.)
        a = torch.zeros(1, 13, 2, 2); a[..., 0, 1] = 1
        for lag in range(13):
            self.assertEqual(lag_message(x, a, lag)[0, 0, 1, -1], 13-lag)

    def test_metrics_zero_targets_and_uneven_batches(self):
        pred = torch.tensor([[[2., 5.]], [[4., 100.]], [[6., 8.]]])
        truth = torch.tensor([[[0., 4.]], [[2., 0.]], [[3., 4.]]])
        mask = torch.tensor([[[True, True]], [[True, False]], [[True, True]]])
        whole, batches = Metrics(1), Metrics(1)
        whole.update(pred, truth, mask)
        batches.update(pred[:2], truth[:2], mask[:2])
        batches.update(pred[2:], truth[2:], mask[2:])
        np.testing.assert_allclose(whole.sums, batches.sums)
        result = whole.result()['overall']
        self.assertEqual(result['valid_count'], 5)
        self.assertEqual(result['mape_count'], 4)
        self.assertAlmostEqual(result['mae'], 12/5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
