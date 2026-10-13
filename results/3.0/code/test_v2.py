"""V2 invariants and exact CPU resume, using tiny synthetic examples only."""
import uuid
import copy
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from causal_model import (GatedLagConv, TemporalTensorConv, DynamicGraphWaveNet,
                          MultiScaleTemporal, PCMCIRefinedConv, CausalReadout)
from causal_data import GraphArchive, decode_strings, pack_cache, unpack_cache, Samples
from train_dynamic import parser, run_mode, load_checkpoint, make_loader, loader_rng, masked_mae, pass_epoch


class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_fixed_shape_loss_matches_indexed_loss_and_gradient(self):
        truth = torch.randn(3, 2, 4)
        valid = torch.rand(3, 2, 4) > 0.4
        a = torch.randn_like(truth, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        fast = masked_mae(a, truth, valid)
        reference = (b-truth).abs()[valid].mean()
        torch.testing.assert_close(fast, reference)
        fast.backward(); reference.backward()
        torch.testing.assert_close(a.grad, b.grad)
        self.assertEqual(masked_mae(a, truth, torch.zeros_like(valid)).item(), 0)

    def test_shared_history_matches_separate_layers_and_gradients(self):
        from unittest.mock import patch
        bank = np.random.default_rng(4).normal(size=(5, 3, 4, 4)).astype('float32')
        bank[0] = 0
        for mode in ('causal', 'combined'):
            fast = DynamicGraphWaveNet(4, 2, [0, 2], bank, mode, hidden=4,
                                       blocks=2, layers=2, context=5, dropout=0)
            slow = copy.deepcopy(fast)
            original = TemporalTensorConv.signed_adjacency
            def no_sharing(module, bank_lag, ids, lag, prepared):
                return original(module, bank_lag, ids, lag, module.prepare_history(ids))
            x = torch.randn(2, 1, 4, 5, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            ids = torch.tensor([[0, 1, 2, 2, 3], [1, 1, 2, 3, 4]])
            a = fast(x, ids)
            unique, inverse = np.unique(ids.numpy(), return_inverse=True)
            lookup = (torch.from_numpy(unique), torch.from_numpy(inverse.reshape(-1)))
            torch.testing.assert_close(a, fast(x, ids, graph_lookup=lookup), rtol=0, atol=0)
            with patch.object(TemporalTensorConv, 'signed_adjacency', no_sharing):
                b = slow(y, ids)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            a.square().sum().backward(); b.square().sum().backward()
            torch.testing.assert_close(x.grad, y.grad)
            for name, p in fast.named_parameters():
                torch.testing.assert_close(p.grad, dict(slow.named_parameters())[name].grad)
            # A fresh forward after an update must not reuse old mixed weights.
            with torch.no_grad():
                fast.graphs[0].graph_conv.weight.add_(torch.randn_like(fast.graphs[0].graph_conv.weight))
            c = fast(x.detach(), ids)
            with patch.object(TemporalTensorConv, 'signed_adjacency', no_sharing):
                d = fast(x.detach(), ids)
            torch.testing.assert_close(c, d)

    def test_pass_rejects_nonfinite_at_boundary_and_skips_empty_targets(self):
        from torch.utils.data import DataLoader, TensorDataset
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(1.))
            def forward(self, x, ids):
                return x * self.weight
        model = Tiny()
        valid = torch.tensor([[[False]], [[True]]])
        dataset = TensorDataset(torch.ones(2, 1, 1), torch.zeros(2, 1, 1), valid,
                                torch.zeros(2, 1, dtype=torch.long), torch.arange(2))
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        metrics, _ = pass_epoch(model, DataLoader(dataset, batch_size=1), torch.device('cpu'), 0., 1., opt)
        self.assertEqual(metrics['overall']['valid_count'], 1)
        self.assertAlmostEqual(model.weight.item(), 0.9, places=6)
        with torch.no_grad(): model.weight.fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            pass_epoch(model, DataLoader(dataset, batch_size=1), torch.device('cpu'), 0., 1.)

    def test_fast_graph_matches_previous_outputs_and_all_gradients(self):
        for fusion in ('signed_convex', 'signed_conv'):
            for cls in (TemporalTensorConv, PCMCIRefinedConv):
                fast = (cls(3, 3, True, False, 0, num_nodes=4, history_fusion=fusion)
                        if cls is TemporalTensorConv else cls(3, 3, 0, 4, history_fusion=fusion))
                slow = copy.deepcopy(fast)
                def old_adjacency(bank_lag, ids, lag, prepared):
                    if fusion == 'signed_convex':
                        return (slow.evolved_adjacency(bank_lag.clamp_min(0), ids, lag),
                                slow.evolved_adjacency((-bank_lag).clamp_min(0), ids, lag))
                    a = slow.evolved_adjacency(bank_lag, ids, lag)
                    return a.clamp_min(0), (-a).clamp_min(0)
                slow.signed_adjacency = old_adjacency
                bank = torch.randn(6, 3, 4, 4)
                bank[0] = 0
                ids = torch.tensor([[0, 1, 2, 2, 3], [2, 2, 3, 4, 5]])
                x = torch.randn(2, 3, 4, 5, requires_grad=True)
                y = x.detach().clone().requires_grad_()
                a, b = fast(x, bank, ids, None), slow(y, bank, ids, None)
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
                a.square().sum().backward(); b.square().sum().backward()
                torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=1e-5)
                for name, p in fast.named_parameters():
                    torch.testing.assert_close(p.grad, dict(slow.named_parameters())[name].grad,
                                               rtol=1e-4, atol=1e-5)

    def test_cache_roundtrip_without_pickle(self):
        state = dict(values=np.arange(12, dtype=np.float32).reshape(3, 4),
                     names=np.array(['flow', 'speed']), nested=dict(mask=np.array([True, False])),
                     metadata=dict(context=25, missing=[3, 7]))
        stream = io.BytesIO()
        np.savez(stream, **pack_cache(state))
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as z:
            restored = unpack_cache(z)
        for key in ('values', 'names'):
            np.testing.assert_array_equal(restored[key], state[key])
        np.testing.assert_array_equal(restored['nested']['mask'], state['nested']['mask'])
        self.assertEqual(restored['metadata'], state['metadata'])

    def test_precomputed_samples_match_previous_timeline_lookup(self):
        data = SimpleNamespace(values=np.arange(40, dtype=np.float32).reshape(10, 4), context=5,
                               splits={'train': np.array([1, 4, 8])},
                               targets={'train': np.ones((3, 1, 2), dtype=np.float32)},
                               masks={'train': np.ones((3, 1, 2), dtype=bool)})
        archive = GraphArchive.__new__(GraphArchive)
        archive.times = np.array([2, 6])
        samples = Samples(data, archive, 'train')
        self.assertFalse(hasattr(samples, 'archive'))
        for i, origin in enumerate(data.splits['train']):
            ts = origin + np.arange(-4, 1)
            expected = data.values[np.maximum(ts, 0)].copy()
            expected[ts < 0] = 0
            x, _, _, ids, _ = samples[i]
            np.testing.assert_array_equal(x.numpy(), expected.T[None])
            np.testing.assert_array_equal(ids.numpy(), archive.ids(ts))

    def test_persistent_workers_resume_sample_order(self):
        data = SimpleNamespace(values=np.arange(80, dtype=np.float32).reshape(20, 4), context=3,
                               splits={'train': np.arange(3, 11)},
                               targets={'train': np.ones((8, 1, 2), dtype=np.float32)},
                               masks={'train': np.ones((8, 1, 2), dtype=bool)})
        archive = GraphArchive.__new__(GraphArchive)
        archive.times = np.array([2, 6])
        args = parser().parse_args(['--workers', '2', '--batch-size', '2'])
        first = make_loader(data, archive, 'train', args)
        resumed = None
        try:
            list(first)
            state = loader_rng(first).get_state()
            expected = torch.cat([batch[-1] for batch in first])
            resumed = make_loader(data, archive, 'train', args)
            loader_rng(resumed).set_state(state)
            actual = torch.cat([batch[-1] for batch in resumed])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(first.persistent_workers)
        finally:
            for loader in (first, resumed):
                if loader is not None and loader._iterator is not None:
                    loader._iterator._shutdown_workers()

    def test_empty_graph_adds_no_causal_signal(self):
        op=GatedLagConv(3,4,True,False,0,4)
        x=torch.randn(2,3,4,7); bank=torch.zeros(2,4,4,4);ids=torch.ones(2,7,dtype=torch.long)
        torch.testing.assert_close(op(x,bank,ids,None),op.local(x))

    def test_causal_operator_has_no_future_messages(self):
        op=GatedLagConv(3,4,True,False,0,4).eval()
        x=torch.randn(2,3,4,7);bank=torch.randn(2,4,4,4);ids=torch.ones(2,7,dtype=torch.long)
        changed=x.clone();changed[...,4:]+=100
        torch.testing.assert_close(op(x,bank,ids,None)[...,:4],op(changed,bank,ids,None)[...,:4])

    def test_lag_sign_channels_do_not_cancel_before_projection(self):
        op=GatedLagConv(1,2,True,False,0,2)
        with torch.no_grad():
            op.lag_project.weight.zero_()
            op.lag_project.weight[0,0]=2  # positive lag zero
            op.lag_project.weight[0,3]=3  # negative lag one
        bank=torch.zeros(2,2,2,2)
        bank[1,0,0,1]=1; bank[1,1,0,1]=-1
        x=torch.ones(1,1,2,4);ids=torch.ones(1,4,dtype=torch.long)
        result=op.causal_message(x,bank,ids)
        self.assertAlmostEqual(result[0,0,1,-1].item(),-1)

    def test_exact_persistence_when_corrections_zero(self):
        model=DynamicGraphWaveNet(4,2,[0,2],np.zeros((2,3,4,4),np.float32),hidden=4,blocks=1,layers=1,context=5)
        with torch.no_grad(): model.end2.weight.zero_();model.end2.bias.zero_()
        x=torch.randn(1,1,4,5);ids=torch.zeros(1,5,dtype=torch.long)
        torch.testing.assert_close(model(x,ids),x[:,0,[0,2],-1][:,None,:].expand(-1,2,-1))

    def test_byte_metadata(self):
        self.assertEqual(decode_strings([b'flow',b'speed']).tolist(),['flow','speed'])

    def test_zero_graph_cannot_mix_unconnected_variables(self):
        model=DynamicGraphWaveNet(4,1,[0,2],np.zeros((2,3,4,4),np.float32),
                                 hidden=4,blocks=2,layers=2,context=5,dropout=0).eval()
        x=torch.randn(1,1,4,5);ids=torch.ones(1,5,dtype=torch.long)
        changed=x.clone();changed[:,:,1,:]+=100
        torch.testing.assert_close(model(x,ids),model(changed,ids))

    def test_receptive_field_covers_default_context(self):
        model=DynamicGraphWaveNet(4,1,[0,2],np.zeros((1,13,4,4),np.float32),
                                 hidden=4,blocks=4,layers=2,context=25,dropout=0)
        self.assertEqual(model.temporal_kernel,3)
        self.assertGreaterEqual(model.receptive_field,25)
        y=model(torch.randn(2,1,4,25),torch.zeros(2,25,dtype=torch.long))
        self.assertEqual(tuple(y.shape),(2,1,2))

    def test_widened_temporal_convolution_ignores_future(self):
        model=DynamicGraphWaveNet(4,1,[0,2],np.zeros((1,3,4,4),np.float32),
                                 hidden=4,blocks=1,layers=2,context=8,dropout=0).eval()
        self.assertGreater(model.temporal_kernel,2)
        x=torch.randn(1,4,4,8)
        changed=x.clone();changed[...,5:]+=10
        dilation,kernel=model.dilations[0],model.temporal_kernel
        def run(value):
            padded=torch.nn.functional.pad(value,(dilation*(kernel-1),0))
            return model.filters[0](padded)
        torch.testing.assert_close(run(x)[...,:5],run(changed)[...,:5])

    def test_temporal_tensor_uses_older_graphs_and_ignores_future_ids(self):
        op=TemporalTensorConv(1,1,True,False,0,kernel_size=3,history_fusion='signed_conv').eval()
        with torch.no_grad():
            op.graph_conv.weight.zero_()
            op.graph_conv.weight[0,0,0]=1
            op.lag_project.weight.zero_()
            op.lag_project.weight[0,0]=1
        bank=torch.zeros(4,1,2,2)
        bank[1,0,0,1]=5
        x=torch.zeros(1,1,2,4); x[0,0,0,-1]=2
        ids=torch.full((1,4),3)
        self.assertAlmostEqual(op.causal_message(x,bank,ids)[0,0,1,-1].item(),10)
        self.assertEqual(op.evolved_adjacency(bank[:,0],ids,0)[0,-1,1,0].item(),0)
        future=ids.clone(); future[:,2:]=1
        torch.testing.assert_close(op.causal_message(x,bank,ids)[...,:2], op.causal_message(x,bank,future)[...,:2])
        with torch.no_grad():
            op.graph_conv.weight.zero_(); op.graph_conv.weight[0,0,-1]=1
        bank[3,0,0,1]=7
        self.assertAlmostEqual(op.evolved_adjacency(bank[:,0],ids,0)[0,-1,0,1].item(),7)
        empty=TemporalTensorConv(2,2,True,False,0)
        self.assertEqual(empty.causal_message(torch.randn(2,2,3,5), torch.zeros(2,2,3,3), torch.ones(2,5,dtype=torch.long)).abs().sum().item(),0)

    def test_resume_matches_uninterrupted_training(self):
        rng=np.random.default_rng(5)
        data=SimpleNamespace(values=rng.normal(size=(40,4)).astype('float32'),context=5,horizon=1,
                             flow_indices=np.array([0,2]), output_mean=np.zeros(2,np.float32),output_std=np.ones(2,np.float32),
                             mean=np.zeros(4,np.float32),std=np.ones(4,np.float32), y_offsets=np.array([1]))
        data.splits={s:np.arange(a,a+4) for s,a in [('train',6),('val',16),('test',26)]}
        data.targets={s:data.values[t+1][:,[0,2]][:,None,:] for s,t in data.splits.items()}
        data.masks={s:np.ones_like(y,dtype=bool) for s,y in data.targets.items()}
        archive=GraphArchive.__new__(GraphArchive);archive.times=np.array([5]);archive.bank=rng.normal(size=(2,3,4,4)).astype('float32');archive.bank[0]=0
        args=parser().parse_args(['--hidden','4','--blocks','1','--layers','1','--context','5','--batch-size','2',
                                  '--epochs','2','--patience','0','--dropout','0.2'])
        test_root=Path(__file__).resolve().parents[1]/'checks'/'unit_temp'
        test_root.mkdir(parents=True,exist_ok=True)
        with self.subTest('resume checkpoint equivalence'):
            directory=test_root/uuid.uuid4().hex
            directory.mkdir()
            assert directory.resolve().parent == test_root.resolve()
            run_mode(args,data,archive,directory/'full','causal')
            args.epochs=1
            run_mode(args,data,archive,directory/'resume','causal')
            args.epochs=2;args.resume=True
            run_mode(args,data,archive,directory/'resume','causal')
            full=load_checkpoint(directory/'full/causal/last.pt','cpu')
            resumed=load_checkpoint(directory/'resume/causal/last.pt','cpu')
            for name,value in full['model'].items():
                torch.testing.assert_close(value,resumed['model'][name],rtol=0,atol=0)

    def test_deduplicated_tensor_matches_explicit_convolution_and_gradient(self):
        op = TemporalTensorConv(1, 2, True, False, 0, kernel_size=3, history_fusion='signed_conv')
        bank = torch.randn(6, 3, 3)
        bank[0] = 0
        ids = torch.tensor([[0, 1, 1, 3, 3], [2, 2, 4, 4, 4]])
        with torch.no_grad():
            op.graph_conv.weight.normal_()
        actual = op.evolved_adjacency(bank, ids, 1)
        history = torch.stack([bank[(ids-age).clamp_min(0)] for age in (2, 1, 0)], -1)
        expected = (history * op.graph_conv.weight[1, 0]).sum(-1)
        torch.testing.assert_close(actual, expected)
        actual_grad = torch.autograd.grad(actual.square().sum(), op.graph_conv.weight, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected.square().sum(), op.graph_conv.weight)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        changed = bank.clone()
        changed[5] += 100  # not yet available for any input timestamp
        torch.testing.assert_close(actual, op.evolved_adjacency(changed, ids, 1))

    def test_temporal_tensor_gate_and_mode_isolation(self):
        op = TemporalTensorConv(2, 3, True, False, 0, num_nodes=4)
        x = torch.randn(2, 2, 4, 5)
        ids = torch.ones(2, 5, dtype=torch.long)
        torch.testing.assert_close(op(x, torch.zeros(2, 3, 4, 4), ids, None), op.local(x))
        for mode in ('causal', 'adaptive', 'combined'):
            model = DynamicGraphWaveNet(4, 2, [0, 2], np.zeros((2, 3, 4, 4), np.float32),
                                        mode, hidden=4, blocks=1, layers=1, context=5, graph_kernel=5)
            self.assertEqual(model.graph_kernel, 5)
            self.assertEqual(hasattr(model, 'nodevec1'), mode != 'causal')
            self.assertEqual(hasattr(model.graphs[0], 'graph_conv'), mode != 'adaptive')
            self.assertEqual(tuple(model(torch.randn(2, 1, 4, 5), ids).shape), (2, 2, 2))

    def test_multiscale_has_no_future_or_cross_node_access(self):
        module = MultiScaleTemporal(3).eval()
        x = torch.randn(2, 3, 4, 25)
        with torch.no_grad():
            module.project.weight.normal_()  # test a learned, nontrivial branch
        changed = x.clone()
        changed[..., 12:] += 100
        torch.testing.assert_close(module(x)[..., :12], module(changed)[..., :12])
        changed = x.clone()
        changed[:, :, 1] += 100
        torch.testing.assert_close(module(x)[:, :, 0], module(changed)[:, :, 0])
        self.assertGreater((module(x) - x).abs().max().item(), 1e-5)
        module(x).square().mean().backward()
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_convex_history_keeps_opposite_sign_messages_separate(self):
        op = TemporalTensorConv(1, 1, True, False, 0, kernel_size=2)
        with torch.no_grad():
            op.graph_conv.weight.zero_()  # equal contributions from two snapshots
            op.lag_project.weight[0, 0] = 2
            op.lag_project.weight[0, 1] = 3
        bank = torch.zeros(3, 1, 2, 2)
        bank[1, 0, 0, 1] = 1
        bank[2, 0, 0, 1] = -1
        ids = torch.full((1, 4), 2)
        x = torch.ones(1, 1, 2, 4)
        actual = op.causal_message(x, bank, ids)
        torch.testing.assert_close(actual[0, 0, 1], torch.full((4,), -0.5))
        actual.sum().backward()
        self.assertGreater(op.graph_conv.weight.grad.abs().sum().item(), 0)

    def test_convex_history_preserves_bound_and_ignores_future(self):
        op = TemporalTensorConv(1, 1, True, False, 0, kernel_size=3)
        bank = torch.rand(5, 2, 2)
        bank[0] = 0
        ids = torch.tensor([[0, 1, 2, 3]])
        adjacency = op.evolved_adjacency(bank, ids, 0)
        self.assertGreaterEqual(adjacency.min().item(), 0)
        self.assertLessEqual(adjacency.max().item(), bank.max().item())
        self.assertEqual(adjacency[:, 0].abs().sum().item(), 0)
        bank[4] += 1000
        torch.testing.assert_close(adjacency, op.evolved_adjacency(bank, ids, 0))

    def test_previous_configuration_is_selectable(self):
        args = parser().parse_args(['--no-multiscale', '--graph-history-fusion', 'signed_conv'])
        self.assertTrue(args.no_multiscale)
        model = DynamicGraphWaveNet(4, 1, [0, 2], np.zeros((2, 3, 4, 4), np.float32),
                                    hidden=4, blocks=1, layers=1, context=5,
                                    multiscale=False, graph_history_fusion=args.graph_history_fusion,
                                    pcmci_refinement=False, causal_readout=False)
        self.assertIsInstance(model.multiscale, torch.nn.Identity)
        self.assertEqual(model.graphs[0].history_fusion, 'signed_conv')

    def test_refined_pcmci_does_not_create_edges_or_read_future(self):
        op = PCMCIRefinedConv(3, 3, 0., 4)
        bank = torch.zeros(3, 3, 4, 4)
        bank[1:, 2, 0, 1] = 0.2  # only source 0 -> destination 1 at lag 2
        x = torch.randn(2, 3, 4, 7)
        ids = torch.ones(2, 7, dtype=torch.long)
        message = op.causal_message(x, bank, ids)
        self.assertEqual(message[:, :, [0, 2, 3]].abs().sum().item(), 0)
        self.assertEqual(message[..., :2].abs().sum().item(), 0)
        changed = x.clone()
        changed[..., 4:] += 20
        torch.testing.assert_close(message[..., :4], op.causal_message(changed, bank, ids)[..., :4])
        changed = x.clone()
        changed[:, :, 3] += 20  # unrelated variable cannot affect selection
        torch.testing.assert_close(message, op.causal_message(changed, bank, ids))
        future_bank = bank.clone()
        future_bank[2] += 100
        torch.testing.assert_close(message, op.causal_message(x, future_bank, ids))
        message.square().sum().backward()
        for name in ('query', 'key', 'lag_select'):
            grad = getattr(op, name).weight.grad
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(grad.abs().sum().item(), 0, name)

    def test_causal_readout_retains_long_lag_and_zero_graph_is_zero(self):
        module = CausalReadout(2, 13, 1, 3, 1, 'signed_convex')
        bank = torch.zeros(2, 13, 3, 3)
        bank[1, 12, 0, 1] = 0.3
        x = torch.randn(1, 2, 3, 13, requires_grad=True)
        ids = torch.ones(1, 13, dtype=torch.long)
        output = module(x, bank, ids)
        self.assertEqual(output[:, :, [0, 2]].abs().sum().item(), 0)
        output.sum().backward()
        self.assertGreater(x.grad[:, :, 0, 0].abs().sum().item(), 0)
        self.assertEqual(module(x, torch.zeros_like(bank), ids).abs().sum().item(), 0)

    def test_adaptive_baseline_identical_with_pcmci_switches(self):
        bank = np.random.default_rng(4).normal(size=(3, 3, 4, 4)).astype('float32')
        models = []
        for enabled in (False, True):
            torch.manual_seed(42)
            models.append(DynamicGraphWaveNet(4, 1, [0, 2], bank, 'adaptive',
                          hidden=4, blocks=1, layers=2, context=5, dropout=0,
                          pcmci_refinement=enabled, causal_readout=enabled))
        self.assertEqual(set(models[0].state_dict()), set(models[1].state_dict()))
        for name, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[name], rtol=0, atol=0)
        x = torch.randn(2, 1, 4, 5)
        ids = torch.ones(2, 5, dtype=torch.long)
        torch.testing.assert_close(models[0](x, ids), models[1](x, ids), rtol=0, atol=0)
        self.assertIsNone(models[1].causal_readout)


if __name__=='__main__': unittest.main(verbosity=2)
