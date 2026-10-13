"""V2 invariants and exact CPU resume, using tiny synthetic examples only."""
import uuid
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from causal_model import GatedLagConv, TemporalTensorConv, DynamicGraphWaveNet, MultiScaleTemporal
from causal_data import GraphArchive, decode_strings
from train_dynamic import parser, run_mode, load_checkpoint


class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

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
                                    multiscale=False, graph_history_fusion=args.graph_history_fusion)
        self.assertIsInstance(model.multiscale, torch.nn.Identity)
        self.assertEqual(model.graphs[0].history_fusion, 'signed_conv')


if __name__=='__main__': unittest.main(verbosity=2)
