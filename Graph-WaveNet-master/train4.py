import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='')
parser.add_argument('--data', type=str, default='data/METR-LA',
                    help='data path')  # 修改为你的数据路径
parser.add_argument('--adjdata', type=str, default='data/results/adj_mx4.pkl',
                    help='adj data path')  # 使用因果邻接矩阵
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=10, help='number of nodes')  # 改为10
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--save', type=str, default='./garage/metr', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

args = parser.parse_args()


def load_my_data(data_dir):
    """
    加载自定义格式的数据
    数据格式: (batch, time, node, feature)
    """
    print("=" * 60)
    print("加载数据")
    print("=" * 60)

    # 加载三个数据集
    train_data = np.load(f"{data_dir}/train.npz")
    val_data = np.load(f"{data_dir}/val.npz")
    test_data = np.load(f"{data_dir}/test.npz")

    X_train = train_data['x']  # (341, 12, 10, 1)
    Y_train = train_data['y']  # (341, 1, 10, 1)
    X_val = val_data['x']  # (97, 12, 10, 1)
    Y_val = val_data['y']  # (97, 1, 10, 1)
    X_test = test_data['x']  # (50, 12, 10, 1)
    Y_test = test_data['y']  # (50, 1, 10, 1)

    print(f"训练集: {X_train.shape}, {Y_train.shape}")
    print(f"验证集: {X_val.shape}, {Y_val.shape}")
    print(f"测试集: {X_test.shape}, {Y_test.shape}")

    # 获取节点数
    num_nodes = X_train.shape[2]
    print(f"节点数: {num_nodes}")

    # 创建简单的dataloader包装
    class DataLoader:
        def __init__(self, x, y, batch_size, shuffle=True):
            self.x = x
            self.y = y
            self.batch_size = batch_size
            self.shuffle_enabled = shuffle  # 重命名避免冲突
            self.num_samples = x.shape[0]
            self.indices = np.arange(self.num_samples)
            self.current_idx = 0

        def shuffle(self):
            if self.shuffle_enabled:  # 使用新名称
                np.random.shuffle(self.indices)
            self.current_idx = 0

        def get_iterator(self):
            self.shuffle()
            for i in range(0, self.num_samples, self.batch_size):
                end_idx = min(i + self.batch_size, self.num_samples)
                batch_indices = self.indices[i:end_idx]
                yield self.x[batch_indices], self.y[batch_indices]

    # 返回类似原始格式的字典
    return {
        'train_loader': DataLoader(X_train, Y_train, args.batch_size, shuffle=True),
        'val_loader': DataLoader(X_val, Y_val, args.batch_size, shuffle=False),
        'test_loader': DataLoader(X_test, Y_test, args.batch_size, shuffle=False),
        'scaler': None,  # 如果不需要归一化可以设为None
        'y_test': Y_test
    }


def main():
    device = torch.device(args.device)

    print("=" * 60)
    print("Graph WaveNet 训练")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"节点数: {args.num_nodes}")
    print(f"输入序列长度: {args.seq_length}")
    print(f"批次大小: {args.batch_size}")
    print("=" * 60)

    # ====== 加载邻接矩阵 ======
    print("\n加载邻接矩阵...")
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)

    # 🔥 修改：正确处理adj_mx的格式
    print(f"原始adj_mx类型: {type(adj_mx)}")

    # 情况1: adj_mx是列表（Graph WaveNet的load_adj返回的就是列表）
    if isinstance(adj_mx, list):
        print(f"adj_mx是列表，包含 {len(adj_mx)} 个矩阵")
        # 取第一个矩阵作为主邻接矩阵
        if len(adj_mx) > 0:
            # 检查第一个元素的类型
            first_elem = adj_mx[0]
            if isinstance(first_elem, np.ndarray):
                adj_matrix = first_elem
            else:
                adj_matrix = np.array(first_elem)

            # 如果是3D，取第一个2D
            if adj_matrix.ndim == 3:
                print(f"矩阵是3D，形状: {adj_matrix.shape}，取第一个")
                adj_matrix = adj_matrix[0]

            print(f"使用第一个矩阵，形状: {adj_matrix.shape}")
        else:
            raise ValueError("adj_mx列表为空")
    else:
        # 如果不是列表，直接转换为数组
        adj_matrix = np.array(adj_mx)

    # 确保是2D
    adj_matrix = np.array(adj_matrix)
    if adj_matrix.ndim != 2:
        raise ValueError(f"邻接矩阵必须是2D，但得到 {adj_matrix.ndim}D，形状: {adj_matrix.shape}")

    print(f"最终邻接矩阵形状: {adj_matrix.shape}")

    # 验证节点数匹配
    if adj_matrix.shape[0] != args.num_nodes:
        print(f"⚠️ 警告: 邻接矩阵节点数 {adj_matrix.shape[0]} != 设置节点数 {args.num_nodes}")
        print(f"使用邻接矩阵的实际节点数: {adj_matrix.shape[0]}")
        args.num_nodes = adj_matrix.shape[0]

    # 转换为tensor - 🔥 关键修改：这里应该用单个矩阵，而不是列表
    supports = [
        torch.tensor(adj, dtype=torch.float32, device=device)
        for adj in adj_mx]
    print(f"supports数量: {len(supports)}, 每个形状: {supports[0].shape}")

    # ====== 加载数据 ======
    print("\n加载数据...")
    dataloader = load_my_data(args.data)

    # 如果没有scaler，创建一个简单的
    if dataloader['scaler'] is None:
        class DummyScaler:
            def inverse_transform(self, data):
                return data

        scaler = DummyScaler()
    else:
        scaler = dataloader['scaler']

    # ====== 初始化模型 ======
    print("\n初始化模型...")
    print(f"  in_dim: {args.in_dim}")
    print(f"  num_nodes: {args.num_nodes}")
    print(f"  seq_length: {args.seq_length}")
    print(f"  nhid: {args.nhid}")

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    engine = trainer(
        scaler,
        args.in_dim,
        args.seq_length,
        args.num_nodes,
        args.nhid,
        args.dropout,
        args.learning_rate,
        args.weight_decay,
        device,
        supports,
        args.gcn_bool,
        args.addaptadj,
        adjinit
    )

    print("\n开始训练...")
    print("=" * 60)

    his_loss = []
    val_time = []
    train_time = []

    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()

        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            # x: (batch, time, node, feature)
            # y: (batch, pred_len, node, feature)

            trainx = torch.Tensor(x).to(device)  # (64, 12, 10, 1)
            trainy = torch.Tensor(y).to(device)  # (64, 1, 10, 1)

            # 转置为模型需要的格式: (batch, feature, node, time)
            trainx = trainx.transpose(1, 3)  # (64, 1, 10, 12)
            trainy = trainy.transpose(1, 3)  # (64, 1, 10, 1)

            # 取出目标值 (预测步长)
            trainy = trainy[:, 0, :, :]  # (64, 10, 1)

            # 打印调试信息（第一次迭代）
            if iter == 0 and i == 1:
                print(f"\n数据维度检查:")
                print(f"  输入 x: {trainx.shape}")
                print(f"  目标 y: {trainy.shape}")
                print(f"  期望节点数: {args.num_nodes}")
                print("=" * 60)

            metrics = engine.train(trainx, trainy)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])

            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

        t2 = time.time()
        train_time.append(t2 - t1)

        # ====== 验证 ======
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            testy = testy[:, 0, :, :]
            metrics = engine.eval(testx, testy)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])

        s2 = time.time()
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
              flush=True)

        # 保存模型
        if i % 10 == 0:  # 每10个epoch保存一次
            torch.save(engine.model.state_dict(),
                       args.save + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # ====== 测试 ======
    print("\n开始测试...")
    bestid = np.argmin(his_loss)

    # 加载最佳模型
    model_path = args.save + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"
    try:
        engine.model.load_state_dict(torch.load(model_path))
        print(f"加载最佳模型: {model_path}")
    except:
        print(f"无法加载模型 {model_path}，使用当前模型")

    # 测试
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]  # (batch, node, pred_len)

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)  # (batch, time, node, feature)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("=" * 60)
    print("测试结果:")
    print("=" * 60)

    amae = []
    amape = []
    armse = []

    for i in range(args.seq_length):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'Average over {:d} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(args.seq_length, np.mean(amae), np.mean(amape), np.mean(armse)))

    # 保存最终模型
    torch.save(engine.model.state_dict(),
               args.save + "_exp" + str(args.expid) + "_best_" + str(round(his_loss[bestid], 2)) + ".pth")
    print(f"\n模型已保存: {args.save}_exp{args.expid}_best_{round(his_loss[bestid], 2)}.pth")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("\n总运行时间: {:.4f} 秒".format(t2 - t1))