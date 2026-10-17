from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        data, x_offsets, y_offsets, add_time_in_day=False, add_day_in_week=False
):
    """
    修改为直接处理numpy数组，不添加时间特征
    """
    num_samples, num_nodes = data.shape

    # 增加特征维度
    data = np.expand_dims(data, axis=-1)  # (T, N, 1)

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))

    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])  # (seq_len, N, 1)
        y.append(data[t + y_offsets, ...])  # (pred_len, N, 1)

    x = np.stack(x, axis=0)  # (样本数, seq_len, N, 1)
    y = np.stack(y, axis=0)  # (样本数, pred_len, N, 1)
    return x, y


def generate_train_val_test(args):
    # ============================
    # 加载你的 .npy 数据
    # ============================
    data = np.load(args.traffic_df_filename)
    print(f"原始数据形状: {data.shape}")  # (500, 10)

    seq_length_x = args.seq_length_x
    seq_length_y = args.seq_length_y

    # ============================
    # 构建偏移量
    # ============================
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))

    # ============================
    # 生成数据（不添加时间特征）
    # ============================
    x, y = generate_graph_seq2seq_io_data(
        data,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,
        add_day_in_week=False,
    )

    print(f"x shape: {x.shape}, y shape: {y.shape}")

    # ============================
    # 划分训练/验证/测试集
    # ============================
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train:num_train + num_val], y[num_train:num_train + num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]

    # ============================
    # 保存为 npz 文件
    # ============================
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(f"{cat} x: {_x.shape}, y: {_y.shape}")

        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,  # (样本数, 12, 10, 1)
            y=_y,  # (样本数, 1, 10, 1)
            x_offsets=x_offsets.reshape(-1, 1),
            y_offsets=y_offsets.reshape(-1, 1),
        )

    print(f"数据已保存到: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/my_dataset",
                        help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str,
                        default="my_pcmci_project/results/traffic_500T_10N.npy",
                        help="Raw traffic data (.npy).")
    parser.add_argument("--seq_length_x", type=int, default=12,
                        help="Input sequence length.")
    parser.add_argument("--seq_length_y", type=int, default=1,
                        help="Output sequence length.")
    parser.add_argument("--y_start", type=int, default=1,
                        help="Y prediction start offset.")
    parser.add_argument("--dow", action='store_true',
                        help="Add day of week feature.")

    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y':
            exit()
    else:
        os.makedirs(args.output_dir)

    generate_train_val_test(args)