from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import sys
import pandas as pd
import h5py


def _decode_str_array(arr):
    arr = np.asarray(arr)
    decoded = [
        v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
        for v in arr.ravel()
    ]
    return np.array(decoded).reshape(arr.shape)


def load_traffic_frame(file_path, h5_key, feature):
    try:
        df = pd.read_hdf(file_path, "data")
        print("成功用pandas读取文件")
        return df
    except Exception as e:
        print(f"pandas读取失败，尝试用h5py读取: {e}")

    with h5py.File(file_path, "r") as f:
        print(f"顶层键: {list(f.keys())}")
        available = list(f.keys())
        key = None
        for cand in (h5_key, "data", "data_model_ready"):
            if cand in f:
                key = cand
                break
        if key is None:
            raise KeyError(f"未找到可用数据键，现有键: {available}")

        print(f"使用数据集: {key}")
        data = f[key][:]
        print(f"数据形状: {data.shape}")

        timestamps = _decode_str_array(f["timestamps"][:]) if "timestamps" in f else None
        node_ids = _decode_str_array(f["node_ids"][:]) if "node_ids" in f else None
        feature_names = _decode_str_array(f["feature_names"][:]) if "feature_names" in f else None

        if data.ndim == 3:
            feature_names_list = list(feature_names) if feature_names is not None else None
            if feature_names_list is not None and feature in feature_names_list:
                feat_idx = feature_names_list.index(feature)
            else:
                feat_idx = int(feature)
            feat_name = feature_names_list[feat_idx] if feature_names_list is not None else str(feat_idx)
            print(f"三维数据，使用特征: {feat_name} (index={feat_idx})")
            df_data = data[:, :, feat_idx]
        elif data.ndim == 2:
            df_data = data
        else:
            raise ValueError(f"不支持的数据维度: {data.ndim}")

        if timestamps is not None:
            time_index = pd.to_datetime(timestamps)
        else:
            time_index = pd.date_range(
                start=pd.Timestamp("2018-01-01 00:00:00"),
                periods=df_data.shape[0],
                freq="5min",
            )
        columns = node_ids if node_ids is not None else None
        df = pd.DataFrame(df_data, index=time_index, columns=columns)
        print(f"创建DataFrame形状: {df.shape}")
        return df


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=False, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    print(f"数据节点数: {num_nodes}")

    # 原始数据：增加特征维度 (T, N) -> (T, N, 1)，保持 float32
    data = np.expand_dims(np.asarray(df.values, dtype=np.float32), axis=-1)
    feature_list = [data]

    # ====== 关键修改：关闭时间特征 ======
    # add_time_in_day 和 add_day_in_week 都设为 False
    # 这样特征数就是 1

    # 如果想要时间特征，可以取消注释下面的代码
    # if add_time_in_day:
    #     if isinstance(df.index, pd.DatetimeIndex):
    #         time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    #     else:
    #         try:
    #             datetime_index = pd.to_datetime(df.index.values, unit='ns')
    #             time_ind = (datetime_index - datetime_index.astype("datetime64[D]")) / np.timedelta64(1, "D")
    #         except:
    #             print("警告：无法转换时间索引，使用位置索引")
    #             time_ind = np.arange(len(df)) / (24 * 60 / 5)
    #             time_ind = time_ind % 1
    #     time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
    #     feature_list.append(time_in_day)
    #
    # if add_day_in_week:
    #     if isinstance(df.index, pd.DatetimeIndex):
    #         dow = df.index.dayofweek
    #     else:
    #         try:
    #             datetime_index = pd.to_datetime(df.index.values, unit='ns')
    #             dow = datetime_index.dayofweek
    #         except:
    #             print("警告：无法转换时间索引，使用位置索引计算星期几")
    #             days = np.arange(len(df)) // (24 * 60 / 5)
    #             dow = (days + 0) % 7
    #     dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
    #     feature_list.append(dow_tiled)

    data = np.concatenate(feature_list, axis=-1)
    print(f"合并后数据形状: {data.shape}")  # (时间, 节点, 特征=1)

    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    t_indices = np.arange(min_t, max_t)
    x = data[t_indices[:, None] + x_offsets]
    y = data[t_indices[:, None] + y_offsets]
    return x, y


def generate_train_val_test(args):
    # ====== 关键修改：seq_length_y 设为 1 ======
    seq_length_x = args.seq_length_x  # 12
    seq_length_y = args.seq_length_y  # 1 (从12改为1)

    file_path = os.path.abspath(args.traffic_df_filename)
    print(f"文件路径: {file_path}")
    print(f"seq_length_x: {seq_length_x}")
    print(f"seq_length_y: {seq_length_y}")

    # ====== 读取数据 ======
    df = load_traffic_frame(file_path, args.h5_key, args.feature)

    print(f"最终DataFrame形状: {df.shape}")
    print(f"节点数: {df.shape[1]}")

    # ====== 创建偏移量 ======
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # ====== 关键修改：y_offsets 只预测1步 ======
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # y_offsets = [1] 因为 seq_length_y=1

    print(f"x_offsets: {x_offsets}")  # [-11, -10, ..., 0]
    print(f"y_offsets: {y_offsets}")  # [1]

    # ====== 生成序列数据 ======
    # add_time_in_day=False, add_day_in_week=False
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,  # 关闭时间特征
        add_day_in_week=False,  # 关闭星期特征
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # 期望: x (样本数, 12, 10, 1), y (样本数, 1, 10, 1)

    # ====== 划分数据集 ======
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    # ====== 保存数据 ======
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(f"{cat} x: {_x.shape}, y: {_y.shape}")
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

    print(f"✅ 数据已保存到: {args.output_dir}")
    print(f"✅ 最终格式: (样本, 时间步, {df.shape[1]}节点, 1特征)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/METR-LA", help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default="data/full_year_cleaned.h5", help="Raw traffic readings.", )
    parser.add_argument("--h5_key", type=str, default="data_model_ready", help="HDF5 dataset key.")
    parser.add_argument("--feature", type=str, default="0", help="Feature name or index for 3D data.")
    # ====== 关键修改：seq_length_y 默认改为 1 ======
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence Length.", )
    parser.add_argument("--seq_length_y", type=int, default=1, help="Sequence Length.", )  # 12 -> 1
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--dow", action='store_true', )

    args = parser.parse_args()

    print("=" * 50)
    print("参数配置:")
    print(f"  traffic_df_filename: {args.traffic_df_filename}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  h5_key: {args.h5_key}")
    print(f"  feature: {args.feature}")
    print(f"  seq_length_x: {args.seq_length_x}")
    print(f"  seq_length_y: {args.seq_length_y}")
    print(f"  y_start: {args.y_start}")
    print(f"  dow: {args.dow}")
    print("=" * 50)

    if os.path.exists(args.output_dir):
        if sys.stdin.isatty():
            reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
            if not reply or reply[0] != 'y':
                exit()
        else:
            print(f"{args.output_dir} already exists, overwrite.")
    os.makedirs(args.output_dir, exist_ok=True)
    generate_train_val_test(args)