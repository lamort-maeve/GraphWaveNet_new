# 动态因果时序预测论文代码

只保留当前 paper 架构作为运行入口，历史版本不再混在训练代码里。原版 GraphWaveNet 参考目录未修改。

## 半监督算法对照

新增独立的 `code/pi_model.py`，用 `--ssl-method pi_model` 选择；原 Mean Teacher 保持不变。论文对照协议见 [SSL_BASELINES.md](SSL_BASELINES.md)。调参用 `--skip-test`，避免查看候选配置的测试成绩。

## 常用入口

- `python newcode/train.py`：单个实验或三模式训练。
- `python newcode/run_experiments.py --suite graphs`：预览新图对照，增加 `--execute` 才运行。
- `python -m unittest discover -s newcode/tests -p 'test*.py'`：小规模程序测试。
- [已有结果复用与补跑清单](RESULT_REUSE.md)：哪些已经跑过、哪些需要补，避免重复九组。
- [论文实验说明](PAPER_EXPERIMENTS.md)：实验定义、命令和论文计划。
- [服务器指令](zhiling.txt)：优先补跑的具体命令。

## 代码结构

```text
newcode/
  train.py                 训练入口
  run_experiments.py       共用配置及批量实验入口
  requirements.txt        环境依赖下限，正式比较仍须锁定相同版本
  code/
    graphwavenet.py        参考 GraphWaveNet 主干，不含 Mean Teacher
    forecast_model.py     PCMCI 滞后消息与预测网络，不含 Mean Teacher
    mean_teacher.py       独立半监督训练工具，仅显式开启时创建教师
    trainer.py            训练、验证、测试和检查点
    data_pipeline.py      数据读取、时间对齐和缓存
    graph_controls.py     固定图及图消融
    metrics.py            MAE/RMSE/MAPE 与监督损失
    check_archive.py      已准备图的运行检查
  tests/                  当前模型、数据、训练与源码检查
  legacy/                 历史代码归档，不参与当前训练
```

## 半监督是独立选项

默认 `--ssl-method none`，不会创建教师，也不会加入一致性损失。
`--ssl-method mean_teacher --label-ratio 0.25` 才启用半监督。旧 `--semi-supervised` 是同一选项的兼容别名，不代表另一种算法。

Mean Teacher 使用学生参数平均生成教师，只用于预测训练，不修改 PCMCI 图发现过程；验证和测试仍使用学生模型。预测网络不会导入 mean_teacher.py。

## 兼容性与历史记录

原 `causal_model_v1.py` 中当前仍需要的主干移到 graphwavenet.py，其余历史模型归档。原 test_v2.py 中有效测试移到 tests/，没有把有用测试一起删掉。旧文件已从 code/ 移除，恢复材料在 legacy/pre_paper_cleanup_20261017.zip。

模型状态字典名称保持不变；不承诺未验证的旧检查点加载。当前代码不接受旧 v1/v2/gwnet 架构参数，旧命令请使用归档源码。代码来源变更后不续训旧运行，既有结果不会因此失效。

本次通过了无需 PyTorch 的源码结构与语法检查。本机未安装 PyTorch，数值与训练测试需在服务器执行。完整训练没有在本机运行。
