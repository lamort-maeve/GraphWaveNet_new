# 当前论文实验版本（2026-10-12）

默认 `--architecture paper`：内部按时间对齐的PCMCI滞后消息；自适应模式直接调用原版计算；联合模式端到端训练。支持监督预算与EMA教师一致性。训练命令见 [zhiling.txt](zhiling.txt)

先跑全监督三模式，再在相同标签预算下比较监督/半监督；半监督默认关闭。不能将本协议称为传感器缺失。需从头训练，不兼容旧架构检查点。纯自适应独立运行不读取PCMCI。

下面内容是历史实现记录，历史默认值及冻结联合模型描述不适用于paper版本。

---

# 动态 PCMCI + Graph WaveNet（已更新 V2）

**2026-10-12 当前默认已改为 `--architecture gwnet`：未经因果改造的参考GraphWaveNet自适应骨干；仅因果关闭自适应模块；组合冻结新训练的自适应模型、只训练目标时刻对齐的PCMCI预测修正。独立adaptive不需要PCMCI文件。不要直接续训2.0/3.0权重。最新方法和证据见 [OPTIMIZATION_V2.md](OPTIMIZATION_V2.md) 顶部，命令见 [zhiling.txt](zhiling.txt)。下方历史V2说明不再代表默认模型。**

**2026-10-11 最新：PCMCI 分支新增受图约束的状态边权选择、节点级滞后门控和浅层因果直接预测分支；图输入保留弱边效应强度并利用 q 值折减。自适应基线计算不变。方法、验证范围与消融见 [OPTIMIZATION_V2.md](OPTIMIZATION_V2.md) 顶部，运行见 [zhiling.txt](zhiling.txt)。新版本精度待云端完整训练验证。**

**最新增强候选版：默认启用节点内多尺度时间模块，以及正负分离、归一化权重的图历史卷积。已通过22项CPU测试；使用新实验名从头训练。详见 [OPTIMIZATION_V2.md](OPTIMIZATION_V2.md) 顶部和 [zhiling.txt](zhiling.txt)。精度提升尚需云端完整实验验证。**

**2026-10-05 更新：时间张量卷积现已接入图消息门控，并对重复图历史去重计算。最新三组云服务器命令见 [zhiling.txt](zhiling.txt)。18 项单元测试及真实数据三模式小规模训练/验证/测试已通过；未执行完整训练，未验证 CUDA。**

**当前版本请使用 `python newcode/train.py`，完整运行与消融说明见 [OPTIMIZATION_V2.md](OPTIMIZATION_V2.md)。**

下面保留的是 V1 设计说明，其中旧入口 `Graph-WaveNet-master/train.py`、13 步默认上下文、`result/runs` 输出路径均不适用于 V2。V2 默认为 25 步、输出到 `newcode/outputs`；仍保留全部 341 张现存图。

## 入口与服务器运行

在项目最外层（包含 `per_window`、`result`、`Graph-WaveNet-master`）运行。保留这三个目录的相对位置，路径随项目迁移，不依赖 Windows 盘符。依赖为 Python 3.10+、NumPy、PyTorch 2.1+；服务器根据 CUDA 版本安装相应 PyTorch。

```bash
# 不训练：读取全部图、校验全部数据时间对齐、生成因果张量与覆盖报告
python Graph-WaveNet-master/train.py --prepare-only

# 仅程序检查：三组各两个训练批次、两个验证批次、两个测试批次
python Graph-WaveNet-master/train.py --smoke --mode all --batch-size 2 --hidden 4 --blocks 1 --layers 2

# 服务器正式实验：同一随机种子、输入和数据划分，依次训练三组
python Graph-WaveNet-master/train.py --device cuda:0 --mode all --epochs 100 --batch-size 32 --seed 42

# 只运行一组
python Graph-WaveNet-master/train.py --device cuda:0 --mode causal --seed 42

# 可选：只使用 flow 变量，三组也都只输入 flow
python Graph-WaveNet-master/train.py --device cuda:0 --mode all --variables flow --seed 42
```

正式实验建议用 42、43、44 等多个种子重复。`--mode adaptive` 为仅自适应图，`--mode combined` 为自适应图加动态图。各组分别按验证集 MAE 选择最佳模型，最后才评估测试集。没有任何让对照组故意变差的逻辑，也不保证仅因果组必然领先。

## 模型与完整性

- **默认使用全部 341 个现存 NPZ 文件**，不限制前几个窗口，不取平均静态图，不做 top-k 截断。逐文件验证配套 CSV 中的节点 ID、变量顺序、边方向和边权，记录文件 SHA256。
- 原始 `val_matrix` / `q_matrix` 为 `(60,60,13)`。默认恢复 `[窗口, lag, source, target] = [341,13,60,60]` 张量，保留 q≤0.05 的有符号权重。60 个变量按 `路段×[flow,occupancy,speed]` 排列，最终只输出 20 路段 flow。`--alpha` 可调整显著性阈值。lag=0 保留为同时刻统计依赖，不能仅凭它解释出严格因果方向。
- 每条消息为 `A_t[lag,source,target] × h_source(t-lag)`。每个输入时刻单独查图；每一层都用这些图传播。不同滞后分别计算，学习层内滞后混合权重；各 lag 按目标节点的入边绝对权重和归一化，保留负边。仅因果组没有 `nodevec`、可学习全连接边或自适应矩阵。
- 时间卷积保持长度，始终只在左侧补零。默认 13 步上下文覆盖最大 lag=12（原数据为 12 步）；新增一步从已核对的原始时间序列读取。若文件中最大 lag 增大，程序自动扩展上下文。三组上下文始终一致。样本开始处没有历史的位置补标准化零。
- 使用原始文件 `data_model_ready` 补充 occupancy/speed 并做训练段统计标准化。**默认 all 模式与旧的仅 flow、12 步输入实验不是相同输入条件**，不能把旧实验分数直接当作这次的自适应基线；应运行本入口的三组。`flow` 模式只选 flow→flow 子张量，属于显式可选消融，不是默认完整配置。
- 所有层均采用 Conv2d，修复旧模型对四维输入使用 Conv1d 的问题。图卷积输出进入 skip 路径，最后一层图卷积也能影响预测。输出固定为 `[batch, horizon, 20]`，支持 batch=1 和 horizon=1，不会 squeeze 掉维度。

## 时间对齐与实验口径

图生效规则：只有 `graph_time_index <= 当前输入时刻` 才可用，同时检查 `window_end_index <= graph_time_index`。第一张图在索引 2016 生效；此前用零图，保留这些样本。验证/测试允许使用此前已完成窗口的 PCMCI 图，属于**滚动在线图更新**评估，不是“训练结束后冻结图”的评估。图生成流程本身是否用全年度统计预处理，无法仅靠这些输出文件确认。

目录里缺少以下 17 个编号：338–344、346–355。程序不会伪造缺失图，而是沿用最近已生效的图；清单保存在 manifest。所有 341 张现有图均要求至少被一个输入时间位置使用，否则直接报错。完整归档不等于把测试时期图给训练样本：训练、验证、测试各用当时可用的图，覆盖情况逐段记录。

程序逐块核对所有 train/val/test 输入、标签与完整序列的位置关系，不能静默接受乱序数据。训练与验证、验证与测试边界剔除目标与下一段输入重叠的前一段样本，三组一致。打乱时将输入、标签、有效掩码、逐时刻图 ID 绑定，末批不补重复样本。目标有效性由原始 `target_observed_mask` 决定，真实零流量计入 MAE/RMSE；MAPE 仅排除零分母，保存的是比例而非百分数。这样与旧版“所有零值都当缺失”的指标口径也不同。

提供的 `static_pcmci_exp.pkl`（K235）、`static_pcmci_exp_K3_lambda0.3.pkl`（K3）是已压缩的 20×20 图，不能恢复窗口和 lag，故默认动态模型不拿它们代替完整 PCMCI 结果，也不拿它们给自适应组初始化。原文件均保留。

## 输出组织

```text
result/
  code/                       新模型、数据加载、训练和测试代码
  legacy/                     修改前 train.py、model.py 备份
  checks/<时间或run-name>/      prepare-only / smoke；不是真实精度结果
  runs/<时间或run-name>/        服务器正式实验
    source/                   该次运行的代码副本
    inputs.json               数据路径与 SHA256
    graphs/manifest.json      全图顺序、时间、缺号、校验值
    graphs/causal_tensor.npz   有符号显著性张量、归一化张量、生效时间
    coverage.json             每个划分使用哪些图、空图样本数量
    causal/                   仅因果
    adaptive/                 仅自适应
    combined/                 自适应＋因果
      config.json
      history.csv             各 epoch 训练/验证指标
      best.pt                 模型、优化器、标准化统计、图 bank、生效时间
      test_metrics.json       总体及逐预测步指标、最佳 epoch
      test_predictions.npz    预测、真实值、有效掩码、样本原始索引
    comparison.csv            三组汇总
```

`--run-name` 可指定实验名，已有同名目录会报错防止覆盖。`--patience 20` 默认早停，0 关闭。代码不创建自动任务，不会后台继续训练。旧 `test.py` 面向旧 checkpoint，新 checkpoint 由新训练入口自动加载评估。

## 已提供的检查

```bash
python result/code/test_dynamic.py
python result/code/check_archive.py result/checks/smoke_all_variables
```

单元检查包括源→目标方向、负号、lag=0…12、缺图与生效边界、批次排列一致性、三个模式的梯度、图切换影响、单样本维度以及未来输入不影响过去消息。`check_archive.py` 对全部图分别执行八层网络前向，并做默认 hidden=32 的单样本反向检查。仅做程序检查，不执行完整训练。

后续优化请提供正式实验的 `comparison.csv`、各模式 `config.json`、`history.csv`、`test_metrics.json` 和 `coverage.json`；需要分析具体时段误差时，再提供预测 NPZ。
