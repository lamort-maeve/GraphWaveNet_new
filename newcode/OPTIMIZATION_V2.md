# 2026-10-12 论文实验版更新

当前默认architecture=paper：内部滞后PCMCI层 + 原版自适应运算 + 可选EMA半监督。联合从头训练；graph-weighting默认confidence；图控制与标签预算见zhiling.txt。本文以下记录为历史版本，不代表当前默认。准确的方法定义与实验限制见论文撰写大纲.md。

---

# 本次优化：V2

## 2026-10-12：原版自适应基线与目标时刻对齐的 PCMCI 修正（当前默认）

### 结果诊断

MAE越低越好。3.0相对2.0：仅自适应测试MAE下降0.029842；仅因果上升0.121373；组合上升0.158730。不能把后两者称为准确度提升。

核对真实config/history后的证据：
- 2.0三组为context=13、batch=64、lr=0.002。3.0自适应为25/16/0.001，但因果和组合为25/64/0.0035，存在明显训练条件混杂。学习率和batch改变可能影响泛化，现有结果不能证明某个模块导致退步。
- 3.0因果最佳验证在epoch34，train=9.453139、val=9.761396，最后train降至9.265063但val升至9.818119；组合最佳在epoch30，train=9.354238、val=9.705672，最后为9.189224/9.747436。这与过拟合或优化后期泛化退步相符。
- 2.0/source也包含多尺度、图门控、LayerNorm、持久性等扩展。因此旧表中的adaptive不是未经修改的完整原版GraphWaveNet，只是使用了它的自适应矩阵公式。此前未明确区分这一点；本次纠正。
- 3.0同时改动q折减、边强度归一化、状态注意力和额外预测分支，不能单独归因。因果图可能包含估计噪声、窗口变化和同节点自回归边；这些是合理解释，尚未通过消融确认。

### 本次方法

默认 `--architecture gwnet`。`causal_model_v1.py`在原有代码后嵌入参考目录model.py的类，只有类名重命名，参考文件不修改。单测对同随机种子下全部参数、输出做逐元素一致性检查。

三模式：
- adaptive：原版GraphWaveNet的softmax(ReLU(E1E2))、二阶图传播、时序卷积、BatchNorm、skip和输出头；无PCMCI参数、无图bank。独立运行该模式时不读取per_window，缺少图目录也能运行。
- causal：保留GraphWaveNet的本地时间预测路径，关闭自适应图模块，另外用PCMCI预测修正。没有nodevec参数或自由学习的空间邻接矩阵；所有额外空间消息来自PCMCI。并非只靠邻接矩阵脱离原始序列预测。
- combined：载入本次gwnet/adaptive/best.pt，冻结整个自适应模型（包括BN统计和dropout状态），只训练PCMCI残差分支。原始输入仍传给两条分支。此为两阶段训练，多了因果分支训练预算，不是与单阶段baseline完全同计算预算的消融。

新增 `ForecastPCMCI` 使用真实y_offsets：对y(t+h)，lag l的源观测为x(t+h-l)。只使用l>=h且仍在输入范围的关系；lag0及需要未来源值的关系不伪造、不填充为观测。使用预测起点t已生效的图及此前快照，不使用未来生成的图。正负边分开做图历史一维卷积，按lag保留源水平与变化量，再用小型节点共享网络预测修正。没有逐层N×N状态注意力或多层图传播，减少噪声注入与计算量；空图时修正严格为0。

输出修正层零初始化、目标节点局部门控，combined起点与载入的自适应模型完全相同。把epoch0的验证结果也保存为候选；若后续训练没有改善验证MAE，best_epoch=0、pcmci_validation_improved=false，表示没有证据证明PCMCI有效，不能称为超过baseline。这只能保证候选选择不劣于同一验证基准，不能保证新样本/测试集更好。测试指标不能用于选训练轮次。

图权重默认恢复2.0式 `legacy` 归一化，取消默认q值再折减；q<=alpha的筛选仍保留。这也是待验证选择，`--graph-weighting confidence`可做单变量消融。旧3.0结构仍可显式使用 `--architecture v2 --graph-weighting confidence`，但不能将它称为本次的原版自适应基线。

输入适配仍为默认60个“路段×变量”节点、每节点1维输入，输出20路段flow；三组保持相同数据划分、标准化、上下文和指标。原版骨干等价不代表重现原论文的道路节点/输入特征设置或指标。不能直接加载2.0/3.0权重，必须新训练原版自适应基线。

### 文件与运行

修改existing code/causal_model_v1.py、causal_model.py、causal_data.py、train_dynamic.py、test_v2.py以及现有说明。保留数据缓存、常驻workers、每轮数值检查、简洁输出，不复制source。不修改Graph-WaveNet-master参考源码。修复train_dynamic.py中函数调用后直接拼接三引号中文说明导致的语法错误。

命令见zhiling.txt。`--mode all`自动按adaptive→causal→combined运行并传入新adaptive权重；分开运行combined必须提供 `--adaptive-checkpoint`。模型及数据、标准化、目标时刻、输入顺序要匹配。续训combined时还检查baseline文件摘要，防止悄悄换基准。若要延长adaptive训练并更新best，应为combined新建实验；不要在已有冻结组合实验里扩展all模式的baseline。

当前修改没有真实数据完整训练结果，不承诺因果或组合一定胜出。旧版本“已验证精度”文字均只适用于旧实验。以下章节保留历史实现说明，不是当前默认设置。

本次验证：36项测试全部通过；参考GraphWaveNet参数/输出逐元素一致；零图修正、目标时刻滞后对齐、冻结权重/BN、已知合成父节点关系的可学习性均通过。小型数据上完成三组训练/预测/保存与续训流程；单独adaptive在图目录不存在时运行通过。真实数据读取全部341张图，三组默认hidden=32、8层、context=25分别完成单批训练与验证，参考源码校验值未变化。未执行真实数据完整训练，CUDA未在本机验证，不能把这些程序检查解读为精度提升。

## 2026-10-11：专门增强 PCMCI 分支

2.0 表中验证 MAE：adaptive 9.535、causal 9.659、combined 9.570。实际测试汇总分别为 9.476720、9.627624、9.533512。因果仍落后，因此本次不改自适应计算，而是改 PCMCI 图输入与消息路径。

注意：2.0 的 config 记录 context=13、batch_size=64、learning_rate=0.002，缺少本地 V2 的一些结构字段。不能将本地默认 context=25、batch=16、lr=0.001 的新实验直接视为只改变因果模块的严格对照。建议同一入口重新跑三模式，或显式统一参数；不能保证完全复现来源配置不完整的旧训练。

修改现有文件，不新增源码或实验报告文件：

- `code/causal_data.py`：新增 `confidence` 图权重策略。仍保留 q≤alpha 的全部显著边和符号，用 `1-0.5*q/alpha` 折减临界显著边，再除以 `max(1, 入边绝对权重和)`。弱单边不再被放大成1。q只是统计证据，不是因果概率，该折减公式是待验证的建模选择。`--graph-weighting legacy` 恢复原先归一化。
- `code/causal_model.py`：新增 `PCMCIRefinedConv`，仍沿过去图快照做时间卷积。根据目标当前状态与源节点 t-lag 的状态产生 `2*sigmoid(score)` 边权修正，只乘在 PCMCI 历史支持集上，不新增自由边、不改变符号；每个节点、时刻再用独立有界滞后门控选择 lag 消息。仅因果模式没有 nodevec 自适应邻接参数。
- 同文件新增 `CausalReadout`：从浅层时序特征做一次带滞后的 PCMCI 聚合，直接输出小幅预测修正，使 lag=12 等长延迟信息不必经过全部深层图消息路径。该分支输出层无偏置、图全零时输出严格零。组合组也使用它，自适应组不创建它。
- `code/train_dynamic.py`：接入以上参数、实现版本和断点配置检查。`--no-pcmci-refinement` 只关闭深层图卷积中的新选择器；`--no-causal-readout` 关闭独立直接预测分支，该分支自带选择器。两项与 `--graph-weighting legacy` 同时使用，可恢复本次修改前本地结构与图处理。旧 checkpoint 不应直接续训新结构。
- `code/test_v2.py`：加入不新增边、不访问未来、无关节点隔离、长滞后梯度、空图严格零，以及自适应模式在开关 PCMCI 增强前后参数和输出逐元素一致的检查。

图历史时间卷积服务于 PCMCI 图；自适应矩阵使用单独的一阶/二阶传播。组合模式两者分别门控再融合。共享的多尺度时间模块服务三组；本次保持其实现不变。本次没有保证因果组必胜的逻辑，应按统一验证 MAE 选择模型，并用多随机种子验证增益。

检查：24项不写文件的CPU单元测试通过；全部341张图通过读取和新权重检查；真实数据三模式、hidden=32、8层、context=25、batch=2，分别完成一个训练、验证、测试批次。检查只在内存进行，未新增输出文件；会写入文件的断点续训测试本次未执行。未进行完整训练或CUDA检查，尚无精度提升结论。

## 精度增强候选版本（同日后续更新）

图片中的核心方法保持不变：沿因果图窗口的时间轴做一维卷积，并让逐时刻图参与图卷积，不把所有窗口压成静态图。新增以下默认模块，属于需要完整实验验证的优化假设：

1. **节点内多尺度时间模块**：在时序骨干前加入核宽 3、膨胀率 1/2/4/8 的深度卷积分支，分别覆盖 3/5/9/17 个采样位置的跨度；根据当前状态与变化量学习尺度权重，采用门控残差、小权重初始化。全部卷积只在左侧补零，不访问未来，也不跨节点混合。三模式全部使用此模块。`--no-multiscale` 可关闭。
2. **正负分离的稳定图时间卷积**：默认 `--graph-history-fusion signed_convex`。先拆分正边和负边，再各自沿图历史卷积，避免不同窗口的相反效应在特征投影之前抵消。卷积核用 softmax 约束为非负、和为 1，默认三窗口初始化为 0.1/0.1/0.8。两个符号分支共享时间核、独立特征投影，完整保留 lag。`--graph-history-fusion signed_conv` 恢复上一次的无约束时间卷积。

新增参数写入 config 并参与断点续训检查。新模型不能直接续训前一版本权重，建议使用新的 `enhanced_*_s42` 实验名。

消融应保持 mode、种子、输入、训练轮数和 batch-size 一致，每次更换 run-name：默认增强版；加 `--no-multiscale`；加 `--graph-history-fusion signed_conv`；同时加二者可重跑上一版本结构。先比较验证 MAE，再报告对应最佳 checkpoint 的测试结果。已有旧实验没有证明这些新模块必然更准。

22 项 CPU 测试通过，覆盖未来信息隔离、节点隔离、正负历史不提前抵消、图权重范围、梯度、三模式与断点续训一致性。真实数据三模式各两个训练、验证、测试批次全部通过，使用341张图、默认hidden=32、8层、context=25；检查目录为 `newcode/outputs/checks/tensor_enhanced_20261005`。未执行完整训练，CUDA未验证。

## 2026-10-05 修改与验证

本次直接修改 `newcode/code/causal_model.py`、`train_dynamic.py` 和 `test_v2.py`。

- `temporal_tensor` 继承门控图卷积，保留本地时序路径，因果消息和自适应消息分别门控后相加。三种模式使用同样的时序骨干；causal 不创建自适应矩阵参数，adaptive 不创建因果张量卷积参数，combined 同时使用两条图分支。
- 时间卷积前对当前批次图 ID 去重，相同图的历史只计算一次，再恢复到每个样本的每个输入时刻。已经验证与显式展开版本的输出和卷积核梯度一致。
- 新增 `--graph-kernel`，默认 3，表示当前已生效图及此前两张已生效图；1 表示只用当前图。这里的时间轴是因果图快照的顺序，不是 PCMCI 的 lag 轴，也不是均匀的五分钟采样轴；缺失窗口继续沿用最近过去的图，不使用未来图补齐。
- 所有图以 `[窗口, lag, source, target]` 保存。对每一个 lag，沿窗口历史做可学习的一维卷积，再按输入时间对齐，传播 `source(t-lag) → target(t)` 消息。因此保留了图片中的图时间维，同时额外保留原始 PCMCI 滞后维。
- 结果 `comparison.csv` 新增 `best_validation_mae`；测试指标与预测均来自各组验证 MAE 最低的 checkpoint。

验证：18 项单元测试通过；读取全部 341 张图，真实数据上以默认 hidden=32、8 层、context=25、batch=2，对三模式分别完成两个训练、两个验证和两个测试批次，结果位于 `newcode/outputs/checks/tensor_gated_20261005`。这些只是可运行性检查，不能用来评价预测精度。本机检查为 CPU，完整训练和 CUDA 由云服务器执行。

本次修改改变了模型参数结构，请新建 run-name；不要用旧 checkpoint 恢复。最新命令见 `zhiling.txt`。

新的运行入口是 `python newcode/train.py`。原来的 `Graph-WaveNet-master/train.py` 仍指向旧 `result/code`，不要用它启动本次优化。

## 已发现的问题与修改

服务器旧动态因果组测试 MAE 为 10.026449，自适应组为 9.780940，因果组落后约 2.51%。训练和验证也存在差距，不能仅靠挑测试轮次来宣称因果组领先。完整对比见 `动态因果优化/1.0/comparison/REPORT.md`。`necode_result/combined` 未提供 checkpoint 或指标，不能推测其表现。

1. **按滞后、符号分别投影。** 旧算子通过一个全局 softmax 将 lag 0–12 的消息压成同一个表示；不同延迟无法保留各自通道变换，正负效应也可能在变换前相消。V2 为每个 lag 的正边、负边保留独立消息通道，再学习投影，全部 PCMCI 图和滞后继续使用。
2. **门控图残差。** 本地时序分支始终保留，由当前节点状态与因果消息决定注入强度，减少不稳定窗口的图信号干扰。空因果图的因果分支严格为零。仅因果模式没有自由学习的邻接矩阵，门控和通道变换不会生成新节点间边。
3. **流量水平与变化分开学习。** 默认预测为最近观测 flow 加网络修正，再叠加一条针对历史变化的线性时序路径。无需让整个深层网络重新学习流量基线。输出仍保持原单位与相同观测掩码。
4. **变量身份嵌入。** 区分 60 个路段/变量身份，缓解原模型共享一维输入投影、无法明确区分 flow/occupancy/speed 的问题。嵌入只作用于节点局部通道，不等于自适应边。
5. **上下文与训练。** 默认输入扩展到 25 步，让长滞后消息有更多历史可用；三组上下文一致。验证 MAE 平台期降低学习率，保存 best/last checkpoint，显式恢复训练并校验代码、数据与配置，保存随机状态以恢复 CPU 小实验的精确轨迹。
6. **时间卷积覆盖全部上下文。** 默认 4 个 block、每块 dilation 1 和 2 时，核宽 2 的感受野只有 13 步，25 步上下文里靠前的历史进不了深层时间栈。V2 现在把因果时间核加宽到感受野至少等于上下文（该默认配置下核宽为 3、感受野为 25），并且只在左侧补零。
7. **近期加权的时间汇合，以及按路段开关图。** 预测不再只取 skip 的最后一帧，而是对整段因果 skip 做时间加权，初始权重偏向最近一步。图门控增加每个节点的偏置和当前时刻变化量；变化量通道初始为 0。空图时因果消息仍严格为 0。输出修正按路段缩放，初始增益为 1，持久性基线加在缩放之后。
8. **时空因果张量上的一维卷积。** 默认图算子 `temporal_tensor` 不再只使用当前这一张因果图。每个时刻把当前图和更早的两张图堆成 `N×N×K`，用分组一维卷积沿这个时间维混合；卷积初始时只保留最新一张图。正边和负边在混合后仍分开投影。自适应模式不读这张因果张量。`--graph-operator gated_lag` 或 `v1` 可以换回以前的图算子。

这些是待验证的优化假设。当前已训练模型里验证 MAE 最低的是 `k3_lambda0.3/combined` epoch 100（9.830400）。本机没有完整训练 V2，未训练权重不能当作超过该结果的证据。测试集只用于诊断，调参看验证集；最终结论需要多个种子的服务器实验。

## 服务器命令

保持项目结构：`newcode/`、`per_window/`、`Graph-WaveNet-master/data/` 同级。

```bash
# 小规模检查
python newcode/train.py --device cuda:0 --smoke --mode all --batch-size 2 --hidden 4 --blocks 1 --layers 2

# V2 三组完整实验（默认早停 patience=20；需要跑满加 --patience 0）
python newcode/train.py --device cuda:0 --mode all --epochs 100 --batch-size 16 --seed 42 --run-name v2_seed42

# 中断后，用相同参数追加 --resume；不会自动载入旧结构 checkpoint
python newcode/train.py --device cuda:0 --mode all --epochs 100 --batch-size 16 --seed 42 --run-name v2_seed42 --resume

# V1 架构使用相同 25 步上下文、相同训练调度器，与 V2 对照
python newcode/train.py --device cuda:0 --architecture v1 --context 25 --mode all --batch-size 16 --seed 42 --run-name v1_control_seed42

# 默认已经是时空因果张量卷积。下面两条只用于对照旧图算子
python newcode/train.py --device cuda:0 --graph-operator gated_lag --mode all --batch-size 16 --seed 42 --run-name v2_gated_seed42
python newcode/train.py --device cuda:0 --graph-operator v1 --mode all --batch-size 16 --seed 42 --run-name v2_old_operator_seed42
```

默认结果写入 `newcode/outputs/runs/<run-name>`。建议 seed 42、43、44 分别运行。三组公平比较时保持参数、种子一致；不同结构参数量会记录在 config。可用 `--no-persistence`、`--no-variable-embedding` 做对应消融；`--no-persistence` 仅关闭最后值相加，保留直接时序变化分支。

如果专门重跑旧 13 步设置，可以用 `--architecture v1 --context 13`；这与 25 步 V2 同时改变了上下文和结构，不能把差异全部归因于新图卷积。

若服务器仍使用先前路径，明确传入路径：

```bash
python newcode/train.py --device cuda:0 --mode all --epochs 100 --batch-size 16 \
  --data /root/autodl-tmp/GraphWaveNet/data/data1 \
  --raw /root/autodl-tmp/GraphWaveNet/data/data1/full_year_cleaned.npz \
  --graphs /root/autodl-tmp/GraphWaveNet/data/per_window \
  --output /root/autodl-tmp/GraphWaveNet/result/v2 \
  --run-name v2_seed42 --seed 42
```

`--gpus 4` 支持从指定 CUDA 索引开始的连续四卡 DataParallel；默认单卡。保存权重不带 `module.` 前缀。服务器旧 PyTorch 1.10 的 checkpoint 加载 API 已做兼容处理，但本地检查环境为 PyTorch 2.7 CPU，多卡执行尚未在本机验证。

NPZ 的字节字符串元数据会自动解码；如果自己的服务器 NPZ 使用 object dtype 的元数据，追加 `--trusted-object-metadata`。只对自己可信的数据使用该选项。训练参数使用连字符形式，不是下划线。

## 检查与兼容

`test_dynamic.py`：方向、滞后、排列、梯度、掩码和指标检查；`test_v2.py`：空图、过去消息不受未来输入影响、正负滞后独立、最近值输出、元数据、断点恢复与连续训练一致性。全部现有 341 张图仍由 GraphArchive 枚举、按生效时间对齐，17 个缺号如实保留。数据/图不做训练后调整。

旧模型实现保存在 `code/causal_model_v1.py`，修改前代码另存 `动态因果优化/1.0/before_optimization/`。新结构不能加载旧 V1 权重继续训练，请为 V2 新建 run-name。`--resume` 不允许代码或输入变化，避免将不同实验混入一个目录。
