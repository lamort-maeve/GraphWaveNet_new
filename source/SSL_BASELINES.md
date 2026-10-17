# 半监督算法对照

## 选择与定位

新增一项经典算法 Π-model（Pi Model），与已有 Mean Teacher 比较。

- Π-model：同一学生网络在两次独立输入噪声及 dropout 下前向，约束两个预测一致；两次预测都参与反向传播，不使用 EMA 教师。
- Mean Teacher：保留原有 EMA 参数教师、干净教师输入、学生噪声和原量纲 MAE 一致性，未修改 mean_teacher.py。
- 全监督仅作标签充足时的参考，不作为另一种半监督 baseline，也不是数学上保证的上界。

来源：Laine & Aila, *Temporal Ensembling for Semi-Supervised Learning*, ICLR 2017，https://arxiv.org/abs/1610.02242；作者实现 https://github.com/s-laine/tempens。本文实现的是该文中的 Π-model，不是 Temporal Ensembling 算法。

原论文主要用于分类。这里明确标注 **Π-model regression adaptation**：监督损失用现有流量 MAE，一致性用标准化连续预测的 MSE，限定在未标注预测起点。模型保持 train 模式，两次前向各更新一次 BN，使用独立 dropout；不把连续流量变成类别，不套分类置信度阈值。它是有文献依据的通用半监督基线适配，不能称为时序预测 SOTA 的官方复现。

## 独立开关

`--ssl-method none`：不使用半监督。

`--ssl-method mean_teacher`：原有方法；仍用 ssl-weight、ssl-warmup、ema-decay、input-noise。旧 --semi-supervised 仅是它的别名。

`--ssl-method pi_model`：新方法；独立使用 pi-weight、pi-noise、pi-warmup。禁止与 --semi-supervised 同用；不构造教师、不保存教师状态、不执行 EMA 更新。监督与验证指标仍在原始流量量纲下计算。

两个方法的一致性损失单位不同，所以相同的 0.1 权重不代表相同约束强度。各自参数必须用验证集选择，不能据默认值的一次运行判断谁更好。

## 主要对照矩阵

| 固定预测网络 | 无半监督 | Mean Teacher | Π-model |
| --- | --- | --- | --- |
| 原版自适应计算的 GraphWaveNet | 已有 sup25_adaptive | mt25_adaptive | 新增 pi25_adaptive |
| GraphWaveNet＋动态 PCMCI | 已有 sup25_combined | mt25_combined | 新增 pi25_combined |

每一行比较半监督算法；每一列比较因果模块。不能把“带因果的 Mean Teacher”和“不带因果的 Π-model”单独比较后就说 Mean Teacher 更好。保持相同训练/验证/测试划分、25%标签掩码、节点与输入、优化器、epochs、batch、学习率和原始数组。记录时间及优化步数；Pi 两次学生前向均保留梯度，计算开销与教师方法不一样。

## 严格实验流程

1. 测试通过后先跑小规模 smoke，不将其结果放进论文。
2. 先固定同一环境，用 seed 42 验证四格实验可运行；已有 Mean Teacher 可作历史结果，不因新增 Π-model 自动失效。
3. 给两种算法相同的验证调参预算。建议各自预先固定 6 个配置：权重 {0.01, 0.1, 1.0} × 输入噪声 {0, 0.02}，warmup 固定10。先只在验证集选定，记录全部尝试，不根据测试集结果增加某一方法的搜索次数。此网格是起点，若重新设计搜索范围，两种方法应获得同等机会。
4. 调参必须追加 --skip-test，只保存验证选择结果与最佳权重，不执行测试前向。根据 validation_selection.json 选择参数。确定后可用同一完整指令追加 --resume 并去掉 --skip-test，在 epochs 不增加时加载该运行的 best.pt 执行测试；不要给所有候选都执行这个步骤。测试集文件仍参与数据完整性检查，但不参与损失、选模或参数选择。
5. 确定参数后用相同种子42—46，报告 MAE/RMSE/MAPE 均值、标准差及逐种子配对差异。沿用旧42前需确认其配置、环境、数组及算法参数一致。验证只选学生模型，不能为某一方法临时改成选教师。
6. 先完成动态组算法对照，再研究动静结合，不同时改图、模型、标签比例及算法。主要结论稳定后补10%/50%标签预算及第二数据集。
7. 标签预算指预测起点监督，历史观测可见。PCMCI必须只使用当时可用数据，不因某方法使用因果图而多获得未来目标信息。

## 命令

在项目根目录运行，COMMON 按 zhiling.txt 定义。新套件不会自动重复旧实验；默认不执行，只有 --execute 才训练。

```bash
# 只补新增两组 Π-model
python newcode/run_experiments.py --suite ssl_baselines --only pi25_adaptive pi25_combined --seeds 42 --prefix ssl_compare --execute "${COMMON[@]}"

# 需要统一环境、重新调参时，四格一起跑（包含 Mean Teacher）
python newcode/run_experiments.py --suite ssl_baselines --seeds 42 --prefix ssl_compare_all --execute "${COMMON[@]}"

# 单独指定新算法参数，不影响 Mean Teacher 参数
python newcode/run_experiments.py --suite ssl_baselines --only pi25_adaptive pi25_combined --seeds 42 --prefix pi_w001_noise002 --pi-weight 0.01 --pi-noise 0.02 --pi-warmup 10 --skip-test --execute "${COMMON[@]}"
```

旧 source hashes 发生变化时不能直接 resume，这是保护；保留旧结果，不覆盖旧运行。新增算法不改变 Mean Teacher 默认参数或预测网络。新增回归测试验证两方法分支隔离、无标签真值不参与训练、Pi 两视图都有梯度。本机无 PyTorch，数值测试必须先在服务器执行：`python -m unittest discover -s newcode/tests -p 'test*.py'`。
