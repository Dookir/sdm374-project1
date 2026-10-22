# 第一版神经网络结构

六种结构已完成首轮五折、三种子训练。当前网络候选为 **building_mlp 三种子集成**，仍需后续调参与真实仿真检验。
网络的自评分、训练损失和接口检查都不作为项目真实成绩。

## 输入和输出

输入是原始的 `[batch,21]` 权重，按三栋楼、每栋七项排列；输出是 `[batch]` 的原单位 ramping。
每栋楼七项权重之和为 1，因此输入只有 18 个独立自由度。
默认在各栋楼内部用固定 Helmert 正交基把 7 维转换为 6 维，再拼成 18 维。此变换不依赖数据、
不丢失合法权重之间的差异，并保留约束切平面内的欧氏距离；不是直接删掉一个任意权重。
原始 21 维表示保留为 MLP 的消融对照。

输入均值/标准差、标签均值/标准差都只从当前训练折计算，并作为冻结的模型 buffer 保存。
验证/测试前向不会更新统计量。对外输出还原到原单位，方便与已有真实标签比较。

## 六组结构

| 名称 | 结构（预处理后） | 目的 |
| --- | --- | --- |
| `mlp_small` | 18 → 64 → 32 → 1 | 主力之一，小数据基准 |
| `mlp_medium` | 18 → 128 → 64 → 1 | 检查增加宽度是否有效 |
| `mlp_large` | 18 → 256 → 256 → 128 → 1 | 容量消融，检验过拟合和欠拟合 |
| `resnet` | 18 → 64；2 个 64 → 32 → 64 残差块；64 → 1 | 检查残差连接与深度作用 |
| `building_mlp` | 三支独立 6 → 32 → 16；拼成 48 → 64 → 32 → 1 | 利用三栋楼的分组结构及跨楼交互 |
| `quadratic_residual` | 当前训练折二次回归 + 18 → 64 → 32 → 1 的修正网络 | 主力之一，保留二次趋势并学习偏差 |

隐藏层默认 SiLU，支持 Tanh/ReLU 消融；输出为线性层。默认不使用 BatchNorm，Dropout 为 0，
保留显式 Dropout 参数供调参。网络规模按数据量设置，算力用于多种子验证和模型比较。
三栋楼的编码器不共享参数，楼宇不是可以任意交换身份的样本。

残差块为 `h + 0.1 * W2(SiLU(W1(SiLU(h))))`（含偏置，可加 Dropout），其分支宽度为 32。
二次残差网络的最后一层零初始化，因此初始输出恰好等于训练折内的二次回归。
二次系数被冻结；网络只学习修正量。**禁止在折外验证中使用已见过全量标签的老师 model.npz 作为固定趋势。**
后续每折都需重新拟合二次基模型，包括其正则化参数选择。

可微前向完整保留对原始 21 个权重的梯度，未来寻优仍需每步投影，保证每栋楼权重合法。
预测层不强行截断负数，以免掩盖外推问题或破坏梯度；发现不合理预测应记录并在验证中处理。

可训练参数量分别为：small 3,329、medium 10,753、large 103,681、resnet 9,665、building 7,505、quadratic_residual 3,329。冻结的二次系数和标准化统计不计入可训练参数。完整结构清单见 `architecture_catalog.json`。

## 代码接口

```python
from neural.models import make_surrogate, save_surrogate, load_surrogate

# X_fit、y_fit 只包含当前训练折；forward 输入为与模型相同 device/dtype 的 Tensor。
model = make_surrogate("mlp_small", X_fit, y_fit).cuda()
predicted_ramping = model(weights_tensor)

# 对 quadratic_residual，另传当前训练折拟合得到的 (c, a, H)。
model = make_surrogate("quadratic_residual", X_fit, y_fit, quadratic=(c, a, H)).cuda()
```

checkpoint 保存结构、预处理、冻结趋势与网络参数，使用 `weights_only=True` 读取。
输入梯度、单样本 batch、float32/float64 和不同设备均保留明确接口。

## 已实现的结构检查

```bash
cd /data/zengxiong/wuxinkai/sdm374-project1
source server/activate.sh
python -m unittest -v neural.test_models
python -m neural.inspect_architectures --device cuda
```

后一个命令仅在 1310 条拟合行上建立统计量及二次趋势，对一个 minibatch 做三步更新，
验证 GPU 前向、参数梯度、输入梯度和 checkpoint 读写；它不报告网络优劣。
结果默认写到仓库外的 `../network-architecture-check.json`。

首轮已完成多种子训练、内部早停、低分区误差和两次选择遗憾值比较；下一阶段研究正则化、输入/激活消融和边界外推。
首轮使用 AdamW、lr=1e-3、weight_decay=1e-4、batch=128，未进行神经网络超参数搜索。
模型选择依赖未用于训练的真实标签，不能用老师代理模型裁判新模型，也不能用网络最低自评分宣称胜出。

结构参考：[Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959)。
本项目的建筑分支和二次趋势修正是待实验检验的设计选择，不是该论文对本数据集的结论。


## 首轮训练与验证

`python -m neural.train --out neural/runs/first_training_20261022`。
先激活 `server/activate.sh`，再设置 `CUDA_VISIBLE_DEVICES` 为一张空闲卡的完整 UUID。
程序启动时复查空闲状态，并要求仅可见一张卡；顺序训练，显存上限为该卡的 5%。

六种网络 × 五个外层测试折 × 三个随机种子（2026/2027/2028）。每折训练集内部留出 20% 选早停轮次，随后用整个外层训练集按该轮次重训；测试标签不参与标准化、训练或早停。
AdamW，学习率 0.001，weight decay 0.0001，batch 128；最多 1500 轮，150 轮无改善停止，每 5 轮检查。
二次岭回归在各训练折内部按老师的五折方法选 alpha；CatBoost 使用此前固定参数作补充对照。

`comparison.csv` 给出逐种子的平均指标和三种子预测平均后的 ensemble 指标；`oof_predictions.csv` 保留每行折外预测。
MAE、RMSE 与低分区误差越小越好。`regret_at2` 为每折挑出预测最低的两个方案后，其最好的真实分数与该折真实最优的差距，再对五折取平均；越接近 0 越好。
这是已有数据上的筛选实验，不能代替对全新权重的真实仿真，也不是未经使用的最终盲测集。首轮固定参数比较不等于系统调参结论。

中断后使用完全相同的命令恢复，已完成实验会跳过；配置或代码改变时必须换结果目录。


## 首轮结果（2026-10-22）

结果目录：[first_training_20261022](runs/first_training_20261022)。所有网络均完成 5 折 × 3 种子，共 90 组；神经网络下表使用三种子平均预测，对照为单个模型。

| 模型 | 折外 MAE | 真实最低20%区域 MAE | 每折 regret@2 均值 |
| --- | ---: | ---: | ---: |
| catboost_reference | 4.388 | 5.063 | 0.795 |
| building_mlp | 4.505 | 4.267 | 0.342 |
| mlp_large | 4.698 | 4.668 | 0.403 |
| resnet | 5.707 | 5.656 | 1.721 |
| quadratic_residual | 5.810 | 6.058 | 0.403 |
| mlp_small | 5.867 | 6.116 | 0.403 |
| mlp_medium | 5.983 | 6.246 | 0.403 |
| quadratic_ridge | 6.927 | 9.675 | 2.493 |

本轮分楼 MLP 的整体 MAE 比 CatBoost 高约 2.7%，但低分区域与两次选择表现更好。只有五个测试折，不能据此认定它对新权重一定优于 CatBoost。
其入选两组预测存在约 5.39 的平均低估，后续寻优仍须检查外推与多模型分歧。
全量训练模型在 `final_models/`；训练轮次取各种子五折内部早停最佳轮次的中位数（270、170、175），全量训练误差不作为验证精度。

复核与绘图：

```bash
CUDA_VISIBLE_DEVICES= python -m neural.audit_results neural/runs/first_training_20261022
CUDA_VISIBLE_DEVICES= python -m neural.plot_results neural/runs/first_training_20261022
python -m unittest -v neural.test_models neural.test_training
```

在服务器上加载三个最终模型并取预测平均：

```python
from pathlib import Path
import torch
from neural.models import load_surrogate
folder = Path("neural/runs/first_training_20261022/final_models")
models = [load_surrogate(folder / f"building_mlp_{seed}.pt") for seed in (2026, 2027, 2028)]
x = torch.as_tensor(weights_3x7, dtype=torch.float32).reshape(1, 21)
with torch.inference_mode():
    prediction = torch.stack([model(x) for model in models]).mean(0)
```

## 老师脚本复测候选

已为六种网络分别生成候选权重，并逐一运行原 `Lab6/baseline.py predict`，完整命令输出和对照见 [统一复测结果](runs/teacher_scoring_20261022)。这组候选没有超过原 baseline 的 75.895701；二次残差网络基本保留原权重。
