# 神经网络第一轮训练与原程序评分（2026-10-23）

4种结构 × 3个种子，共12次GPU训练，固定100/300/600轮检查点。36个单模型候选+12个三种子集成候选+原baseline，共49次原程序评分。用时498.82秒，仅使用启动时空闲的0号A100，目前已释放。

所有候选生成后均实际调用：

```bash
python Lab6/baseline.py predict --weights <候选.json>
```

不传其他 --model，评分模型始终为原 Lab6/model.npz。训练MSE与网络自身预测仅为训练/搜索过程数据，不参与最终排序。

| 结构 | 该结构最低原程序评分 | 对应候选 |
| --- | ---: | --- |
| mlp64_relu | 97.882395 | mlp64_relu_2027_e600 |
| mlp128_relu | 88.140128 | mlp128_relu_2027_e100 |
| mlp128_tanh | 98.030091 | mlp128_tanh_2028_e300 |
| mlp256_deep | 95.190742 | mlp256_deep_2027_e100 |

原baseline为75.895701。本轮所有神经网络候选均未超过它，因此 weights_recommended.json 保留原权重。

best_neural/ 保留评分最低的神经网络：21→128→64→1 ReLU，种子2027，第100轮；候选权重评分88.140128。此目录是本轮最佳网络，不表示它已经胜过baseline。

- models/：36份完整网络检查点、预处理参数和训练MSE历史。
- weights/：各候选的3×7合法权重。
- commands/：49次原评分命令、stdout与stderr。
- comparison.csv/json：逐方案原程序评分。
- complete.json：最终选择、固定基线、单卡UUID与运行时间。
- audit.json：评分来源、原文件校验、合法权重与检查点完整性。
- training_and_teacher_scores.png：左图为同一结构各检查点最低原程序评分，右图为仅作诊断的训练MSE。

复核命令：

```bash
CUDA_VISIBLE_DEVICES= python -m neural_baseline.audit neural_baseline/results/first_run_20261023
python Lab6/baseline.py predict --weights neural_baseline/results/first_run_20261023/best_neural/weights.json
```

网络加载：

```python
from neural_baseline.models import load
model = load("neural_baseline/results/first_run_20261023/best_neural/model.pt")
# model接收float32 [batch,21]原始权重，返回原单位ramping预测；不会替换老师评分程序。
```
