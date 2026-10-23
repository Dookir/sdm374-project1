# 保留的五种神经网络

按自身预测 ramping 从低到高选取；相同隐藏层和激活函数只留一份。

|网络|种子|轮数|预测 ramping|
|---|---:|---:|---:|
|relu_64_32|2026|600|40.701691|
|relu_128_64|2028|300|53.974716|
|relu_256_128|2028|100|60.568375|
|relu_512_256_128|2028|100|69.556396|
|relu_1024_512_256|2026|50|71.960045|

每个 retained/<网络名>/ 内保留模型、3×7权重、选择记录、训练与搜索记录。

```bash
source server/activate.sh
CUDA_VISIBLE_DEVICES= python -m neural_baseline.retained --verify
CUDA_VISIBLE_DEVICES= python -m neural_baseline.retained --name relu_64_32
```

无需GPU。--weights 可指定另一份3×7 JSON。旧实验归档位置见 retained/manifest.json。
