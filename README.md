# SDM374 Project 1 / Week7

当前按最新指令使用神经网络：**训练网络参数 → 投影梯度寻找3×7权重 → 原 baseline.py predict 评分**。
候选最终只按原评分程序输出排序，固定使用原 Lab6/model.npz。

## 目录

- Lab6/：老师原始材料、模型、权重和2048行数据，保持原样。
- neural_baseline/：当前神经网络实现，[结构与训练计划](neural_baseline/PLAN.md)。
- baseline_restart/：保留的线性/多项式/L1/L2回归对照。
- server/：服务器环境与激活入口。

## 本轮结果

[神经网络第一轮](neural_baseline/results/first_run_20261023)：4种结构、3个种子、12次单卡GPU训练，49个原程序评分对照。
最低网络候选为中型ReLU、种子2027、第100轮，原程序评分88.140128；原baseline为75.895701，因此推荐权重仍保留原方案。
最佳网络和对应权重在结果目录的 best_neural/；推荐提交权重为 weights_recommended.json。

## 服务器运行

```bash
cd /data/zengxiong/wuxinkai/sdm374-project1
source server/activate.sh
nvidia-smi
# 先选一张空闲卡的完整UUID，替换下面占位值；只能一张。
export CUDA_VISIBLE_DEVICES=<GPU_UUID>
python -u -m neural_baseline.run --out neural_baseline/results/new_run
```

完成后的目录禁止覆盖。训练检查点固定为100/300/600轮，不按MAE筛选；每份权重立即交给原程序评分。

```bash
CUDA_VISIBLE_DEVICES= python -m unittest -v neural_baseline.test_neural
CUDA_VISIBLE_DEVICES= python -m neural_baseline.audit neural_baseline/results/first_run_20261023
python Lab6/baseline.py predict --weights neural_baseline/results/first_run_20261023/best_neural/weights.json
```

旧树模型与旧神经网络实验退出当前目录，保存在 Git 标签 archive-before-baseline-restart-20261023 和服务器同名归档目录。当前使用新的 neural_baseline 流程。
