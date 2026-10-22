# 用老师原脚本统一计算候选分数

运行 `python -m neural.score_candidates`。只使用一张空闲 A100，老师文件保持原样。
六种网络分别使用三个全量训练模型，训练轮次取此前五折内部早停轮次中位数；复用已有分楼网络模型，其余模型保存在 `full_models/`。
各网络使用相同的 33 个起点（均匀权重、实测最好权重、30 个随机点及原 baseline），用自身三种子平均预测做投影 Adam 寻优，最多 1000 步。搜索期间未使用老师分数挑选候选。

然后每组权重均实际运行原命令：

```bash
python Lab6/baseline.py predict --weights neural/runs/teacher_scoring_20261022/weights_building_mlp.json
```

`teacher_stdout_*.json` 保存实际命令和输出。此入口输出 `predicted_ramping`，是老师固定二次模型的预测；仓库未提供独立真实模拟评估器。

| 方案 | 老师脚本 predicted_ramping（越低越好） |
| --- | ---: |
| original_baseline | 75.895701 |
| previous_catboost_conservative | 114.576925 |
| measured_best | 130.648301 |
| mlp_small | 110.709707 |
| mlp_medium | 96.081314 |
| mlp_large | 133.281999 |
| resnet | 108.068123 |
| building_mlp | 131.997254 |
| quadratic_residual | 75.895708 |

二次残差网络候选与原 baseline 的最大单项权重差为 4.6e-08，相当于保留了原方案（float32 与投影存在微小数值差）。本轮候选没有超过原 baseline。
已有 CatBoost 对照来自历史 CatBoost/HistGB 搜索，本轮只复测其权重，没有重跑该搜索。实测最好样本的标签是 115.764819，但老师模型对此权重预测为 130.648301。两列含义不同，见 comparison.csv。
