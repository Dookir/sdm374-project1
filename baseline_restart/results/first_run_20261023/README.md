> 历史结果：本文件记录旧的固定baseline预测对照，不代表当前方案排序规则。请先阅读[当前方法](../../../docs/WORKFLOW.md)与[最新结果](../../../docs/RESULTS.md)。

# 第一轮：沿用原 baseline 程序（2026-10-23）

所有候选均实际调用原 `Lab6/baseline.py optimize` 生成，再调用原 `predict` 评分。`provided_baseline` 行复测原权重，评分模型始终固定为原 `Lab6/model.npz`。
原始命令、stdout 和 stderr 在 `commands/`，生成的参数在 `models/`，候选在 `weights/`。

| 方案 | 原评分输出 |
| --- | ---: |
| provided_baseline | 75.895701069 |
| provided_model_native_optimize | 75.895701069 |
| native_train_then_optimize | 75.895701069 |
| linear_ols | 121.850231397 |
| quadratic_ols | 76.751303768 |
| quadratic_ridge_0.1 | 76.748907989 |
| quadratic_ridge_1 | 76.727588256 |
| quadratic_ridge_10 | 76.536740346 |
| quadratic_ridge_100 | 75.895701069 |
| quadratic_ridge_1000 | 78.295160987 |
| quadratic_lasso_0.01 | 76.147560021 |
| quadratic_lasso_0.1 | 78.266686467 |
| quadratic_lasso_1 | 89.735518667 |
| provided_model_expanded_starts | 75.895701069 |

本轮未超过原 baseline。`weights_recommended.json` 与原权重完全一致，没有把浮点误差视为改进。
扩展起点共1384个；仍沿用原梯度算法，没有修改步长或评分定义。多起点复现同一结果不能当作全局最优的数学证明。
OLS 导出时为原 NPZ 格式补写 alpha=0 的元数据，表示没有正则化；拟合仍为 LinearRegression，原导出与寻优函数不变。
完整校验在 complete.json。测试：`python -m unittest -v baseline_restart.test_restart`。
