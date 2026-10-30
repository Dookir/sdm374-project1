# 全数据最终MLP的控制权重寻优

固定final_mlp256/results/full_20261030/model.pt，不重新训练。使用三个固定种子，每组约6.5万候选、64起点、800步投影梯度、1000步投影Adam、权重转移与8起点SLSQP。仅优化原3×7合法权重上的本模型自身预测ramping，CPU复核。

最低找到的predicted_ramping：**96.31106567382812**。这是搜索找到的最低预测值，不是全局最优证明。权重为weights.json，nearest_training_distance仅诊断，不加入评分。负预测字段：False。未消耗老师测试机会，原模型与冻结包不变。

复现：在仓库根目录运行 `CUDA_VISIBLE_DEVICES= python -m cv_experiment.predict --checkpoint final_mlp256/results/full_20261030/model.pt --weights final_mlp256_search/results/run_20261030/weights.json`。
