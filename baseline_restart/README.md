# 历史回归对照：非当前主流程

[首页](../README.md) · [当前评判方法](../docs/WORKFLOW.md) · [最新结果](../docs/RESULTS.md)

此目录保存回到原baseline流程时的回归实验：一次线性、二次普通回归、Ridge、Lasso，以及扩展起点对照。`run.py`、`PLAN.md`和原结果保留用于追溯，不是新队友的默认运行入口。

**历史实验曾使用固定 `Lab6/model.npz` 对所有候选统一预测；当前已改为各模型自身预测ramping。** `PLAN.md`与结果说明里的“唯一评分”等表述只适用于当时那轮实验，不能覆盖当前规则。旧 `teacher_script_score` 是固定回归模型的预测，不是老师实测成绩。

查看历史数据可读 `results/first_run_20261023/comparison.json`，字段 `generating_model` 是生成候选的回归模型，`teacher_script_score` 是当时另一个固定模型计算的对照值，两者不能混称。

老师原始baseline仍在 `../Lab6/`，自身预测75.8957。当前最好神经网络预测40.4908的模型和权重见[最新结果](../docs/RESULTS.md)。本目录不参与11个神经网络的加强寻优，也不要求队友先跑它。
