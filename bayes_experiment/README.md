# 三种MLP：TPE贝叶斯调参与嵌套十折验证

仅训练ReLU 64→32、128→64、256→128。复用原外层十折(seed=20261029)，每个外层训练集内部三折，每结构每外层折30次配置试验(前10次初始化，含原始配置，之后20次TPE)。共900试验、2700内部训练和30次最终重训及寻优。Optuna版本见运行manifest。

搜索学习率[0.0001,0.003]和AdamW权重衰减[0.0001,0.03](均对数)，隐藏层Dropout∈{0,0.05,0.1,0.2}。其余保持批量128、标准化MSE、梯度裁剪5、余弦T_max=600、eta_min=初始学习率×0.01。每个内部训练仅使用其训练子集拟合标准化；验证MAE逐轮早停patience30并恢复最佳检查点。调参目标为三内部折最佳检查点CPU验证MAE按样本数加权，不访问外层测试。

每外层折选择最小内部MAE配置，以其三个内部最佳轮数的中位数在完整外层90%数据上从头重训；沿原T_max=600调度在选定轮次结束，无外层测试早停。外层每条样本恰好一次留出预测。同一历史外层划分用于改进比较，不宣称全新盲测。

每个最终模型继续原加强权重寻优，只传入该外层训练数据。各模型自身predicted_ramping为候选成绩，MAE为独立预测诊断；不替换为baseline统一评分、不截断负预测。原结果不可覆盖。

## 运行

在服务器仓库根目录source server/activate.sh。仅一张启动时空闲GPU，CUDA_VISIBLE_DEVICES必须是完整GPU UUID。后台tmux运行；不持续跟踪。

```bash
CUDA_VISIBLE_DEVICES=GPU-实际空闲卡UUID python -m bayes_experiment.run --out bayes_experiment/results/bo10_20261029
# 中断后(必须先确保原进程停止且该卡空闲)
CUDA_VISIBLE_DEVICES=GPU-实际空闲卡UUID python -m bayes_experiment.run --out bayes_experiment/results/bo10_20261029 --resume
# CPU快照/完整审计
CUDA_VISIBLE_DEVICES= python -m bayes_experiment.check --out bayes_experiment/results/bo10_20261029
CUDA_VISIBLE_DEVICES= python -m bayes_experiment.check --out bayes_experiment/results/bo10_20261029 --audit
CUDA_VISIBLE_DEVICES= python -m bayes_experiment.predict --checkpoint <model.pt> --weights <weights.json>
CUDA_VISIBLE_DEVICES= python -m unittest bayes_experiment.test_bayes -v
```

输出包括30个最终模型、逐样本train/test预测、OOF、MAE汇总、30份候选权重、每试验内折历史/标准化/验证预测、Optuna SQLite数据库、参数分配及sampler快照。内部检查点只用于选择和生成保存的验证预测，不将2700个内部检查点作为交付模型。审计复算全部内部选择记录及最终30模型预测。

断点恢复验证数据/配置/源码/旧文件哈希；已完成内折保留，未完成内折从头重跑。SQLite保留试验状态；每个试验重新构造固定种子的TPE采样器，已分配参数独立持久化，故恢复不依赖未保存的随机状态。参数已完整写入数据库但分配文件未落盘时复用该RUNNING试验；参数不完整时标记FAIL并保留记录，第一组原配置会重新入队。浮点预测以float64文本导出，保留float32实际数值精度。

历史归档标签：archive-before-bayes-20261029；服务器独立压缩备份位于仓库外。未使用老师测试机会。
