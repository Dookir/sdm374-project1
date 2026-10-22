# SDM374 Project 1 / Lab6

楼宇能源控制器的数据驱动参数整定作业。根据 2048 组控制权重与真实仿真指标数据，
学习代理模型，搜索三栋楼各 7 项控制权重，目标为降低第 1–8 周平均用电波动量（ramping）。
每栋楼权重非负且和为 1。

## 项目内容

```text
SDM374_Project1_v3.pptx   项目课件
Lab6/                    老师提供的原始材料
  baseline.py            二次岭回归训练、寻优与预测
  model.npz              原始预测模型
  weights.json           原始候选权重
  dataset_same8/         2048 条训练数据与说明
offline/                 自行实现的离线模型比较与搜索
  offline.py             树模型、核方法、交叉验证及约束搜索
  solution.py            原模型统一预测比较与保留原方案的实验入口
  test_offline.py         原流程测试
  test_solution.py        统一比较流程测试
  results/               参数实验、已训练模型、验证记录与候选权重
  revised_results/       修正版比较结果与候选
```

## 当前状态

- 已完成二次岭回归、CatBoost、HistGradientBoosting、ExtraTrees、SVR 和核岭回归的初步比较。
- 已保存参数实验、交叉验证、一次内部检查及权重搜索结果。
- CatBoost 在已有内部检查中的平均预测误差较小；这不能证明其候选权重的真实优化成绩更好。
- `solution.py` 是一次统一原模型预测口径的实验修正，不应将老师的代理预测当作其他模型优劣的最终裁判。
- 已实现六种可微神经网络：三档 MLP、瓶颈残差网络、楼宇分支网络和二次趋势残差网络；CPU 与 A100 结构检查通过，详见 [网络结构说明](neural/README.md)。
- **待开展：神经网络正式训练、神经网络/高斯过程/树模型的统一比较，以及以两次提交效果为主的嵌套验证。** 目前尚未据此选出最佳网络。
- 当前未包含老师的真实评估器，没有消耗真实提交机会，也没有验证新权重超过 baseline。

## 分数含义

训练表中的 `ramping` 是已提供的真实仿真结果。`baseline.py predict` 和自建模型输出的是预测，
二者均不能代替对新权重的真实评估。不同模型的最低预测值不能直接用于比较优化效果。
MAE、RMSE、R² 用于检查模型预测能力，不是课程最终成绩。

## 服务器开发环境

代码开发与实验主目录已迁移到 A100 服务器 `/data/zengxiong/wuxinkai/sdm374-project1`。
使用 Python 3.12 项目环境；进入仓库后执行 `source server/activate.sh`。
详见 [服务器环境说明](server/README.md)。本地已有实验和环境元数据保留为历史记录。

## 运行

当前实验使用默认 Python 3.14 环境，实际包版本记录于 `offline/requirements.txt` 与
`offline/results/environment.json`；老师原包的 Python 3.11 依赖说明保留在 `Lab6/`。

```bash
python3 -m pip install -r offline/requirements.txt

# 使用老师原模型预测原权重
python3 Lab6/baseline.py predict --weights Lab6/weights.json

# 统一列出已有方案在老师原模型和自建模型下的预测
python3 offline/solution.py compare

# 测试
cd offline
python3 -m unittest -v test_offline.py test_solution.py
```

模型训练、实验恢复及候选搜索命令详见 [离线实验说明](offline/README.md)。
已保存的 joblib 模型仅用于加载本项目自己的可信产物。

## 仓库范围

这是课程项目的私人备份，保留课件、原始数据、模型和实验记录。
缓存、临时文件、运行日志、虚拟环境及包含登录信息的本地服务器脚本不纳入版本控制。
