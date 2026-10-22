# Lab6 离线预测与权重搜索

使用 2048 条已有实验，比较代理模型，并搜索可提交真实评估的楼宇控制权重。
**新权重的分数都是预测，不能当作真实仿真成绩。** 本程序不调用在线评估器。

## 修正版：统一比较，保留原始基准

**现在以 `solution.py` 为推荐入口。** 之前的树模型程序和结果保留，用于模型训练和诊断。
此前默认候选根据新模型的预测误差及预测值选出，无法保证在老师原模型上更好；距离限制
还排除了老师原方案。MAE 改善不能当作权重成绩改善。

```bash
cd "/Users/Admin/Documents/课程/机器学习/week7/offline"

# 所有方案都调用老师原 baseline.predict 和原 model.npz 进行同口径比较
python3 solution.py compare

# 重新进行 512 个随机起点及实测较好起点的约束寻优，保留原始权重作为基准
python3 solution.py optimize

# 对最终推荐使用同一个老师原模型估分
python3 solution.py score --weights revised_results/weights_recommended.json
```

`revised_results/weights_recommended.json` 仅在老师原模型预测严格降低超过 `1e-6` 时
替换原方案；否则保留老师原 `weights.json` 的原始内容，避免较差方案替换基准，也不把
数值舍入误差宣称为提升。多起点 SLSQP 没找到更低值，不等同于数学上证明了全局最优。

`revised_results/comparison.csv` 和 `comparison.json` 对所有权重统一列出：

- `teacher_prediction`：老师原模型预测，是当前本地比较的主列，越低越好。
- `teacher_prediction_delta`：相对老师原权重的变化，负数才表示同口径预测改善。
- `learned_model_prediction`：新模型的辅助预测，不作为第二套评分标准。
- `measured_ramping`：只有权重与训练记录完全相同时才有已有实测标签。
- `official_evaluator_score`：当前均为空，真实评估器尚未提供。

**老师原脚本的 `predict` 本身也是代理预测；最终官方成绩仍需要真实评估器。**
`solution.py` 不通过重训或改写老师模型来降低这列分数。

修正版还可输出 `weights_challenger.json`：在老师原模型预测增加不超过 5% 的范围内，
寻找新模型认为更可靠的邻域方案。该文件是明确标注的实验备选，不替换最终推荐，也不
宣称已经超过 baseline。用 `--teacher-slack 0` 可要求它的老师预测也不增加；若没有符合
条件的备选，不导出该文件。旧的约 103 分候选仍在比较表中，可直接看到其老师预测。

```bash
python3 -m unittest -v test_offline.py test_solution.py
```

修正版为单独脚本与输出目录，不修改此前训练代码、模型选择或内部检查记录。
`revised_results/run.json` 记录本次优化起点、主评分来源、是否有改善，以及输入文件校验值。

## 原模型训练与研究流程

当前使用默认 Python 环境，不创建虚拟环境。首次安装新增依赖：

```bash
python3 -m pip install catboost==1.2.10
cd "/Users/Admin/Documents/课程/机器学习/week7/offline"
```

依次运行四个命令：

```bash
# 五折调参、三种随机种子稳定性检查，锁定模型后做一次内部最终检查
python3 offline.py benchmark

# 根据锁定的选择，用全部 2048 行重新训练并保存
python3 offline.py train

# 随机搜索 10 万组，再做局部权重转移搜索
python3 offline.py optimize

# 查看候选方案的预测及风险提示
python3 offline.py predict --weights results/weights_conservative.json
python3 offline.py predict --weights results/weights_aggressive.json
```

命令不依赖当前目录，也可以用 `offline.py` 的绝对路径调用。默认数据是
`../Lab6/dataset_same8/train.csv`；所有输出位于本文件夹的 `results/`。
可用 `--out 另一个目录` 建立单独的实验，不会覆盖老师提供的文件。

## 如何理解算法

- HistGradientBoosting、CatBoost 为主力，ExtraTrees、RBF-SVR、RBF 核岭回归为对照。
- 原二次岭回归固定 `alpha=100`，作为公平比较的参考。
- 固定留出 410 行内部检查集，1638 行开发集参与调参。编号不作特征。
- 内部检查集与开发集来自同一八周环境，并不是额外的官方测试数据。
- 调参使用种子 2026 的五折验证；最好的三个非 baseline 模型及 baseline
  再用 2026、2027、2028 三个种子做五折稳定性验证。
- 每一折单独拟合预处理和模型。各模型族内、最终候选之间都先选 MAE
  不超过最小值 5% 的模型，再选各折预测前 10 名的真实平均分最低者。
  前 10 名的平均分相差不超过 0.1 kWh 时，选真实低分 20% 区域 MAE 更低者。
- 稳定性入围模型中，MAE 不超过最佳非 baseline 模型的 1.2 倍者参与复核。
  以 MAE 平方倒数归一化加权的集成也参与最终模型选择。若只有一个合格模型，
  不创建重复的集成；此时模型分歧为零，不代表预测没有误差。
- 开发集指标参与调参、集成权重和选择，因此有选择偏差。锁定选择后才评估内部检查集，
  不再用检查集改模型。之后用全量数据重新训练，不能再把这些训练行当作测试。

权重搜索先生成 Dirichlet 全局样本和实测最好 50 组附近的局部样本，之后在每栋楼
内部转移权重，保持 7 项非负且和为 1。树模型不直接使用梯度下降。

搜索目标为 **选定模型预测 + 系数 × 复核模型分歧**。稳健/进取方案系数分别是 1、0.25；
与最近训练数据的距离上限分别采用训练数据自身最近邻距离的第 50、95 百分位。
距离使用原始归一化权重的欧氏距离；不标准化后计算距离。
复核均值也会保存，但选择单一模型时不等于该模型的预测值。
**模型分歧不是经过校准的置信区间。**

保留已测最好方案；若找不到比它代理目标更低的新方案，会明确回退。进取方案排除
与稳健方案欧氏距离不超过 0.001 的新权重。数据支持半径只是经验筛选，不保证真实效果。

## 看哪些结果

| 文件 | 用途 |
| --- | --- |
| `results/trial_comparison.csv` | 每组参数的五折结果 |
| `results/model_comparison.csv` | 三种划分种子下的平均指标及集成对照 |
| `results/stability.json` | 各种子指标，便于检查稳定性 |
| `results/selection.json` | 在最终检查前锁定的模型、参数与选择依据 |
| `results/internal_check.json` | 选定模型与 baseline 的一次内部检查成绩 |
| `results/model.joblib` | 全量数据训练后的模型，仅加载自己生成的可信文件 |
| `results/weights_conservative.json` | 稳健候选，3×7 JSON |
| `results/weights_aggressive.json` | 进取候选，3×7 JSON |
| `results/weights_measured_best.json` | 数据中已测最好的权重，作为备选 |
| `results/candidates.json` | 候选预测、逐模型复核、分歧、距离、已知实测分数 |
| `results/candidate_shortlist.csv` | 两种搜索目标下的各前 100 个候选 |
| `results/manifest.json` | 数据校验值、开发/检查行编号、原始文件及代码校验值 |
| `results/environment.json` | 实验环境和包版本 |
| `results/verification.json` | 自动验收检查结果 |
| `results/source_revision.json` | 本次导出兼容性修复前后的代码校验值及冻结产物校验值 |

`MAE`、`RMSE` 越低越好；`R²` 越接近 1 越好。`top10_true_mean` 是各折中按预测
选出的前 10 个组合的真实平均分，再跨折求平均，越低说明筛选效果越好。
`top10_true_best_mean` 是各折所选 10 个中最好实测分数的跨折均值。
`low20_mae` 是被验证样本中真实分数最低 20% 的 MAE。

`candidates.json` 中 `measured_ramping: null` 表示没有真实成绩；只有与已提供训练行
完全相同的权重才附带已有实测标签。baseline 的 75.90 也是预测，不能用于认定胜出。

## 时间预算与恢复

默认调参 60 分钟、稳定性验证预留 20 分钟、搜索约 20 分钟；实际小数据可能更快完成。
时间预算在折与轮次之间检查，正在执行的拟合或批次会完成，不强行终止数值计算。
每个完成的五折实验立即保存；中断后重复相同 `benchmark` 命令即可复用缓存。
一次实验未完成的五折会重算。验证预算不足时保留进度，不提前偷看内部检查集。
选择锁定后再次运行只读取内部检查结果；全量模型存在时 `train` 不重复训练。
源代码、数据或原始文件校验值变化时须用新输出目录，防止混入旧结果。

例如减少第一轮搜索规模（另建目录，避免混用正式实验）：

```bash
python3 offline.py benchmark --out quick_results --trial-scale 0.1 --tune-minutes 10 --validation-minutes 10
```

正式运行不要利用内部检查成绩反复更换随机种子或挑结果。运行日志可重定向到本文件夹中，
例如 `python3 offline.py benchmark > benchmark.log 2>&1`。

## 验证程序

```bash
python3 -m unittest -v test_offline.py
```

共 9 项检查，覆盖数据分割、OOF 目标隔离、折内标准化、缓存恢复、预算到期、选择规则、
重复数据拒绝、权重约束与随机种子复现，以及包含原 baseline 预测的完整候选导出。
训练保存后还会自动检查重新载入的预测是否逐位一致。

本次实验在调参及内部检查完成后，修复了候选导出中 NumPy 2.x 不允许直接将一维数组
转换为标量的问题。修复仅将调用原 baseline 时的输入改为其支持的单组 `3×7` 格式。
代码修订已单独登记，锁定选择、内部检查预测、检查成绩及模型文件的校验值均保持不变，
没有重新调参或再次评估内部检查集。
