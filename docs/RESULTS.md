# 最新结果与模型/权重对应表

[首页](../README.md) · [快速开始](QUICKSTART.md) · [方法说明](WORKFLOW.md)

数据来自2026-10-23完成的实验，文档整理于2026-10-28。统一主成绩为每个保存模型自身预测的ramping，越低越好。表中前后使用同一模型，变化仅来自输入控制权重寻优。

## 11个固定模型加强寻优结果

|模型/配置|原预测分|加强寻优后|下降量|
|---|---:|---:|---:|
|relu_64_32|40.7017|40.4908|0.2109|
|relu_128_64|53.9747|53.8163|0.1584|
|relu_256_128|60.5684|60.5461|0.0223|
|relu_512_256_128|69.5564|69.5466|0.0098|
|relu_1024_512_256|71.9600|71.9598|0.0003|
|res_periodic128|106.8607|106.8590|0.0017|
|res_ple128|106.8077|106.8071|0.0007|
|res_periodic256|105.6795|105.6542|0.0253|
|tabm_raw128_k8|91.7101|91.7101|0.0000|
|tabm_ple128_k16|98.1219|98.1219|0.0000|
|tabm_periodic256_k16|98.1162|97.4020|0.7142|

原baseline自身预测为75.8957，属于单独的二次岭回归对照，不在这11个神经网络中。

当前最好为 **relu_64_32，40.49078369**，较其原候选40.70169067降低0.21090698。模型种子2026，保留第600轮检查点，隐藏层64→32，输入21、输出1。加强寻优没有更新这个网络的参数。

## 每个结果对应哪两个文件

下表的每行都要配对使用。链接可直接打开文件；所有路径相对于仓库根目录。

|配置|固定模型|加强寻优后的权重|
|---|---|---|
|relu_64_32|[model.pt](../neural_baseline/retained/relu_64_32/model.pt)|[权重JSON](../weight_search/results/run_20261023/weights/relu_64_32.json)|
|relu_128_64|[model.pt](../neural_baseline/retained/relu_128_64/model.pt)|[权重JSON](../weight_search/results/run_20261023/weights/relu_128_64.json)|
|relu_256_128|[model.pt](../neural_baseline/retained/relu_256_128/model.pt)|[权重JSON](../weight_search/results/run_20261023/weights/relu_256_128.json)|
|relu_512_256_128|[model.pt](../neural_baseline/retained/relu_512_256_128/model.pt)|[权重JSON](../weight_search/results/run_20261023/weights/relu_512_256_128.json)|
|relu_1024_512_256|[model.pt](../neural_baseline/retained/relu_1024_512_256/model.pt)|[权重JSON](../weight_search/results/run_20261023/weights/relu_1024_512_256.json)|
|res_periodic128|[model.pt](../tabular_next/results/run_20261023/models/res_periodic128_2028/epoch50.pt)|[权重JSON](../weight_search/results/run_20261023/weights/res_periodic128.json)|
|res_ple128|[model.pt](../tabular_next/results/run_20261023/models/res_ple128_2028/epoch150.pt)|[权重JSON](../weight_search/results/run_20261023/weights/res_ple128.json)|
|res_periodic256|[model.pt](../tabular_next/results/run_20261023/models/res_periodic256_2026/epoch50.pt)|[权重JSON](../weight_search/results/run_20261023/weights/res_periodic256.json)|
|tabm_raw128_k8|[model.pt](../tabular_next/results/run_20261023/models/tabm_raw128_k8_2028/epoch50.pt)|[权重JSON](../weight_search/results/run_20261023/weights/tabm_raw128_k8.json)|
|tabm_ple128_k16|[model.pt](../tabular_next/results/run_20261023/models/tabm_ple128_k16_2026/epoch50.pt)|[权重JSON](../weight_search/results/run_20261023/weights/tabm_ple128_k16.json)|
|tabm_periodic256_k16|[model.pt](../tabular_next/results/run_20261023/models/tabm_periodic256_k16_2027/epoch400.pt)|[权重JSON](../weight_search/results/run_20261023/weights/tabm_periodic256_k16.json)|

对于5种MLP：使用 `python -m neural_baseline.retained --name <配置名> --weights <该行权重路径>`。对于另6种配置：使用 `python -m tabular_next.predict --checkpoint <该行模型路径> --weights <该行权重路径>`。

## 实验阶段与原始记录

|阶段|范围|完成状态|记录|
|---|---|---|---|
|保留MLP|5种不同结构，各留一个检查点|已完成|[模型清单](../neural_baseline/retained/manifest.json)|
|编码残差MLP与TabM训练|6配置×3种子，4检查点，共72候选|已完成|[完整比较](../tabular_next/results/run_20261023/comparison.json)、[完成记录](../tabular_next/results/run_20261023/complete.json)|
|固定模型加强寻优|5种MLP＋6种新配置的原最低分检查点，共11个|已完成|[完整前后分数](../weight_search/results/run_20261023/comparison.json)、[完成记录](../weight_search/results/run_20261023/complete.json)|

旧MLP的其它检查点已归档，不在主分支批量保留。11个加强寻优模型不等于对全部72个检查点逐一重搜。

## 如何核对结果

从仓库根目录运行：

```bash
python -m weight_search.check --audit
```

该命令核对11个结果，正常末尾显示 `passed: true`。已有[审计记录](../weight_search/results/run_20261023/audit.json)包含：权重合法、预测可复现、分数未退步、源文件和受保护文件未变。部分分数最后几位受float32及CPU/GPU计算差异影响，微小差异不应夸大为效果改善。

TabM/残差网络训练结果的[审计记录](../tabular_next/results/run_20261023/audit.json)与本轮加强寻优审计分开保存。所有预测标签保持 `predicted_ramping`，未使用老师两次测试机会；最近两轮神经网络没有独立留出集评估，详见[方法说明](WORKFLOW.md)。

## 2026-10-29：三种MLP的10折交叉验证

本轮三种网络均使用相同600轮配置，各训练10次，从零初始化。三组共享折划分，每折约90%训练、10%测试。训练MAE按训练样本数加权；OOF测试MAE合并2048行唯一留出预测，标准差为10折测试MAE的样本标准差。测试结果未参与超参数、轮数或候选选择。

|模型隐藏层|训练MAE|OOF测试MAE|各折测试MAE标准差|
|---|---:|---:|---:|
|64→32|0.818004|7.472761|0.543856|
|128→64|0.012963|6.787915|0.694987|
|256→128|0.000193|5.889622|0.439818|

|模型隐藏层|10折最低自身预测ramping|均值|标准差|
|---|---:|---:|---:|
|64→32|41.047272|50.752283|6.004521|
|128→64|45.522560|60.754009|7.689765|
|256→128|58.850212|69.952431|4.427000|

256→128的OOF测试MAE最低；64→32的自身预测ramping最低。这是两类不同指标，不互相替代。新折模型与原全数据模型分开保存，旧40.4908结果仍保留。30组最终候选均无负预测。

模型、折划分、逐样本预测、候选权重与汇总见[本轮结果目录](../cv_experiment/results/cv10_20261029/README.md)，[OOF预测](../cv_experiment/results/cv10_20261029/oof_predictions.csv)、[汇总JSON](../cv_experiment/results/cv10_20261029/summary.json)及[CPU审计记录](../cv_experiment/results/cv10_20261029/audit.json)。运行方法见[CV实验说明](../cv_experiment/README.md)。仅使用一张GPU，无老师测试提交；后台任务与自动检查已结束。

## 2026-10-29：原配置加早停

|模型|拟合训练MAE|验证MAE|早停OOF测试MAE|原600轮测试MAE|平均选中/停止轮数|最低自身预测ramping|
|---|---:|---:|---:|---:|---:|---:|
|64→32|3.5782|6.4160|6.7034|7.4728|68.0 / 98.0|57.5980|
|128→64|2.8073|6.0691|6.6656|6.7879|50.8 / 80.8|63.4854|
|256→128|1.7980|5.9339|6.2450|5.8896|63.8 / 93.8|60.5976|

64→32和128→64的OOF测试MAE分别降低0.7693和0.1223；256→128增加0.3554。三组均在全部10折触发早停，选中模型平均约51～68轮，实际停止平均约81～98轮。各组最低自身预测ramping均未改善对应原600轮结果；仍保留原有候选。30组新候选均无负预测。

本轮外层十折与原实验完全相同；每折90%内部再留10%作验证，实际约81%拟合/9%验证/10%测试，原固定600轮实验为90%拟合/10%测试，所以变化不能全部归因于早停。标准化和寻优只使用实际拟合子集，测试不参与模型选择。平均选中轮数是恢复检查点所在轮数，停止轮数还包含30轮无改善等待。

保留原学习率、权重衰减0.001、无Dropout、批量128、最多600轮，仅新增内部验证早停。详见[结果README](../earlystop_experiment/results/es10_20261029/README.md)、[汇总JSON](../earlystop_experiment/results/es10_20261029/summary.json)与[审计记录](../earlystop_experiment/results/es10_20261029/audit.json)。MAE作为预测诊断与自身预测ramping分开记录，未使用老师测试机会。

## 2026-10-29：贝叶斯调参及单卡三折并行

|模型|本轮训练MAE|原600轮OOF MAE|早停OOF MAE|本轮OOF MAE|本轮折MAE标准差|
|---|---:|---:|---:|---:|---:|
|64→32|2.545762|7.472761|6.703445|4.612087|0.459250|
|128→64|1.944671|6.787915|6.665582|4.514524|0.327055|
|256→128|1.350317|5.889622|6.245016|4.561498|0.311983|

|模型|原600轮最低自身预测ramping|早停最低自身预测ramping|本轮最低自身预测ramping|本轮最佳折|
|---|---:|---:|---:|---:|
|64→32|41.047272|57.598030|84.638474|5|
|128→64|45.522560|63.485405|92.961670|7|
|256→128|58.850212|60.597565|86.392906|3|

三种结构的OOF测试MAE均低于原固定600轮和原配置早停实验；128→64本轮最低，为4.514524。候选按各模型自身预测ramping比较：本轮最低84.638474（64→32，第5折），未超过原600轮最佳41.047272或历史全数据最佳40.490784，因此保留旧推荐模型与权重。MAE与自身预测ramping分别展示，不使用baseline统一评分。

本轮30份候选均无负预测。

|模型|平均重训轮数（范围）|Dropout选择次数|学习率范围|权重衰减范围|
|---|---:|---|---:|---:|
|64→32|243.0（167–285）|0.2×9、0.1×1|0.00243912–0.00299188|0.000204154–0.0264228|
|128→64|215.5（131–293）|0.2×10|0.00204516–0.00296697|0.000229454–0.0270478|
|256→128|203.2（170–275）|0.2×10|0.00181148–0.00293857|0.000843976–0.0266097|

每个结构、每个外层折独立30组TPE配置试验，内部三折选择超参数与最佳轮次；以三个内部最佳轮次的中位数，在完整外层90%数据重新训练。标准化只使用对应训练子集，测试标签不参与选择或寻优。外层沿用历史十折，因此属于同划分改进比较，并非全新盲测。

原配置早停实验只使用约81%数据拟合；本轮最终模型与原600轮模型均使用外层90%数据拟合。不能将不同实验的全部变化单独归因于Dropout、早停或权重衰减。

共900组配置试验、2700次内部训练、30个最终模型和30次寻优。后半程迁移并复用16个模型和1482次已完成内部训练，7110个继承文件哈希未变；原串行程序及输出、仓库外快照保留。内部三折以三个独立进程和私有MPS只在同一张A100上并行。80轮短程对照加速2.88–3.01倍，逐轮验证与最终模型参数逐位一致；此倍数不代表完整任务的总加速。CPU审计通过，私有MPS已自动退出，未使用老师测试机会。

结果见[本轮结果README](../bayes_accel/results/bo10_20261029/README.md)、[逐折配置](../bayes_accel/results/bo10_20261029/selected_configurations.csv)、[OOF预测](../bayes_accel/results/bo10_20261029/oof_predictions.csv)、[审计记录](../bayes_accel/results/bo10_20261029/audit.json)。复现见[单卡并行程序说明](../bayes_accel/README.md)。
