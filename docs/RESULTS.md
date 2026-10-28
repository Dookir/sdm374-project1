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
