# 固定11个模型加强寻优

[首页](../README.md) · [直接复现40.4908](../docs/QUICKSTART.md) · [完整分数表](../docs/RESULTS.md)

## 做了什么

2026-10-23已完成全部11个模型的加强寻优：5种保留MLP，加上最新6种配置各自原先预测最低的检查点。固定网络参数，只搜索控制权重；没有重训、没有覆盖原权重。每个模型使用自身预测ramping评分，TabM平均全部分支。

每模型评估约6.5万混合候选：不同浓度Dirichlet、稀疏面、343个精确顶点、优秀样本与原候选附近扰动。保留64个不同起点，三档步长投影梯度800步、三档步长投影Adam1000步，再做权重转移与8起点双精度CPU SLSQP精修。最终CPU float32重算，下降才接受；原候选始终保留。

## 最新最好结果

**64→32 MLP：40.70169067 → 40.49078369，下降0.21090698。**

配对文件：

- 模型：`neural_baseline/retained/relu_64_32/model.pt`。
- 新权重：`weight_search/results/run_20261023/weights/relu_64_32.json`。

从仓库根目录直接复现：

```bash
python -m neural_baseline.retained --name relu_64_32 --weights weight_search/results/run_20261023/weights/relu_64_32.json
python -m weight_search.check --audit
```

两条都默认CPU，不启动搜索。第二条核对11组新权重、前后分数、原模型/权重/材料和源文件哈希；不代表存在独立验证集。

## 结果在哪里

|路径|内容|
|---|---|
|results/run_20261023/comparison.json|11组完整前后分数|
|results/run_20261023/weights/|新的3×7权重JSON，文件名对应模型名|
|results/run_20261023/details/|每个模型的分阶段搜索记录|
|results/run_20261023/manifest.json|固定检查点、旧权重路径、参数和哈希|
|results/run_20261023/complete.json|完成汇总与最好结果|
|results/run_20261023/audit.json|11项CPU复核通过记录|

模型与新权重的完整对应关系见[结果表](../docs/RESULTS.md)。其他模型的新权重不能默认交给64→32模型预测。

## 重新寻优

只有确实要开展新搜索时才执行。从原服务器仓库根目录选择一张空闲卡，使用新输出目录：

```bash
source server/activate.sh
CUDA_VISIBLE_DEVICES=GPU_UUID python -u -m weight_search.run --out weight_search/results/team_run_20261028
```

将 `GPU_UUID`替换为完整空闲卡UUID，目录名按本次实验唯一命名。运行范围是上述11个固定检查点，不是对72个检查点全部重搜。一次仅一张卡，不与本项目训练任务同时运行。

**当前 `weight_search.check` 的路径固定为原 `run_20261023`，没有 `--out` 参数。** 查看新实验请读取新目录的 `status.json`、`comparison.json`、`complete.json`或`failure.json`；不要把原目录的审计结果当成新实验已验证。代码的这项限制如实保留，本文没有修改算法或脚本。

CPU测试入口：`python -m unittest -v weight_search.test_search`。该测试用合成函数检查投影与寻优行为，不调用真实评测、不使用GPU。
