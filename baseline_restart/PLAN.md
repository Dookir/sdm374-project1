# 按原 baseline 流程重新开始（2026-10-23）

## 固定规则

当前唯一方案排序依据为原 `Lab6/baseline.py predict --weights ...` 的 `predicted_ramping`，越小越好。评分始终使用原 `Lab6/model.npz`，不将生成模型传给评分命令，不重训练或覆盖评分模型。MAE、R²、遗憾值、自建模型自评分不参与最终排序。

原附带模型 alpha=100；原训练入口在固定五折中选择 alpha，重跑可能选择1000。这是生成模型的变化，评分模型始终固定。原训练代码自带的交叉验证按原样执行，不另建一套选优体系。

## 课程算法依据

- week1/SDM374_1b.pptx：20–25页搜索与梯度下降，37页线性/多项式回归作业。
- week2/SDM374_2a.pptx：29–30页批量/随机梯度下降，31–38页正规方程、伪逆及与梯度下降的比较。
- week2/SDM374_2b.pptx：5–10页标准化、多项式特征；18–24页 L2 正则化及正则化回归。
- week2/SDM374_Lab2.pptx：第11页多元线性、多项式回归、L1/L2分析。
- week7/SDM374_Project1_v3.pptx：第8–9页为训练代理模型→梯度寻优→评估权重的原始流程。第5/7页说明真实黑盒在第8周提供；当前脚本分数按其实际输出记录，日后可在相同流程末端接入黑盒。

## 第一轮顺序

1. 原样复测原权重，并使用原附带模型调用原 optimize，确认起点分数可复现。
2. 原样执行 train→optimize→predict，生成模型保存至新目录。
3. 从前两周算法开始：一次线性最小二乘、二次多项式最小二乘、二次 Ridge（0.1/1/10/100/1000）、二次 Lasso（0.01/0.1/1）。所有模型用2048行和相同特征顺序训练，标准化及参数导出沿用原 baseline。
4. 每个模型都由原 optimize 生成3×7权重，再由原 predict 计算分数。候选之间只按该输出比较。
5. 固定原模型系数与原梯度算法，仅扩展起点：原16起点、原权重、343个顶点、1024个不同浓度随机起点，检查原16起点是否漏掉更低点。
6. 只接受评分下降超过1e-8的新权重；无改善就保留原权重，明确报告没有超过baseline。

这轮先不用树模型、核方法和神经网络，不自动进入在线阶段或真实提交。全程CPU即可。

## 回档与交付

旧 `offline/`、`neural/` 已移出活动项目，服务器保存在 `/data/zengxiong/wuxinkai/archive-before-baseline-restart-20261023/`，Git标签为 `archive-before-baseline-restart-20261023`。老师材料原样保留。

交付为生成参数、候选权重、原命令与stdout、唯一评分比较表、推荐权重和完整性校验。不开新评分体系，不写课程报告。

## 运行

```bash
cd /data/zengxiong/wuxinkai/sdm374-project1
source server/activate.sh
CUDA_VISIBLE_DEVICES= python -u -m baseline_restart.run
```

结果目录 `baseline_restart/results/first_run_20261023/`。每条已完成结果立即保存，中断后保留。改配置重跑需新目录；不覆盖原文件。
