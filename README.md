# SDM374 Project 1 / Week7

保留自身预测 ramping 最低的五种不同网络，每种仅保留一份模型和对应权重。详见 [五种网络](neural_baseline/README.md)。

- Lab6/：老师原始材料、2048条数据和二次岭回归 baseline，保持原样。
- neural_baseline/：五种网络、权重、训练/搜索记录与CPU预测入口。
- baseline_restart/：历史回归对照。
- server/：服务器环境与依赖。

```bash
cd /data/zengxiong/wuxinkai/sdm374-project1
source server/activate.sh
CUDA_VISIBLE_DEVICES= python -m neural_baseline.retained --verify
```

各网络用自身模型预测 ramping，越低越好。旧网络与重复实验已移出项目归档，位置见 retained/manifest.json。保留老师两次测试机会，未提交。

新增实验：[数值编码残差MLP与TabM](tabular_next/README.md)，后台单卡训练；按自身预测ramping筛选，不替换现有五个网络。
