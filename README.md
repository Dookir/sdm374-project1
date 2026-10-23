# SDM374 Project 1 / Week7

当前按最新指令训练神经网络，并固定老师评分结构：**训练网络参数 → 投影梯度寻找3×7权重 → 原 predict 评分**。所有候选只按原评分脚本输出排序。

- `Lab6/`：老师原始模型、程序、权重与2048条数据，保持原样。
- `neural_baseline/`：当前神经网络训练、权重寻优与固定原程序评分，详见 [网络计划](neural_baseline/PLAN.md)。
- `baseline_restart/`：保留的课程回归对照与原程序复现。
- `server/`：服务器环境与激活入口。
- `SDM374_Project1_v3.pptx`：项目课件。

详细方案、前两周课程算法对应页码和运行步骤见 [重新开始的计划](baseline_restart/PLAN.md)。

```bash
cd /data/zengxiong/wuxinkai/sdm374-project1
source server/activate.sh
CUDA_VISIBLE_DEVICES= python -u -m baseline_restart.run
```

所有候选评分固定调用 `python Lab6/baseline.py predict --weights <候选.json>`，评分模型始终为原 `Lab6/model.npz`。生成参数与评分参数分别保存，原文件不覆盖。

此前树模型、神经网络及其结果已退出当前工作目录。可从 Git 标签 `archive-before-baseline-restart-20261023` 或服务器同名归档文件夹恢复。

第一轮已完成14个对照，尚未超过原baseline；推荐权重保留原方案。结果见 [原程序评分对照](baseline_restart/results/first_run_20261023/README.md)。

当前神经网络训练入口：`python -u -m neural_baseline.run`。启动前必须选择一张空闲GPU的完整UUID。回归入口仍只需CPU。
