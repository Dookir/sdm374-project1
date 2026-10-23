# 固定模型加强寻优

11个固定检查点：保留的5种MLP，加上最新6种配置各自原先自身预测最低的检查点。不重新训练，不改变checkpoint，不消耗老师测试机会。新权重另存，原有候选保持不变。

每模型约6.5万混合候选，包括不同Dirichlet浓度、稀疏面、343个精确顶点、优秀样本附近及原候选附近扰动；原候选始终纳入。选择64个有差异的起点，三档步长投影梯度800步、三档自适应步长投影Adam1000步，再做权重转移局部搜索和8个起点的双精度CPU SLSQP。最终用原模型float32在CPU重算，下降才保留，防止丢失原最好结果。

唯一主成绩是每个模型自身预测ramping，越低越好；TabM始终平均所有分支。不改变模型标签、输出缩放或评分方式。一次只使用一张启动时空闲的GPU，CPU复核不触及其他卡。后台tmux运行，自动化单次检查，完成时输出前后比较。

```bash
source server/activate.sh
CUDA_VISIBLE_DEVICES= python -m unittest -v weight_search.test_search
CUDA_VISIBLE_DEVICES=<GPU_UUID> python -u -m weight_search.run
CUDA_VISIBLE_DEVICES= python -m weight_search.check --audit
```
