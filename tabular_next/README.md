# 数值编码残差 MLP 与 TabM

主成绩仅为各模型自身 predicted_ramping，越低越好。保留原有五个网络作为对照，当前最低40.70169067。

六个配置、三个随机种子、800轮；检查点50/150/400/800，共18次训练、72组模型/权重候选。残差MLP比较周期编码128/256和分段线性编码128；官方TabM比较原始输入128/k8、分段线性编码128/k16、周期编码256/k16。TabM逐分支计算训练MSE，推理和寻优才取分支平均。训练标签为原CSV ramping，不使用baseline标签；训练误差仅诊断，不参与排名。

每个检查点由自身模型做多起点投影梯度与投影Adam搜索。每栋楼权重非负且和为1。每类模型最低自身预测候选导出到结果目录best/，不更改现有五个模型。负预测原样标记negative_prediction供分析，不截断或隐瞒。

仅允许一张启动时空闲的GPU（完整UUID）；显存上限10%，其他GPU不可见。tmux运行与failure/status/complete文件支持自动化单次检查，不持续跟踪。

```bash
source server/activate.sh
CUDA_VISIBLE_DEVICES= python -m unittest -v tabular_next.test_models
# 训练须指定一张空闲卡UUID
CUDA_VISIBLE_DEVICES=<GPU_UUID> python -u -m tabular_next.run
# 自动化只读快照；完成后加 --audit 在CPU复核导出
CUDA_VISIBLE_DEVICES= python -m tabular_next.check
```

参考官方实现：https://github.com/yandex-research/tabm 与 https://github.com/yandex-research/rtdl-num-embeddings 。依赖版本见requirements.txt；不使用第三方服务，不消耗老师测试机会。
