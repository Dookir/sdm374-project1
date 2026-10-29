# 原配置＋早停：三种MLP十折实验

原64→32、128→64、256→128三组结构及学习率0.001、AdamW权重衰减0.001、批128、ReLU、无Dropout、梯度裁剪5、600轮余弦调度保持不变。最大训练600轮；每轮按内部验证集原单位MAE检查，连续30轮没有严格下降则停止，恢复验证MAE最低的检查点，不使用最后一轮冒充最佳模型。

## 数据使用

外层十折完全沿用 `cv_experiment/results/cv10_20261029/splits.json`。每折90%内部再随机留10%作验证，三结构共享该划分及初始化/批次种子：8折1658拟合/185验证/205测试，2折1659拟合/185验证/204测试。总计约81%拟合、9%验证、10%外层测试。标准化只拟合实际梯度训练子集；验证集只用于选停止点，外层测试只在保存最佳模型后评估。寻优参考/扰动中心也只取实际拟合子集。

本轮不在全部90%外层训练行重新拟合；因此对比原90%拟合、固定600轮实验时，训练样本量也发生变化，不能把差异全部归因于早停。三组各10折，共30次训练和30次固定检查点寻优。原结果不覆盖，不消耗老师测试机会。

## 运行

```bash
source server/activate.sh
CUDA_VISIBLE_DEVICES= python -m unittest -v earlystop_experiment.test_earlystop
CUDA_VISIBLE_DEVICES=GPU_UUID python -u -m earlystop_experiment.run --out earlystop_experiment/results/es10_20261029
# 中断后校验原数据、配置、代码与已完成阶段文件再恢复
CUDA_VISIBLE_DEVICES=GPU_UUID python -u -m earlystop_experiment.run --out earlystop_experiment/results/es10_20261029 --resume
# 状态为单次轻量读取；完成后在CPU复核全部结果
CUDA_VISIBLE_DEVICES= python -m earlystop_experiment.check --out earlystop_experiment/results/es10_20261029
CUDA_VISIBLE_DEVICES= python -m earlystop_experiment.check --out earlystop_experiment/results/es10_20261029 --audit
```

仅选择一张启动时空闲GPU的完整UUID；后台tmux执行，30分钟自动化单次检查，正常运行时静默。未完成训练的折重新从零训练，完成训练的折直接继续寻优，不恢复不完整的优化器状态。

## 交付与成绩

结果保存30个最佳验证检查点、逐轮验证MAE、实际训练/选中轮数、拟合/验证/测试逐样本预测、OOF预测、30份新控制权重，以及与原固定600轮的比较表。训练MAE只统计实际拟合子集，验证MAE统计所选模型的内部验证误差；外层OOF测试每样本一次。

MAE用于分析预测能力；候选仍按自身预测ramping比较。每折模型先按验证MAE选定，跨折导出的best/再按自身预测ramping选候选，外层测试MAE不用于任何选择。

```bash
python -m earlystop_experiment.predict --checkpoint earlystop_experiment/results/es10_20261029/best/relu_64_32/model.pt --weights earlystop_experiment/results/es10_20261029/best/relu_64_32/weights.json
```
