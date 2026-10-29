# 三种MLP：10折交叉验证与权重寻优

本实验独立于2026-10-23全数据模型；原模型、候选和结果保持不变。只训练64→32、128→64、256→128，全部ReLU、无Dropout。种子20261029；三组共享10折与每折独立的初始化/批次打乱种子，每条数据各有一次留出测试。共30次从零训练和30次固定模型寻优。

## 固定配置与指标

- 原2048行按candidate_id排序；KFold(10, shuffle=True, random_state=20261029)。8折训练1843/测试205，2折训练1844/测试204。
- 每模型600轮、批128、AdamW lr0.001/wd0.001、余弦降至0.00001、梯度裁剪5；只在折训练数据拟合输入/标签标准化。
- 固定最后一轮，不用测试MAE选轮数、调参或提前停止。训练误差只是诊断。
- 训练集MAE按各折训练样本数加权，测试MAE合并各行唯一一次OOF预测；另报10折测试MAE样本标准差。
- 每折只用训练数据产生寻优参考、扰动中心和候选，不引用全数据旧模型候选。约6.5万候选、64起点、投影梯度800步、投影Adam1000步、权重转移、8起点SLSQP。CPU重算并只接受不退步的合法3×7权重。
- MAE用于预测能力诊断；自身预测ramping为候选比较成绩，分开汇总。无额外独立最终测试集，不消耗老师测试机会。

## 运行与恢复

从仓库根目录，使用原服务器已有环境，启动前选择一张空闲卡的完整UUID；一次只用一张卡，其他卡不可见。

```bash
source server/activate.sh
CUDA_VISIBLE_DEVICES= python -m unittest -v cv_experiment.test_cv
CUDA_VISIBLE_DEVICES=GPU_UUID python -u -m cv_experiment.run --out cv_experiment/results/cv10_20261029
```

中断后用同一配置和数据恢复（仍须一张当前空闲卡）：

```bash
CUDA_VISIBLE_DEVICES=GPU_UUID python -u -m cv_experiment.run --out cv_experiment/results/cv10_20261029 --resume
```

未完成训练的折从头重跑；已完成训练且文件一致的折直接继续寻优。每个阶段有完整标记和哈希，数据、配置、源码或完成阶段文件不匹配时拒绝恢复。已全部完成时--resume不会重复训练。CPU复核与查看状态：

```bash
CUDA_VISIBLE_DEVICES= python -m cv_experiment.check --out cv_experiment/results/cv10_20261029
CUDA_VISIBLE_DEVICES= python -m cv_experiment.check --out cv_experiment/results/cv10_20261029 --audit
```

长任务由后台tmux运行，30分钟自动化单次检查，正常时静默，完成/失败通知后停止；不持续跟踪。

## 文件结构

结果目录中的manifest.json保存配置、数据/源码哈希；splits.json保存每折索引及种子；models/<结构>/foldXX/保存模型、600轮训练历史、训练/测试逐样本预测、训练/寻优完成标记、新权重和搜索详情。fold_metrics.json记录30项指标，oof_predictions.csv含3×2048行唯一留出预测，summary.json/csv与README.md分别列MAE和预测ramping。

best/<结构>/保存该结构10折中自身预测ramping最低的模型和对应权重，不按最低测试MAE挑选。直接预测：

```bash
python -m cv_experiment.predict --checkpoint cv_experiment/results/cv10_20261029/best/relu_64_32/model.pt --weights cv_experiment/results/cv10_20261029/best/relu_64_32/weights.json
```

本次统一600轮/wd0.001与旧保留模型的600/300/100轮及部分不同正则化不完全相同，不能把新旧误差差异全部归因于划分方式。
