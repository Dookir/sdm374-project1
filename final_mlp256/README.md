# 全数据最终256→128模型

保留 frozen_mlp256_cv10_20261030/ 整包及十折结果不变。本程序在全部2048条原数据内部做30组TPE、每组内部三折，共90次内部训练；完全沿用此前空间、种子和三折并行训练实现。根据内部验证MAE选定超参数，以其三个最佳轮数的中位数作为最终轮数，在全部2048条数据从头训练一次。最终阶段不再使用验证集、不早停，不因训练误差继续下降而加训，不做ramping寻优。

ReLU 256→128；学习率[0.0001,0.003]、AdamW衰减[0.0001,0.03]均对数采样，Dropout∈{0,0.05,0.1,0.2}；最多600轮、patience30、批量128、标准化MSE、clip5、余弦T_max600和eta_min=初始学习率×0.01。种子SEED=20261029，最终初始化SEED，独立批次种子SEED+100000；内部划分、TPE及内部训练种子沿用原方法，记录在数据分割/配置产物中。

只选一张启动时内存≤64MiB且利用率0%的GPU；无空闲卡即退出并暂停，不抢占、不另找第二张。任务使用该卡私有MPS与三个独立进程并行内部三折，减少CPU同步；最终全数据模型仍单进程训练。运行期间不自动迁移显卡。存在其他进程意外占用时不主动驱逐。

```bash
source server/activate.sh
bash final_mlp256/launch.sh --out final_mlp256/results/full_20261030
# 确认旧任务已退出后恢复
bash final_mlp256/launch.sh --out final_mlp256/results/full_20261030 --resume
CUDA_VISIBLE_DEVICES= python -m final_mlp256.check --out final_mlp256/results/full_20261030
CUDA_VISIBLE_DEVICES= python -m final_mlp256.check --out final_mlp256/results/full_20261030 --audit
```

结果只报告最终训练MAE和用于选配置的内部验证MAE，独立测试MAE明确为null。历史十折OOF MAE 4.561498是先前方法的参考，不能称为这个最终模型的测试成绩。内部验证值用于选配置，也不是新盲测。模型与标准化、配置、轮数、种子、数据/源码/旧文件哈希、全部试验和训练记录独立保存。老师可加载最终检查点进行预测；跨未知硬件从头训练不承诺逐位相等。未使用老师测试机会。
