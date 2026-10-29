# 单张A100上的内部三折并行续训

保持bayes_experiment原始源码、分割、种子、900组试验预算、AdamW/Dropout/早停和ramping评分方法不变。新执行器将每组参数的三个内部折交给三个spawn独立进程，进程各自持有模型、优化器、随机数和早停状态；通过私有目录的NVIDIA MPS只共享原GPU UUID。配置试验仍按原TPE顺序，三个内部结果全部返回后才选择下一组。

训练仅减少CPU同步：每批损失留在GPU按原float64加权顺序累计，每轮合并一次回传；非有限值检查延迟到轮末。最佳检查点在GPU克隆，最终正常导出CPU模型。保留float32、关闭TF32、不改变批量128和学习率轨迹。取消每个内部折后的强制GC/empty_cache，保持缓存供下一次使用。

迁移从停止状态的串行实验快照复制，新manifest记录原manifest与所有已完成阶段文件哈希，完成的模型、预测、参数试验和内部折不重算、不改写。旧程序原目录保持不变。旧进程没有完整epoch检查点，迁移时未完成的一个内部折按原种子重跑；新执行器支持pause.request在轮次边界保存模型/AdamW/调度器/早停/CPU与GPU随机状态，--resume精确恢复。

```bash
# 在服务器仓库根目录激活环境；只向工作进程暴露同一完整GPU UUID
bash bayes_accel/launch.sh --out bayes_accel/results/bo10_20261029 --resume
CUDA_VISIBLE_DEVICES= python -m bayes_accel.check --out bayes_accel/results/bo10_20261029
CUDA_VISIBLE_DEVICES= python -m bayes_accel.check --out bayes_accel/results/bo10_20261029 --audit
CUDA_VISIBLE_DEVICES= python -m bayes_accel.predict --checkpoint <模型> --weights <JSON>
CUDA_VISIBLE_DEVICES= python -m unittest bayes_accel.test_accel -v
```

运行需由启动脚本设置仅绑定该卡的CUDA_VISIBLE_DEVICES、CUDA_MPS_PIPE_DIRECTORY和CUDA_MPS_LOG_DIRECTORY；不修改全局MPS服务或显卡计算模式。完成后关闭本任务私有MPS服务。benchmark.json记录原串行、减少同步串行、内部三折并行耗时及逐轮验证数值一致性。GPU利用率是观测值，不人为增加无效计算追求100%。

80轮×三个内部折速度对照：三结构加速2.88～3.01倍，模型参数及逐轮验证MAE逐位一致；CPU暂停恢复及GPU 64→32暂停恢复通过。上述倍数为短程训练对照，不含全部TPE/导出/寻优开销，不等于总任务缩短同样倍数。
