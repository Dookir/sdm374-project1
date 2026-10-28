# 数值编码残差MLP与TabM

[首页](../README.md) · [统一方法](../docs/WORKFLOW.md) · [最新结果](../docs/RESULTS.md)

## 已完成的实验

2026-10-23完成6配置×3种子，共18次训练。每次800轮，保存50/150/400/800轮，共72组模型和候选。训练用全部2048行原始ramping标签，未划分独立验证集。训练误差仅用于诊断；主成绩始终是各模型自身预测ramping。

|配置名|方法|编码|宽度/残差块或TabM层数|TabM分支|
|---|---|---|---|---:|
|res_periodic128|残差MLP|周期，8维|128 / 2|—|
|res_ple128|残差MLP|分段线性，8维|128 / 2|—|
|res_periodic256|残差MLP|周期，16维|256 / 3|—|
|tabm_raw128_k8|官方TabM|无|128 / 2|8|
|tabm_ple128_k16|官方TabM|分段线性，8维|128 / 2|16|
|tabm_periodic256_k16|官方TabM|周期，16维|256 / 2|16|

残差MLP使用LayerNorm与SiLU。TabM使用[官方实现](https://github.com/yandex-research/tabm)，训练时逐分支计算MSE，预测和寻优时才平均；编码使用[官方数值编码包](https://github.com/yandex-research/rtdl-num-embeddings)。依赖固定在 `requirements.txt`。

## 找到结果并直接预测

结果目录：`results/run_20261023/`。`comparison.json`记录72个自身预测分；`best/resmlp/`与`best/tabm/`各保存本轮最好模型、权重和选择记录。原训练寻优结果最低分别为105.6795、91.7101。后续加强寻优结果在 `weight_search/`，残差MLP最低105.6542，TabM最低仍为91.7101。

从仓库根目录执行，默认CPU：

```bash
python -m tabular_next.predict --checkpoint tabular_next/results/run_20261023/best/tabm/model.pt --weights tabular_next/results/run_20261023/best/tabm/weights.json
python -m tabular_next.check --audit
```

第一条应约为91.7101。`predict`只加载指定模型和权重；`check --audit`核对原72组候选的合法性、两类导出模型、源文件及受保护文件的哈希，不是独立测试集评估。

## 重新训练前再读这里

训练和预测是不同入口。运行下面命令会开始新的GPU训练，不是查看旧结果。先读[单卡服务器指南](../server/README.md)，选择一张空闲卡；不要同时启动两个任务。

```bash
source server/activate.sh
# 将GPU_UUID替换成nvidia-smi查询到的完整空闲卡UUID
CUDA_VISIBLE_DEVICES=GPU_UUID python -u -m tabular_next.run --out tabular_next/results/team_run_20261028
```

该新目录必须尚不存在实验manifest；日期示例可改为本次实验唯一名称。默认仍跑18次训练、72组候选，结果在新目录。查看新实验时指定同一目录：

```bash
python -m tabular_next.check --out tabular_next/results/team_run_20261028
```

`check --audit`的数量检查按原6×3×4设计；如果改种子或检查点数量，现有审计器不能直接当作通用验证器。代码文件在训练开始时记入哈希，实验期间不要修改受保护的 `.py` 文件。

## 原实验文件说明

|文件|内容|
|---|---|
|manifest.json|配置、种子、依赖、数据/代码哈希、对照分|
|models/<配置_种子>/epoch*.pt|冻结模型与标准化/编码统计量|
|weights/*.json|每个检查点生成的3×7控制权重|
|comparison.json|全部72项预测分|
|best/|按本轮自身预测分选出的两类最好候选|
|complete.json / status.json|完成汇总/状态快照|
|audit.json|原结果的CPU复核记录|

CPU检查：`python -m unittest -v tabular_next.test_models`。负预测如出现会原样标记，不截断或隐瞒。本轮72组没有负预测，没有使用老师测试机会。
