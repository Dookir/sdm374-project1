# 冻结版：256→128 MLP，10折原模型与全部训练配方

这是2026-10-29贝叶斯调参实验的256→128十个最终折模型。2026-10-30按用户要求冻结，不重新训练、不再选择参数、不继续寻优。主要指标为留出测试MAE。原OOF测试MAE为 **4.5614982095**，训练MAE为1.3503167219，折测试MAE标准差0.3119828717。

4.5615是2048条唯一留出预测合并后的十折结果，不属于某一折，也不是十模型平均预测得到的测试分。十折各有独立参数、初始化与批次种子，不能混用。所有原始模型文件逐字节复制；本包不把测试MAE最低的折挑成正式单模型。

## 老师先验证保存的模型（不训练、不需要GPU）

解压后进入本目录。推荐Python3.12，参考精确版本与Linux/A100环境见environment.json。使用现成环境即可，不自动创建环境。

```bash
python -m pip install -r requirements.txt
python verify.py
```

verify.py验证整个冻结包的SHA256、数据编号、训练/测试不相交、训练集标准化、10模型原始参考预测及2048条OOF覆盖。返回passed=true即表示保存模型在本机通过预测核对。预测允许绝对误差1e-3（原ramping单位），这只是推理兼容性阈值，不是跨硬件重训成功的保证。服务器实际复核差异见verification.json。

## 老师从头重训指定折

不重新运行TPE，不使用测试集选择参数或轮数，直接读取recipes.json中的已冻结配方。程序顺序运行所选折，只用一张卡，输出到单独目录，拒绝覆盖既有折。

```bash
# 使用与参考环境匹配的NVIDIA显卡，尽可能复现GPU训练
# 先匹配PyTorch 2.5.1+cu121；参考安装命令（Linux/Windows CUDA机器）：
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python retrain.py --fold 1 --device cuda --gpu 0 --out retrained
# 十折依次重训
python retrain.py --all --device cuda --gpu 0 --out retrained_all
# 没有NVIDIA显卡时可以运行CPU版本，但不保证与GPU训练结果一致
python retrain.py --fold 1 --device cpu --out retrained_cpu
```

固定内容包括：原CSV及哈希、候选编号排序、21列输入顺序、每折训练/测试编号、标准化统计、ReLU与Dropout0.2、AdamW学习率/衰减、批量128、梯度裁剪5、余弦T_max600与最低学习率比例0.01、每折训练轮数、Xavier初始化、Python/NumPy/Torch种子、独立批次生成器种子、确定性算法、TF32关闭、CPU线程4与精度。CPU与CUDA的随机排列和Dropout实现不同；操作系统、显卡、PyTorch/CUDA/cuDNN/驱动变化也可能导致重新训练数值变化。不能承诺未知老师电脑逐位相同；即便环境接近也必须实际核验。

重训程序会分别报告参数state_dict是否逐位相等、预测最大绝对差异及MAE，不把“种子相同”当成“参数相同”。不同.pt文件的序列化字节不用于判断重训参数是否相等，使用逐张量比较。原.pt的SHA256只用于检查保存文件未被修改。

## 本包包含什么

- models/fold01…fold10：原.pt、各折参考预测、原训练历史、选择记录、标准化参数。
- recipes.json：每折完整参数、轮数、所有有效随机种子、训练/测试行号及候选编号。
- train.csv：原始2048条数据，candidate_id只排序，不输入网络。
- model.py、training.py、helpers.py、frozen_data.py：可独立运行的固定代码，不需要原项目其他目录。
- provenance/：原始源码与实验记录快照，仅用于追溯，不参与动态导入。
- manifest.json：全包文件SHA256；environment.json为参考环境；reference/oof_predictions.csv为原OOF记录。

本次只做复制、封存及CPU推理核验，没有重新训练。现有十折结果来自已参与项目研发的历史数据，属于交叉验证诊断，不宣称全新盲测。未消耗老师测试机会。
