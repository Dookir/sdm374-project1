# 快速开始：无需训练复现结果

[返回首页](../README.md) · [方法说明](WORKFLOW.md) · [结果表](RESULTS.md)

## 1. 获取项目并选择Python

```bash
git clone https://github.com/Dookir/sdm374-project1.git
cd sdm374-project1
python --version
```

推荐Python **3.12**，服务器实际版本为3.12.15。以下 `python` 必须指向你准备使用的解释器；若本机命令名是 `python3`，统一替换即可。使用已有合适环境即可，不要求新建环境。不要直接用Python3.14照装本项目固定的PyTorch2.5.1。

已经能登录原服务器的队友，可以跳过安装：

```bash
cd /data/zengxiong/wuxinkai/sdm374-project1
source server/activate.sh
```

该脚本依赖仓库父目录已有的 `.venv`，仅用于原服务器。刚克隆到自己电脑时，不要运行它。

## 2. 在自己的电脑安装推理依赖

若已有兼容的PyTorch2.5.1，无需重新安装。否则按系统二选一：

Linux或Windows，CPU版：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

macOS：

```bash
python -m pip install torch==2.5.1
```

然后安装其他依赖：

```bash
python -m pip install numpy==2.5.2 pandas==3.0.6 scipy==1.18.1 scikit-learn==1.9.1
python -m pip install -r tabular_next/requirements.txt
```

以上数值库版本对应当前服务器记录。安装源或平台无法提供某版本时，先核对Python3.12与平台支持，不要在原共享服务器中任意升级。PyTorch来源与平台命令见[官方历史版本页面](https://pytorch.org/get-started/previous-versions/#v251)。本项目不需要torchvision、torchaudio。

当前 `server/requirements-linux.txt` 是历史依赖入口，引用的旧 `offline/requirements.txt` 已不在主分支；**本次不要使用该入口安装**。`server/requirements-linux.lock.txt`也只是历史完整记录，不是本次最小安装清单。

## 3. 复现40.4908

所有命令从仓库根目录运行。下面的预测入口默认将模型加载到CPU，Windows也可直接运行。

```bash
python -m neural_baseline.retained --name relu_64_32 --weights weight_search/results/run_20261023/weights/relu_64_32.json
```

输出包含：

```json
{"name": "relu_64_32", "predicted_ramping": 40.49078369140625}
```

实际JSON还包括 `verified`、`models` 列表和权重路径。浮点数末位可能随平台有微小差异，先核对四位小数 **40.4908**。

两份文件的配对关系：

- 模型：`neural_baseline/retained/relu_64_32/model.pt`。
- 新权重：`weight_search/results/run_20261023/weights/relu_64_32.json`。

不要将新权重与其他宽度的网络随意配对后仍称作这个结果。

## 4. 查看全部模型和加强寻优结果

查看5种MLP的原权重预测及文件校验：

```bash
python -m neural_baseline.retained --verify
```

这个命令默认使用各自保留目录中的**原权重**，所以第一个结果是 **40.7017**，不是40.4908。

查看11个模型加强寻优前后结果，并在CPU重新验证最新权重：

```bash
python -m weight_search.check --audit
```

正常结束时末尾应包含 `"passed": true`、`"models_verified": 11`。

## 5. 预测一份TabM候选

```bash
python -m tabular_next.predict --checkpoint tabular_next/results/run_20261023/best/tabm/model.pt --weights tabular_next/results/run_20261023/best/tabm/weights.json
```

应输出约 **91.7101**。此处加载的是TabM自身模型，输出是它各分支的平均预测。

## 常见问题

|现象|检查方法|
|---|---|
|`No module named neural_baseline`|确认当前目录是仓库根目录，使用 `python -m ...`；不要切进模块目录运行|
|`No module named tabm`|安装 `tabular_next/requirements.txt`，并确认pip与运行命令使用同一个Python|
|无法安装torch2.5.1|检查Python版本与平台，优先使用3.12；安装命令见上方官方来源|
|输出40.7017而非40.4908|确认传入了 `weight_search/.../relu_64_32.json`，而非保留目录原权重|
|`Weights must...` 或 `Expected legal...`|JSON必须为3×7有限非负数，每行和为1，不能把model.pt当权重|
|审计提示哈希变化|确认没有改过原始数据、模型、原候选或受保护脚本；保留修改并核对Git差异，不要通过改哈希跳过检查|
|没有CUDA/GPU|本页所有预测和审计命令均使用CPU，不需要GPU|
|`Output exists` 或 `Choose fresh output directory`|这是训练/寻优入口，不是预测入口；重跑实验需要新输出目录|

以上预测和审计命令已在现有Linux服务器环境复核；macOS/Windows的全新依赖安装未逐平台实测，安装失败时按上表排查。

下一步想看原理，读[方法说明](WORKFLOW.md)；想做新实验，读[服务器说明](../server/README.md)。
