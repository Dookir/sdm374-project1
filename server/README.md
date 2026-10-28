# 服务器环境与单卡运行

[首页](../README.md) · [新成员快速开始](../docs/QUICKSTART.md)

当前工作目录为 `/data/zengxiong/wuxinkai/sdm374-project1`；已配置Python环境位于父目录 `.venv`。当前支持MLP预测、残差MLP/TabM训练与固定模型寻优；之前“仅CPU回归、不启动神经网络”的说明已经过时。

## 直接使用已有环境

使用负责人提供的服务器账号登录后：

```bash
cd /data/zengxiong/wuxinkai/sdm374-project1
source server/activate.sh
python -m neural_baseline.retained --name relu_64_32 --weights weight_search/results/run_20261023/weights/relu_64_32.json
```

预测入口内部加载到CPU，不需要选GPU。`activate.sh`是Bash脚本，会激活仓库父目录的已有环境并切到项目根目录；本机新克隆目录通常不具备该环境，不要直接照搬。若当前shell不兼容，先进入Bash再source。

## 当前环境记录

|组件|版本|
|---|---|
|Python|3.12.15|
|PyTorch|2.5.1+cu121|
|NumPy|2.5.2|
|pandas|3.0.6|
|SciPy|1.18.1|
|scikit-learn|1.9.1|
|tabm|0.0.3|
|rtdl-num-embeddings|0.0.12|

版本来自当前已运行实验的环境，非要求队友使用最新版本。普通CPU推理的安装步骤见[快速开始](../docs/QUICKSTART.md)。

`requirements-linux.txt` 和 `requirements-linux.lock.txt` 保留为历史记录；前者引用已移出主分支的 `../offline/requirements.txt`，不能作为当前直接安装入口。当前最小安装清单以快速开始为准；TabM新增依赖在 `../tabular_next/requirements.txt`。这次仅修正文档，没有重装环境或改动依赖文件。

`python server/check_environment.py`只检查原baseline所需环境、2048行数据和75.8957预测，不是全部神经网络依赖检查。完整结果检查使用各模块的入口。

## GPU任务：只用一张空闲卡

先检查占用，不要终止或影响其他人的进程：

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
```

选择显存空闲、利用率为0的卡，将下面 `GPU_UUID` 替换为查询到的完整 `GPU-...` UUID。程序启动时还会核对空闲状态，只暴露这一张卡；不允许为同一任务叠加多卡。显存上限为该卡的10%。

下面两种任务**择一启动，完成后再启动另一个**；已有输出目录不能覆盖：

```bash
# 新训练：保存输出到新目录
CUDA_VISIBLE_DEVICES=GPU_UUID python -u -m tabular_next.run --out tabular_next/results/team_run_20261028
# 或新寻优：固定现有11个模型
CUDA_VISIBLE_DEVICES=GPU_UUID python -u -m weight_search.run --out weight_search/results/team_run_20261028
```

原服务器已有tmux；需要断开终端继续运行，可先 `tmux new -s sdm374-team`，在里面激活环境并执行其中一个命令，再按Ctrl+B、D脱离。用 `tmux attach -t sdm374-team`恢复。组员之间先确认会话名和GPU占用，避免重复启动。

长任务按项目约定使用低频自动化检查，正常运行时静默，只在完成/失败时通知；检查 `status.json` / `complete.json` / `failure.json` 即可，不需要持续跟踪。GitHub仓库不包含聊天应用里的自动化，也不包含服务器登录凭据；队友不会因克隆仓库自动获得监控服务或服务器权限。

## CPU复核

```bash
python -m neural_baseline.retained --verify
python -m tabular_next.check --audit
python -m weight_search.check --audit
```

这些默认核对原实验结果；新实验的检查路径和限制见各模块README。下载或连接不通时检查网络/代理，不打印凭据、不修改全局Git配置或关闭SSH/TLS校验。仓库专用SSH密钥放在仓库外。
