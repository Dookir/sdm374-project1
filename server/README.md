# Server environment

工作目录 `/data/zengxiong/wuxinkai/sdm374-project1`，环境 `/data/zengxiong/wuxinkai/.venv`（Python 3.12）。
2026-10-23 起当前流程回到原 baseline，使用 CPU，不启动神经网络或 GPU 检查。

```bash
cd /data/zengxiong/wuxinkai/sdm374-project1
source server/activate.sh
CUDA_VISIBLE_DEVICES= python server/check_environment.py
CUDA_VISIBLE_DEVICES= python -m unittest -v baseline_restart.test_restart
CUDA_VISIBLE_DEVICES= python -u -m baseline_restart.run --out baseline_restart/results/new_run
```

系统 Python、GPU 驱动与原环境安装包保持原样。历史完整依赖记录保留在 requirements-linux.lock.txt；当前流程使用 NumPy、pandas、SciPy 和 scikit-learn。
GitHub 仓库专用 SSH key 保存在仓库外，不存储登录密码或账户令牌。
如下载阻塞，检查代理，但不打印凭据、不修改全局 Git 重写、不放宽 SSH/TLS 校验。当前重跑无需安装任何包。
