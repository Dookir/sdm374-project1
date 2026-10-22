# A100 server environment

Primary workspace: `/data/zengxiong/wuxinkai`.
Project checkout: `/data/zengxiong/wuxinkai/sdm374-project1`.
Python environment: `/data/zengxiong/wuxinkai/.venv` (Python 3.12).
Managed Python and uv are kept under the workspace; system Python is unchanged.

## Daily use

```bash
cd /data/zengxiong/wuxinkai/sdm374-project1
source server/activate.sh
python --version
nvidia-smi
python server/check_environment.py --catboost-gpu
cd offline
python -m unittest -v test_offline.py test_solution.py
```

Default `CUDA_VISIBLE_DEVICES=0` uses one GPU. Check current use before long jobs.
To choose another GPU, set `CUDA_VISIBLE_DEVICES` before sourcing the activation script.
No substantial training is started by environment setup; the smoke check uses only five tiny synthetic optimizer steps.

## Rebuild

The current NVIDIA 535 driver is paired with the official PyTorch 2.5.1 CUDA 12.1 wheel.
The driver and system CUDA installation are unchanged.

```bash
source server/activate.sh
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-cache \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  "torch @ https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp312-cp312-linux_x86_64.whl"
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-cache \
  --index-url https://mirrors.aliyun.com/pypi/simple -r server/requirements-linux.txt
```

`server/requirements-linux.lock.txt` records the resolved Linux environment after installation.
CPU package pins match the previous experiment, allowing the existing scikit-learn model to load.
The macOS-only runtime metadata remains historical evidence; it is not overwritten.
Runtime validation results and installer logs are outside the repository under the workspace.

GitHub access uses a repository-scoped SSH deploy key outside the checkout, with verified GitHub host keys.
No GitHub account token or server password is stored in this repository.
All further code edits and model computations should take place in this remote checkout.

If a download stalls, inspect proxy settings without printing credentials. The server currently has no HTTP/HTTPS/ALL_PROXY environment override. Use the Aliyun index for ordinary dependencies and the official direct PyTorch wheel URL above; the official all-package index previously stalled on small CPU dependencies. Do not modify the machine-wide GitHub URL rewrite or weaken TLS/SSH host verification.
