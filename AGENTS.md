# Project working instructions

- Primary checkout: `/data/zengxiong/wuxinkai/sdm374-project1` on the user's A100 server.
- Perform project code edits, training, model comparison, and computational checks on that server. The macOS checkout is a backup, not the execution environment.
- Activate with `source server/activate.sh` from this checkout. Python lives at `/data/zengxiong/wuxinkai/.venv/bin/python`.
- Default GPU is device 0. Check `nvidia-smi` before substantial experiments and respect other users' running jobs; do not assume all GPUs are available.
- Keep teacher-provided `Lab6/` files unchanged. Preserve existing experiment provenance and use new output directories for revised experiments.
- Model prediction, validation error, and real simulator score are different quantities. Do not claim that minimizing a surrogate proves improved real performance.
- Do not store SSH passwords, private keys, access tokens, virtual environments or runtime caches in Git. Repository SSH keys are outside the checkout.
- Remote environment setup is complete; the expanded neural-network/nested-validation research remains pending. Environment smoke checks do not constitute model-quality experiments.
