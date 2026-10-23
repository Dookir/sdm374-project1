# Working instructions

- Human correction on 2026-10-23 takes precedence: restart from original baseline. Generate model parameters -> original optimize generates weights -> original predict scores. Rank ONLY by unchanged Lab6/baseline.py predict output under unchanged Lab6/model.npz. Do not replace this criterion with MAE, R2, regret, neural predictions, or an invented composite.
- Active implementation is baseline_restart/. See baseline_restart/PLAN.md. Start with course weeks 1-2: linear/polynomial regression, normalization, ordinary least squares, L1/L2 regularization and gradient descent.
- Primary checkout is /data/zengxiong/wuxinkai/sdm374-project1 on the A100 server. All project edits and computation occur there. Activate source server/activate.sh. Local macOS files are reference/backup only.
- Teacher Lab6/ files and original scoring model are immutable. Save every generated model and 3x7 weight JSON separately. Never pass a generated model to the fixed scoring CLI as --model.
- Preserve stdout/argv for each native train/optimize/predict call. If no actual fixed-script improvement is found, retain original weights and state so. Never label numerical roundoff as improvement.
- Old experiments are archived outside the active checkout at ../archive-before-baseline-restart-20261023 and in Git tag archive-before-baseline-restart-20261023. Do not resume those pipelines unless the user requests them.
- Current methods need only CPU. If a GPU is used later, use at most one initially idle card, checking occupancy first; never affect other users' jobs.
- Keep predictions and real simulator outputs named accurately. Current predict field is predicted_ramping; real black-box software is not included. No external submissions unless authorized.
- No passwords, credentials, private keys, environments, or runtime caches in Git. Repository SSH key remains outside checkout.
