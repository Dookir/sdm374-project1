# Working instructions

- Human correction on 2026-10-23 takes precedence: restart from original baseline. Generate model parameters -> gradient search generates legal weights -> original predict scores; neural search uses input gradients because original optimize only accepts quadratics. Rank ONLY by unchanged Lab6/baseline.py predict output under unchanged Lab6/model.npz. Do not replace this criterion with MAE, R2, regret, neural predictions, or an invented composite.
- Latest human instruction on 2026-10-23 explicitly switches to neural networks, referencing weeks 5-6. Active pipeline is neural_baseline/; read neural_baseline/PLAN.md. Train networks -> projected-gradient input search -> original predict scores; rank only by that fixed scorer. baseline_restart/ remains the regression reference. Never restore old MAE-based model ranking.
- Primary checkout is /data/zengxiong/wuxinkai/sdm374-project1 on the A100 server. All project edits and computation occur there. Activate source server/activate.sh. Local macOS files are reference/backup only.
- Teacher Lab6/ files and original scoring model are immutable. Save every generated model and 3x7 weight JSON separately. Never pass a generated model to the fixed scoring CLI as --model.
- Preserve stdout/argv for each native train/optimize/predict call. If no actual fixed-script improvement is found, retain original weights and state so. Never label numerical roundoff as improvement.
- Old experiments are archived outside the active checkout at ../archive-before-baseline-restart-20261023 and in Git tag archive-before-baseline-restart-20261023. Do not resume those pipelines unless the user requests them.
- Neural training uses at most one initially idle GPU, selected by UUID after checking occupancy. Never affect other users' jobs. Regression reference needs only CPU.
- Keep predictions and real simulator outputs named accurately. Current predict field is predicted_ramping; real black-box software is not included. No external submissions unless authorized.
- No passwords, credentials, private keys, environments, or runtime caches in Git. Repository SSH key remains outside checkout.

- Current neural run completed: neural_baseline/results/first_run_20261023. Twelve trials and 49 fixed-teacher scores. Lowest neural candidate is mlp128_relu_2027_e100 at 88.140128, baseline 75.895701 remains incumbent. Do not describe this neural round as beating baseline. Best neural artifacts are separate from recommended submission weights.

- Latest request expands neural parameters. Active expanded entry: neural_baseline.expanded_run; see EXPANDED_PLAN.md. Twelve configurations x three seeds, 1200 epochs, 4 checkpoints, fixed teacher scoring only; preserve both original baseline and previous best neural score as incumbents. One initially idle GPU only.
