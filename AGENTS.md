# Working instructions

- Latest human decision: each model uses its OWN predicted ramping as the main offline score, lower is better. Compare and select using this score. Do not substitute the fixed baseline predictor, MAE/R2 or composite scores. Cross-model predictions are optional diagnostics. Do not use unavailable simulation to evade analysis; diagnose score behavior concretely.
- Label score fields predicted_ramping. Preserve both teacher test opportunities for final candidates; no submissions without human authorization.
- Current cleanup retains exactly five distinct network architectures, one selected checkpoint and candidate per architecture, ranked by lowest own prediction among existing candidates. See neural_baseline/retained/manifest.json. Keep models paired with their weights. Deduplicate seeds/epochs/regularization variants by hidden widths and activation.
- Current entry: source server/activate.sh, then CUDA_VISIBLE_DEVICES= python -m neural_baseline.retained --verify. Default predicts all five on CPU; --name and --weights are supported.
- All project edits and computation occur in /data/zengxiong/wuxinkai/sdm374-project1 on the server. Future training may use at most one initially idle GPU. Do not start training or restore archives without an applicable request.
- Lab6 teacher files, data and original baseline are immutable. baseline_restart/ is a historical regression reference, not the current neural selection criterion.
- Old neural outputs/code are archived outside checkout; location is in the retained manifest. Earlier archives are in ../archive-before-baseline-restart-20261023. Archived policies are obsolete.
- No credentials, private keys, environments or runtime caches in Git; repository SSH key stays outside checkout.
