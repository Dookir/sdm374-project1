from cv_experiment.common import ROOT,SEED,CONFIGS,SEARCH,sha,dump,read,data,make_splits,validate_splits,validate_files,summarize
from sklearn.model_selection import KFold
import numpy as np
TRAIN=dict(epochs=600,batch_size=128,gradient_clip=5.,loss='normalized MSE',patience=30,min_delta=0.,eta_min_ratio=.01,trials=30,inner_folds=3,startup_trials=10,learning_rate_range=[.0001,.003],weight_decay_range=[.0001,.03],dropout_choices=[0.,.05,.1,.2],selection='minimum sample-weighted inner best-validation MAE',refit='full outer training; median selected inner epoch; cosine T_max remains 600')
def inner_splits(outer):
 indices=np.asarray(outer['train']); seed=SEED+300000+outer['fold']-1
 return [dict(inner_fold=k+1,train=indices[a].tolist(),validation=indices[b].tolist(),init_seed=outer['init_seed']+10000+k,shuffle_seed=outer['shuffle_seed']+10000+k) for k,(a,b) in enumerate(KFold(3,shuffle=True,random_state=seed).split(indices))]
