import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,shutil,traceback
from pathlib import Path
import numpy as np,torch
from bayes_experiment.common import ROOT,read,dump,sha
from neural_baseline.models import load
from cv_experiment.training import predict_rows
from final_mlp256_search import run as original
PREVIOUS=ROOT/'final_mlp256_search/results/run_20261030'
EXTRA_SEEDS=list(range(20261033,20261040))

def audit(out):
 torch.set_num_threads(4);record=read(out/'complete.json');manifest=read(out/'manifest.json');model=load(original.CHECKPOINT)
 assert sha(original.CHECKPOINT)==manifest['model_sha256']=='ca1f5d9da334706f47b8c5078ef9204d43cb43d44077b15f48a679cf22cb23a0'
 assert sha(Path(__file__))==manifest['entrypoint_sha256']
 for path,digest in manifest['previous_files'].items():assert sha(ROOT/path)==digest,path
 for path,digest in read(out/'new_searches/manifest.json')['protected_files'].items():assert sha(ROOT/path)==digest,path
 assert len(record['searches'])==10 and len({x['seed'] for x in record['searches']})==10
 scores=[]
 for item in record['searches']:
  directory=PREVIOUS if item['seed'] not in EXTRA_SEEDS else out/'new_searches'
  w=np.asarray(read(directory/str(item['seed'])/'weights.json'));assert w.shape==(3,7) and np.isfinite(w).all() and (w>=0).all() and np.allclose(w.sum(1),1,atol=1e-12,rtol=0)
  pred=float(predict_rows(model,w.reshape(1,21))[0]);assert abs(pred-item['predicted_ramping'])<1e-5;scores.append(pred)
 w=np.asarray(read(out/'weights.json'));pred=float(predict_rows(model,w.reshape(1,21))[0]);assert pred==record['predicted_ramping']==min(scores)
 assert pred<=record['previous_best_predicted_ramping'] and record['negative_prediction']==(pred<0)
 report=dict(passed=True,verified_searches=10,new_searches=7,weights_legal=True,cpu_predicted_ramping=pred,previous_best_preserved=True,model_unchanged=True)
 dump(out/'audit.json',report);print(json.dumps(report),flush=True)

def run(args):
 out=(ROOT/args.out).resolve();assert out.is_relative_to(ROOT)
 if out.exists():raise ValueError('Fresh output required')
 out.mkdir(parents=True);previous=read(PREVIOUS/'complete.json');assert read(PREVIOUS/'audit.json')['passed']
 hashes={str(p.relative_to(ROOT)):sha(p) for p in PREVIOUS.rglob('*') if p.is_file() and p.name!='run.lock'}
 dump(out/'manifest.json',dict(model_sha256=sha(original.CHECKPOINT),entrypoint_sha256=sha(Path(__file__)),previous_files=hashes,extra_seeds=EXTRA_SEEDS))
 dump(out/'status.json',dict(state='running',stage='seven_additional_seed_searches',previous_best=previous['predicted_ramping']))
 original.SEEDS=EXTRA_SEEDS
 child=argparse.Namespace(out=str((out/'new_searches').relative_to(ROOT)),owns_output=False)
 original.run(child)
 if not (out/'new_searches/complete.json').exists():
  dump(out/'status.json',dict(state='paused',reason='Selected GPU not idle; no extra search completed'));return
 extra=read(out/'new_searches/complete.json');combined=[*previous['searches'],*extra['searches']];best=min(combined,key=lambda r:r['predicted_ramping'])
 source=PREVIOUS if best['seed'] not in EXTRA_SEEDS else out/'new_searches';shutil.copy2(source/str(best['seed'])/'weights.json',out/'weights.json')
 record=dict(state='complete',model=str(original.CHECKPOINT.relative_to(ROOT)),model_sha256=sha(original.CHECKPOINT),model_retrained=False,previous_best_predicted_ramping=previous['predicted_ramping'],predicted_ramping=best['predicted_ramping'],improvement=previous['predicted_ramping']-best['predicted_ramping'],best_seed=best['seed'],negative_prediction=best['predicted_ramping']<0,new_searches=7,total_searches=10,searches=combined,global_optimum_proven=False)
 dump(out/'complete.json',record);audit(out);dump(out/'status.json',dict(state='complete',predicted_ramping=best['predicted_ramping']))
 (out/'README.md').write_text('# 最终全数据MLP：追加7个寻优种子\n\n保留原3个种子，新加入20261033–20261039，共比较10次搜索。固定同一最终模型，不重新训练、不改参数。保留旧最优，只接受不退步的候选。每种子沿用约6.5万候选和64起点的投影梯度/Adam、权重转移、SLSQP。\n\n最低找到的自身预测ramping：**'+str(best['predicted_ramping'])+'**；原最低：'+str(previous['predicted_ramping'])+'；改善：'+str(record['improvement'])+'。并非全局最优证明。十次搜索均在CPU复核；原模型、旧候选、原十折冻结包不修改，未使用老师测试机会。\n\nweights.json为当前最优，complete.json列出全部10个种子的结果；new_searches/为追加7次的原始记录。其由旧执行器生成的内部README中固定“三种子”字样不适用于追加记录，准确数量以本README和manifest.extra_seeds为准。\n')
 print('COMPLETE '+json.dumps(record),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',default='final_mlp256_search/results/extended_20261030');p.add_argument('--audit',action='store_true');a=p.parse_args()
 if a.audit:audit(ROOT/a.out);return
 try:run(a)
 except Exception:
  out=ROOT/a.out
  if out.exists():dump(out/'failure.json',dict(state='failed',traceback=traceback.format_exc()))
  raise
if __name__=='__main__':main()
