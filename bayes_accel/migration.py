import argparse,shutil,subprocess
from pathlib import Path
from importlib.metadata import version
from bayes_experiment.common import ROOT,SEED,CONFIGS,TRAIN,SEARCH,sha,dump,read,data,make_splits,validate_files

def validate_inherited(out,manifest):
 if 'inherited_files' not in manifest:raise ValueError('Missing migration provenance')
 legacy=read(out/'provenance/serial_manifest.json')
 for key in ['seed','architectures','training','search','data_sha256','candidate_ids','n_rows','versions']:assert manifest[key]==legacy[key],key
 for path,digest in legacy['source_sha256'].items():assert sha(ROOT/path)==digest,path
 for path,digest in manifest['inherited_files'].items():assert sha(out/path)==digest,path
 assert sha(out/'migration.json')==manifest['migration_sha256']

def migrate(source,destination,benchmark):
 from bayes_accel.run import dependency_hashes
 source=Path(source);out=(ROOT/destination).resolve()
 assert out.is_relative_to(ROOT) and not out.exists()
 legacy=read(source/'manifest.json')
 for path,digest in (legacy['source_sha256']|legacy['protected_files']).items():assert sha(ROOT/path)==digest,path
 inherited={};counts={}
 for pattern in ['models/*/fold*/training_complete.json','models/*/fold*/search_complete.json','tuning/*/fold*/complete.json','tuning/*/fold*/trial*/complete.json','tuning/*/fold*/trial*/inner*/complete.json']:
  markers=list(source.glob(pattern));counts[pattern]=len(markers)
  for marker in markers:
   info=read(marker);validate_files(marker.parent,info);inherited[str(marker.relative_to(source))]=sha(marker)
   for name,digest in info['files'].items():inherited[str((marker.parent/name).relative_to(source))]=digest
 shutil.copytree(source,out,ignore=shutil.ignore_patterns('run.lock','*.tmp'))
 provenance=out/'provenance';provenance.mkdir();shutil.copy2(source/'manifest.json',provenance/'serial_manifest.json');inherited['provenance/serial_manifest.json']=sha(provenance/'serial_manifest.json')
 shutil.copy2(benchmark,out/'benchmark.json');inherited['benchmark.json']=sha(out/'benchmark.json')
 dump(out/'migration.json',dict(source_snapshot=str(source),source_commit=legacy['base_commit'],completed_counts=counts,unfinished_legacy_inner='restart from original seeds; legacy process had no epoch checkpoint',execution='same GPU UUID; three independent processes with private MPS; no hyperparameter or split changes'))
 ids,X,y=data();assert read(out/'splits.json')==make_splits(len(y))
 tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
 protected={p:sha(ROOT/p) for p in tracked if p and not p.startswith('bayes_accel/') and not p.endswith('.md') and (ROOT/p).is_file()}
 manifest=dict(version=1,seed=SEED,architectures=CONFIGS,training=TRAIN,search=SEARCH,data_sha256=sha(ROOT/'Lab6/dataset_same8/train.csv'),candidate_ids=ids,n_rows=len(y),source_sha256=dependency_hashes(),versions={n:version(n) for n in ['torch','numpy','pandas','scikit-learn','scipy','optuna']},splits_sha256=sha(out/'splits.json'),protected_files=protected,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),inherited_files=inherited,migration_sha256=sha(out/'migration.json'),execution=dict(inner_parallelism=3,gpu_uuid='GPU-e9b4fef6-be25-1e26-6b9a-753c8d74ef66',precision='float32 models; original float64 validation MAE; TF32 disabled'))
 dump(out/'manifest.json',manifest);dump(out/'status.json',dict(state='ready_to_resume',completed_training=counts['models/*/fold*/training_complete.json'],completed_search=counts['models/*/fold*/search_complete.json'],inherited_files=len(inherited)))
 validate_inherited(out,manifest);print('MIGRATED',counts,'INHERITED_FILES',len(inherited),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--benchmark',required=True);a=p.parse_args();migrate(a.source,a.out,a.benchmark)
if __name__=='__main__':main()
