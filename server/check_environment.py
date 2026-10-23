"""CPU environment check for the original baseline workflow."""
import json,sys
from pathlib import Path
import numpy,pandas,scipy,sklearn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from Lab6 import baseline
if __name__=="__main__":
    _,X,y=baseline.load_table(ROOT/"Lab6/dataset_same8/train.csv")
    score=baseline.predict(json.loads((ROOT/"Lab6/weights.json").read_text()))
    assert X.shape==(2048,21) and abs(score-75.89570106874649)<1e-8
    print(json.dumps(dict(python=sys.version,rows=len(y),original_script_prediction=score,
        numpy=numpy.__version__,pandas=pandas.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,gpu_used=False),indent=2))
