"""Verify the original program was used and the original scorer stayed fixed."""
import hashlib,json,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"baseline_restart/results/first_run_20261023"
class RestartTests(unittest.TestCase):
    def test_scores_are_native_cli_stdout_without_alternate_model(self):
        rows=json.loads((OUT/"comparison.json").read_text())
        self.assertEqual(len(rows),14)
        for row in rows:
            command=json.loads((OUT/"commands"/(row["name"]+"_predict.json")).read_text())
            self.assertNotIn("--model",command["argv"])
            self.assertEqual(Path(command["argv"][1]),ROOT/"Lab6/baseline.py")
            self.assertEqual(command["argv"][2],"predict")
            self.assertEqual(json.loads(command["stdout"])["predicted_ramping"],row["teacher_script_score"])
    def test_native_optimizer_stdout_matches_weights(self):
        rows=json.loads((OUT/"comparison.json").read_text())
        for row in rows:
            if row["generating_model"] is None: continue
            command=json.loads((OUT/"commands"/(row["name"]+"_optimize.json")).read_text())
            argv=command["argv"]; self.assertEqual(argv[2],"optimize")
            self.assertEqual(Path(argv[argv.index("--model")+1]),ROOT/row["generating_model"])
            weights=np.array(json.loads((ROOT/row["weights"]).read_text()))
            np.testing.assert_array_equal(np.array(json.loads(command["stdout"])),weights)
    def test_original_files_and_selected_weights(self):
        done=json.loads((OUT/"complete.json").read_text())
        for name,checksum in done["protected_files"].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),checksum)
        rows=json.loads((OUT/"comparison.json").read_text())
        for row in rows:
            w=np.array(json.loads((ROOT/row["weights"]).read_text()))
            self.assertEqual(w.shape,(3,7)); self.assertTrue(np.isfinite(w).all() and (w>=0).all())
            np.testing.assert_allclose(w.sum(1),1,rtol=0,atol=1e-12)
        self.assertLessEqual(done["selected_score"],min(r["teacher_script_score"] for r in rows)+1e-8)
        selected=next(r for r in rows if r["name"]==done["selected"])
        self.assertEqual(json.loads((ROOT/selected["weights"]).read_text()),json.loads((OUT/"weights_recommended.json").read_text()))
if __name__=="__main__": unittest.main()
