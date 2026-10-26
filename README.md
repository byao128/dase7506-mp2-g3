# Answer-Group Reranking for LLM Solution Selection

DASE7506 MP2 · Student **3036779750** · GitHub **byao128**

The submitted method is the original frozen G3. Its complete official test
result is **381/467 = 81.5845824411135%**, reproduced without retraining.
The report is [report/report.pdf](report/report.pdf), eight pages, including
AI/reuse disclosure and the evaluation-history limitation.

## 1. Install and obtain data

Use Python 3.12; the observed environment was Python 3.12.14, NumPy 2.5.4 and
CatBoost 1.2.8. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r code\requirements.txt
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:NUMEXPR_NUM_THREADS = '4'
$env:PYTHONDONTWRITEBYTECODE = '1'
```

Obtain the complete official data separately from
<https://github.com/xudongwu-0/DASE7506-MP2>. Keep its contents unchanged.
Point to that repository's `code/data` directory:

```powershell
$DATA_DIR = 'C:\path\to\DASE7506-MP2\code\data'
```

No dataset, environment, external pretrained checkpoint or API key is shipped
or needed by this selector. Retained data license/attribution files are under
`code/data/`. Fresh installation in an empty environment and timing on the
course reference Xeon have not been independently tested here.

## 2. Verify the checkpoint and reproduce the fixed result

The original freeze binds eight inference sources and six assets:

```powershell
python -c "import sys; from pathlib import Path; sys.path.insert(0, 'code'); from evaluate_group_tree_bundle import verify_freeze; verify_freeze(Path('artifacts/group-tree-v1/bundle'), Path('records/group-tree-v1/freeze.json')); print('G3 freeze: OK')"

python code\evaluate_fixed_g3.py `
  --data-dir $DATA_DIR `
  --checkpoint artifacts\group-tree-v1\bundle `
  --freeze records\group-tree-v1\freeze.json `
  --split test `
  --output verification\original-g3-test
```

The output directory must be new. The wrapper runs unchanged official
prediction/scoring utilities as two fresh, sequential processes, with four
threads. It produces `predictions.json`, unmodified `raw-score.json`, separate
`resources.json`, and process logs. Submitted aggregates are under
`records/reproduction/`; per-question predictions and logs are generated
afresh, not distributed as cached outputs. This reproduces a fixed checkpoint
and must not be used to tune or select alternatives using test results.

## 3. Rebuild the fixed model using train only

```powershell
python code\train_fixed_g3.py `
  --data-dir $DATA_DIR `
  --output reproduced\group-tree-v1\bundle `
  --record reproduced\group-tree-v1\train-record.json

python code\evaluate_group_tree_bundle.py `
  --checkpoint reproduced\group-tree-v1\bundle `
  --freeze reproduced\group-tree-v1\freeze.json `
  --create-freeze
```

All output paths must be new. The fixed recipe is depth 4, leaf L2 10,
600 trees and alpha 0.5. B0/tree seeds are 2028; three-fold B0 cross-fitting
uses split seed 6100. No validation/test reference enters fitting. The
provided checkpoint is never overwritten. Check a rebuilt model on validation
using the evaluation command with its new paths, `--split validation`, and a
new output directory. Supplied weights are authoritative for the reported
test score.

## 4. Reproduce the report's development comparisons

Historical training launchers use local `code/data`. For these train-only
research runs, first copy the official data locally:

```powershell
Copy-Item -Path (Join-Path $DATA_DIR '*') -Destination code\data -Recurse -Force
Push-Location code
python run_group_tree.py --fold-seed 2026 --output ..\reproduced\g3-oof
python run_group_tree.py --fold-seed 2027 --output ..\reproduced\g3-oof
python run_group_tree_bagging.py --stage primary --output ..\results\group-tree-bagging-v1
Pop-Location
```

Run in a fresh checkout/output namespace. Do not publish copied data,
environments or generated outputs. Both G3 outer runs use five folds and
train-only inner selection; seed 2027 also offsets training seeds by 10,000.
BAG3 uses primary split seed 6061 and the fixed parameters/seed bases in
`group_tree_bagging_cv.py`. Its comparison is separate from original G3 nested
selection. BAG3 weights are not the submission checkpoint.

The packaged BAG3 launcher uses the packaged freeze location and hashes its
included research sources instead of depending on private tests or planning
documents. Its learning algorithm and registered settings are unchanged.

Fixed validation component/confidence analysis:

```powershell
python code\validation_analysis.py `
  --source-dir code `
  --data-dir $DATA_DIR `
  --checkpoint artifacts\group-tree-v1\bundle `
  --freeze records\group-tree-v1\freeze.json `
  --output verification\validation-analysis.json `
  --plot-data verification\validation-plot-data.json
```

This retrospective analysis uses five fixed components and ten confidence
bins, fits no model/calibrator and changes no settings. Compare its aggregates
with `records/validation-analysis.json`.

## 5. Results, costs and interpretation

| Record | Result / purpose |
|---|---|
| `records/group-tree-v1/oof-seed-2026.json` | G3 1716/2114 hard vs B0 group 1702; wall 1459.668s |
| `records/group-tree-v1/oof-seed-2027.json` | G3 1701/2114 hard vs B0 group 1698; wall 2009.366s |
| `records/group-tree-v1/bundle-build.json` | Final configuration, fit cost 70.368s and asset hashes |
| `records/group-tree-v1/validation-bundle-score.json` | Official frozen validation: 745/798 |
| `records/group-tree-v1/validation-bundle-official.json` | Validation process/resource record |
| `records/validation-analysis.json` | Fixed components, paired changes and confidence metrics |
| `records/research/bagging/primary.json` | Aggregate train-only BAG3: 1707 vs 1704 hard; failed gate |
| `records/research/bagging/test-score.json` | Historical BAG3 official aggregate: 378/467 |
| `records/group-tree-v1/test-score.json` | Original G3 score/resources: 381/467 |
| `records/reproduction/raw-score.json` | Unmodified fresh official G3 score: 381/467 |
| `records/reproduction/resources.json` | Fresh timing, dependencies, threads and hashes |

Local hardware was Intel Core i5-13500H / Windows 11. Fresh prediction took
2.249177s, independent scoring 0.465109s, and their external wrapper interval
2.748836s. Assets total 1,827,288 bytes (1.742638 MiB). Reference-Xeon timing
determines course eligibility; local timing alone does not certify it. Total
historical training/search cost is unknown; logged runs are not an exhaustive
total. BAG3's primary external controller interval was 1154.434s.

G3 was retained on development evidence before BAG3 testing. Subsequently,
both test scores were observed and the present preference for G3 followed
their comparison. A fixed-G3 reproduction again scored 381/467. This history
is disclosed in the report and cannot be described as a single untouched
final holdout. Instructor clarification is recommended; none is claimed.
Technical reproduction does not certify compliance with the rule against
test-based method selection.

## 6. Version, hashes and attribution

`VERSION.json` identifies the code-content version and matching checkpoint.
`records/group-tree-v1/freeze.json` preserves original inference-source and
asset hashes. `SHA256SUMS` lists package file hashes. After publication, pin
code/checkpoint links to the same real Git commit. No public repository or
commit is claimed to have been created here.

Official parser, scorer, data interface and fixed representations are reused
unchanged. GSM8K questions/references retain the MIT license and attribution
in `code/data/`. Course candidates/representations used Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`; no Qwen weights are distributed.
NumPy and CatBoost provide numerical/ranking routines. Report Section 8
states the student/AI division of work.
