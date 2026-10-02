# A800 V2 preparation coordination

User approved implementation on 2026-09-29. Detailed specification: `docs/superpowers/plans/2026-09-29-a800-server-training-plan.md`.

## Ownership and shared interfaces

- Parent: data preparation, provenance, Linux environment/bootstrap, bundle builder/integration, source documents.
- Trainer agent: `training/a800_runner.py`, `training/a800_common.py`, `scripts/run_allocated_gpu.sh`, trainer/resource tests, matrix configs.
- Model agent: `airwatch/models/uav_multiscale_tcn.py`, `training/a800_objective.py`, model/objective tests.
- Evaluation agent: `training/a800_evaluation.py`, `training/a800_delivery.py`, `airwatch/inference/uav_a800_contract.py`, evaluation/delivery tests.
- Do not edit frozen old model/preprocessing/inference modules or unrelated existing files.

Data API: `airwatch.data.ku_leuven_training_v2.A800Dataset(root, split='train', verify_hashes=True)` reads the same three array/CSV artifacts and metadata layout as `KULeuvenMaterializedDataset`, with artifact type `ku_leuven_a800_v2`; returns `(tensor, label, index)` and provides `samples`, `sample_metadata`, `recording_ids`, `label_map`, `data_identity`, `close`. Only train and validation permitted. V1 control uses existing KULeuvenMaterializedDataset. Metadata and all artifacts must be hash bound. Data paths are bundle relative: `data/prepared/known-iq-v1`, `data/prepared/dense-train-v2`. Validation/prototype base is always V1. Main split assignments are under `manifests/known-split-v1/known-split-assignments.csv`.

Model API: `MultiScaleRFTCN(num_classes=3)` yields logits and implements `forward_features`. Factory `build_a800_model(name, num_classes=3)` accepts `tcn` / `multiscale_tcn`. Objective API `paired_objective(model, view1, view2, labels, js_weight=0.0)` returns loss and a dictionary of detached CE/JS diagnostics; concatenate views for one forward. Model agent supplies stateless `make_paired_views` with exact signature announced to trainer.

Trainer owns JSON configuration schema and checkpoint schema; announce exact schema to evaluator and parent before implementation. Must support small synthetic unit fixtures without relaxing formal-run gates. Checkpoints must bind data/config/code and complete recovery state. Evaluation imports trainer factory only if needed without creating import cycles. No model import from Qt.

All command line entry points use `python -m training.<module>` and support `--help`. JSON paths portable relative to bundle root. Server GPU execution only through explicit allocation gate; no local formal training.

## Progress ledger

- [x] T1 VTI password/format audit (metadata only, no predictions)
- [x] T2 immutable dense training + known/unknown manifests
- [x] T3 Linux candidate dependencies and CPU bootstrap
- [x] T4 resumable trainer and allocation gate
- [x] T5 multi-scale model and paired objective
- [x] T6 frozen evaluation and threshold protocol
- [x] T7 runtime export and GPU benchmark
- T8 independent upload bundle: initial independent checks passed; final readiness is owned exclusively by bundle-status.json and archive SHA-256, not a manually checked box.

## Rulings

- No Git checkout exists; use immutable source hashes and snapshots, no invented commits.
- E drive has about 15.6 GB free. Bulk staging/output uses `F:/AirWatch_A800_Preparation`; source remains in workspace. This avoids filling the project drive.
- Local Windows checks cannot certify Linux CUDA execution. The bundle must distinguish local preparation evidence from mandatory server bootstrap/allocation checks.

## 2026-10-02 continuation

- Server executor changed by user from OpenCode to Claude Code; CLI and protocol are unchanged.
- VTI outer integrity/password verified, all PP/DNN metadata and one RAW header per band audited; current frozen representation incompatible. No VTI predictions.
- Dense data: 29440 training windows, 115 members; original 3680 windows bitwise preserved; validation unchanged. Unknown metadata: 204 members.
- Linux closure resolved and independently checked: 46 packages. This is metadata/wheel verification, not Linux runtime verification.
- Final publication now requires nonzero tests per mandatory module, an immutable input-index binding and no newer failed check.

- Initial independent bundle run: 82 tests passed, zero skipped; true V1/V2 arrays and sealed metadata verified. Final source changes must be re-indexed and rechecked before publication.
- Historical regression: 207 core tests + 30 waterfall tests + model-weight smoke + compileall passed under dedicated Python with CUDA hidden.

- Final GPU preflight also requires exact serialized 4-step continuous vs 2+2 recovered engineering equivalence for both architectures. Server CUDA execution pending; same control path tested on CPU.
- Freeze binds all environment candidate files and Linux installed-environment/pip-freeze snapshots.
