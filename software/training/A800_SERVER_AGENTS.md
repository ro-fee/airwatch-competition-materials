# AirWatch A800 server execution rules

Read START_HERE.md and MODEL_TRAINING_PLAN.md first. User authorized CPU preparation and the prescribed experiment, with GPU execution only after explicit operator/scheduler allocation and actual remaining minutes.

- Use `/root/lyh_workspace/envs/airwatch-a800/bin/python`. Do not install into base or other environments; do not change GPU drivers, MIG, power settings or other users' jobs.
- Verify `SHA256SUMS`, `bundle-files.json` and `bundle-status.json`. Immutable payload files must remain unchanged. All generated state goes under `outputs/`.
- Run `bash scripts/bootstrap_server.sh` for CPU preparation. Missing real data, missing required tests and skipped required tests are failures.
- Run GPU work only through `scripts/run_allocated_gpu.sh --phase train|finalize` with real allocation details. Exit 75 means wait/resume after a new explicit allocation; never automatically seize an idle GPU.
- Preserve the 12-run fixed protocol, source/data identity, full recovery checkpoints, and validation-only selection/calibration. No test/unknown/guard/VTI training or selection.
- VTI is audit-complete but incompatible with the current frozen input protocol. Preserve the conditional external-evaluation limitation; do not invent conversion or metrics.
- Do not manually edit metrics or status to force completion. Do not delete failed evidence. Report protocol changes before attempting them.
- The bundle is independently CPU-checked on Windows. Linux installation, CUDA behavior and final target desktop performance remain to be measured.
- Before any gcc/g++ compilation, run `conda deactivate` in the same shell process. No C/C++ build is expected for the pinned binary wheel route.
- Do not depend on Codex plugins or local workstation paths. Legacy absolute paths inside retained V4 evidence are historical provenance, not server input locations.
- Return the complete release contract, evidence, raw benchmark samples, environment and failure records using `scripts/export_results.sh`.
