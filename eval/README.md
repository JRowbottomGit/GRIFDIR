# eval/

Two-stage evaluation: the scripts here **compute** (sample, reconstruct, score) and write
`.pt` tensors + metrics into the run directory; `figures/` then **renders** the paper
figures from those. The repo-root `evaluate.py <task>` dispatches to the file below. All
torch-only — cached meshes are loaded from disk, no dolfinx needed.

| File | Purpose |
|------|---------|
| `sampling.py` | Shared engine imported by the others: unconditional Heun sampler (`sample_ve_heun`), sparse-sensor reconstruction samplers Fun-DPS (`run_dps_eval_ve`) / Fun-DAPS (`run_daps_eval_ve`), plus plot helpers and metrics (posterior-mean RMSE, energy score, data consistency). |
| `resolution_invariance.py` | Main battery (**Figs 3–4**): score a trained model at the **same / finer / coarser** resolution, swapping the FEM mesh hierarchy (`swap_mesh_hierarchy`) or grid (`swap_grid`); builds or loads cached hierarchies (`load_hierarchy_pt`). |
| `gaussian_blob_reconstruction.py` | Square domain (**Figs 5–6**): sparse-sensor DPS/DAPS reconstruction of the conductivity field → `<run>/reconstruction/*.pt`. |
| `pinball_reconstruction.py` | Pinball / Navier–Stokes (**Fig 7**): sparse-sensor reconstruction of the flow field. |
| `sensor_sweep.py` | Runs a reconstruction task across several sensor counts and aggregates the per-run summaries into **Table 1** (RMSE / energy-score vs #sensors, Fun-DPS vs Fun-DAPS) — LaTeX table + errorbar plot. |

Each takes `--run_dir <run>` (a training run or a shipped checkpoint — both carry
`config.yaml` + `model_ema_latest.pt`) and writes its outputs alongside it.
