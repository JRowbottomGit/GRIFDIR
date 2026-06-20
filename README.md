# GRIFDIR

**Graph Resolution-Invariant FEM Diffusion Models in Function Spaces over Irregular Domains**

James Rowbottom, Elizabeth L. Baker, Nick Huang, Ben Adcock, Carola-Bibiane Schönlieb, Alexander Denker

[Paper (arXiv:2605.03497)](https://arxiv.org/abs/2605.03497) · [Dataset (Zenodo)](https://doi.org/10.5281/zenodo.20753300)

Score-based diffusion in function spaces via finite-element graph convolutions. The score network is resolution-invariant and handles unstructured meshes and irregular domains natively.

---

## Installation

```bash
pip install -r requirements.txt
```

Data generation additionally requires FEniCSx/dolfinx (conda-only):

```bash
conda env create -f environment.yml
```

FNO and GAOT baselines require external packages (install only what you use):

- **FNO** (`--layer_type fno`) — [Li et al., 2021](https://arxiv.org/abs/2010.08895) — `pip install neuraloperator` ([MIT](https://github.com/neuraloperator/neuraloperator/blob/main/LICENSE))
- **GAOT** (`--layer_type gaot`) — Wen et al., 2025 — clone [camlab-ethz/GAOT](https://github.com/camlab-ethz/GAOT), set `GAOT_REPO`, `pip install rotary-embedding-torch` (see [upstream license](https://github.com/camlab-ethz/GAOT/blob/main/LICENSE))
- **CNN/UNet** (`--layer_type cnn`) and **MP-GNN** (`--layer_type simple_mp`) — included, no extra dependencies

---

## Data

Datasets are hosted on Zenodo ([10.5281/zenodo.20753300](https://doi.org/10.5281/zenodo.20753300)); mesh hierarchies and checkpoints are committed in this repo:

```bash
python data_tools/download_data.py
```

To regenerate from scratch (requires the `grifdir-datagen` conda env):

```bash
# Square Gaussian-blob conductivity fields on a uniform 32x32 mesh.
# -> data/conductivity_nx32_ny32_n10000.pt
python data_tools/generate_training_data.py --seed 0

# Conductivity fields on the irregular domains (circle, L/x/e-shape, plus, holes, ...)
# for the multidomain experiment. --all does every domain; one .pt per domain.
python data_tools/create_mesh_dataset.py --all --seed 0
```

The pinball flow data was generated with [SHRED-ROM](https://github.com/MatteoTomasetto/shred-rom) (Tomasetto & Riva, [MIT](https://github.com/MatteoTomasetto/shred-rom/blob/main/LICENSE)), adapted for this work.

---

## Usage

### Training

```bash
python train.py \
    --domain square --conv_type multiscale --layer_type fem_conv \
    --fem_use_radius True --fem_radius_mult 2.0 --fem_basis_type p1 --fem_k_hops 2 \
    --fem_lumped_mass True --mixing_type vector \
    --hidden_dim 128 --num_layers 4 --use_latent_transformer True \
    --pooling_type learned_res --use_edge_geom True --use_pos_reinject True --film_time_cond True \
    --sde_type ve --sigma_data 0.5 \
    --nx 32 --ny 32 --num_samples 5000 --batch_size 32 --num_epochs 250 --lr 1e-3
```

Sweep configs reproducing paper experiments are in `configs/`; run with `wandb sweep configs/<name>.yaml`.

### Evaluation

```bash
# Resolution invariance on the square: evaluate the blob model on same / finer / coarser meshes.
python evaluate.py resolution_invariance --run_dir checkpoints/gaussian_blob --eval_suite --n_samples 4 --n_steps 400

# Unconditional sample grids for the blob model (32x32 and 64x64).
python figures/gaussian_blob_samples.py --run_dir checkpoints/gaussian_blob --n_samples 4 --n_steps 400

# Resolution invariance for the multidomain model across the irregular shapes.
python evaluate.py resolution_invariance --run_dir checkpoints/multidomain --eval_suite --n_samples 4 --n_steps 400

# Sparse-sensor reconstruction on the blob (DPS) -> reconstruction_n50/, then render its figures.
python evaluate.py gaussian_blob_reconstruction --run_dir checkpoints/gaussian_blob --n_samples 4 --batch_size 16 --n_steps 400 --n_sensors 50
python figures/gaussian_blob_reconstruction.py checkpoints/gaussian_blob/reconstruction_n50

# Sparse-sensor reconstruction on the pinball flow (DPS), then render its figures.
python evaluate.py pinball_reconstruction --run_dir checkpoints/pinball --output_dir checkpoints/pinball/reconstruction --n_samples 4 --batch_size 16 --n_steps 400 --n_sensors 25 --skip_unconditional
python figures/pinball_reconstruction.py checkpoints/pinball/reconstruction

# Table 1: sweep sensor count (25/50/75/100) for both samplers, aggregating RMSE / energy score.
python evaluate.py sensor_sweep --task pinball_reconstruction --run_dir checkpoints/pinball --n_sensors 25 50 75 100 --samplers dps daps --n_samples 4 --n_steps 400 --skip_unconditional
```

`python evaluate.py <task> --help` for task-specific flags.

---

## License and citation

Released under the [Apache 2.0 License](LICENSE.txt).

```bibtex
@article{rowbottom2026grifdir,
  title   = {GRIFDIR: Graph Resolution-Invariant FEM Diffusion Models in
             Function Spaces over Irregular Domains},
  author  = {Rowbottom, James and Baker, Elizabeth L. and Huang, Nick and
             Adcock, Ben and Sch{\"o}nlieb, Carola-Bibiane and Denker, Alexander},
  journal = {arXiv preprint arXiv:2605.03497},
  year    = {2026}
}
```
