# Subspace Selection on the Grassmannian for Koopman Autoencoders

Official code for the ESANN 2027 submission **"Subspace Selection on the Grassmannian for Koopman Autoencoders"**. [MIT License](LICENSE).

## Setup

```bash
pip install -r requirements.txt
```

The pinned environment uses Python 3.11.18 and PyTorch with CUDA 12.1.

Run commands from the repository root. Set the device in
`src/configs/system_config.yaml` or pass `--device cpu` / `--device cuda:0`.
When no device is configured, `auto` selects CUDA if available, otherwise CPU.
Install **FFmpeg** for video export.

## Data preparation

Synthetic datasets are generated automatically. Goodwin, Lorenz, and Dysts
trajectories are cached under `data/<dataset>/cache/`.

For the two real-world datasets, download:

- **PAMY2:** [Recordings 1–6](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.OMM0JP).
  Extract `pamy_data_*.df` into `data/PAMy2/pam_data_1-6/`.
- **Monkey reaching:** [DANDI 000070, subject Jenkins](https://dandiarchive.org/dandiset/000070).
  Place `.nwb` files in `data/monkey_reaching/reaching_data/sub-Jenkins/`.

```bash
python data/PAMy2/process_pamy.py
python data/monkey_reaching/process_reaching.py --phases execution --bin-ms 10 --smooth-ms 10
```

These create `data/PAMy2/pamy_data.h5` and
`data/monkey_reaching/reaching_data.h5`. Use `--overwrite` to regenerate them.

## Train

```bash
# KAE
python src/train.py --config src/configs/lorenz/koopman_baseline.yaml \
  --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder \
  --no-grassmann --outdir outputs/lorenz/koopman_baseline_grassmann_false

# GKAE
python src/train.py --config src/configs/lorenz/koopman_baseline_grassmann.yaml \
  --model_file src/model/koopman_baseline.py --model_class KoopmanAutoencoder \
  --grassmann --outdir outputs/lorenz/koopman_baseline_grassmann_true
```

Replace `lorenz` with a dataset folder from `src/configs/`, including
`monkey_reaching` or `PAMy2`. Edit its YAML to configure data, model, loss, and
training. More experiment commands are in `train.sh`.

## Visualize

```bash
python src/visualize.py --checkpoint outputs/lorenz/koopman_baseline_grassmann_true/best.pt
```

Add `--animate` for MP4s; omit `--checkpoint` to select a run interactively.

## Summarize

```bash
python src/summarize.py --dataset lorenz
```

Compares rollout NRMSE, DTW, normalized DTW, and improvements across models.
Add `--checkpoint_type last` to evaluate the latest checkpoint.

## Video gallery

State-conditioned Grassmannian subspaces along validation trajectories, shown in Fig. 2 in the paper.

| **Goodwin Oscillator** | **Lorenz-63** |
|:---:|:---:|
| ![Goodwin subspace animation](fig/videos/05c_goodwin_state_conditioned_grassmann_rgb.mp4) | ![Lorenz subspace animation](fig/videos/05c_lorenz_state_conditioned_grassmann_rgb.mp4) |
| **Chua** | **Dadras** |
| ![Chua subspace animation](fig/videos/05c_chua_state_conditioned_grassmann_rgb.mp4) | ![Dadras subspace animation](fig/videos/05c_dadras_state_conditioned_grassmann_rgb.mp4) |
| **Lü–Chen–Cheng** | **Shimizu–Morioka** |
| ![Lü–Chen–Cheng subspace animation](fig/videos/05c_lu_chen_cheng_state_conditioned_grassmann_rgb.mp4) | ![Shimizu–Morioka subspace animation](fig/videos/05c_shimizu_morioka_state_conditioned_grassmann_rgb.mp4) |
