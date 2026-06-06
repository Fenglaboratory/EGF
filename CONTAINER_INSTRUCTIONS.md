# EGF Container Instructions

This document explains how to build and run the Entropy Guided Folding (EGF) container using [Apptainer](https://apptainer.org) (formerly Singularity). The resulting `.sif` image is fully self-contained and portable across Linux systems with an NVIDIA GPU.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Apptainer ≥ 1.0** | `apptainer --version`. On some clusters the command is still `singularity`. |
| **NVIDIA GPU** | CUDA compute capability 3.7+ (Kepler and newer). |
| **NVIDIA driver** | Must match or exceed CUDA 12.1 requirement (driver ≥ 525). |
| **~25 GB disk** | For the `.sif` image and conda packages during build. |
| **Internet access** | Required during build to pull packages from PyPI/conda/GitHub. |

> **Cluster note:** The build step typically requires `--fakeroot` or root access. Most HPC clusters support `--fakeroot`; contact your sysadmin if it is not enabled.

---

## 1. Building the Container

Run from the **EGF project root** (the directory containing `egf.def`):

```bash
cd /path/to/EGF_public

# Option A – fakeroot (recommended on HPC clusters, no root needed)
apptainer build --fakeroot --bind "$(pwd)":/tmp/EGF_build_src egf.sif egf.def

# Option B – root (if you have sudo on the build machine)
sudo apptainer build --bind "$(pwd)":/tmp/EGF_build_src egf.sif egf.def
```

The build downloads conda packages, PyTorch wheels, and compiles CUDA/Cython extensions. It typically takes **20–45 minutes** on a machine with a good internet connection.

The `--bind "$(pwd)":/tmp/EGF_build_src` flag makes the project source available inside the container during the build, so you **must** run the command from the project root.

### Verify the build

```bash
apptainer test egf.sif
# Should print: "=== All checks passed ==="
```

---

## 2. Transferring to Another System

The `.sif` file is a single portable image. Copy it to any Linux machine with Apptainer and an NVIDIA GPU:

```bash
scp egf.sif user@remote-cluster:/scratch/user/
```

No conda, no Python installation, no compilation needed on the target machine.

---

## 3. Data Preparation (one-time, outside the container)

Large data (AlphaFold parameters, MSAs, structures) are **not** included in the image and must be prepared on the host. Bind-mount them at runtime (see Section 4).

### AlphaFold parameters

```bash
mkdir -p /your/alphafold_params
chmod +x scripts/download_alphafold_params.sh
apptainer exec egf.sif bash /opt/EGF_public/scripts/download_alphafold_params.sh /your/alphafold_params/
```

### PDB sequence database

```bash
wget https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz
gunzip pdb_seqres.txt.gz
```

Then create `sequence_mapping.json`:

```bash
apptainer exec egf.sif python3 - <<'EOF'
import json
with open("pdb_seqres.txt") as f:
    lines = [x.strip() for x in f.readlines()]
mapping = {}
for i in range(0, len(lines), 2):
    tag = lines[i].split(" ")[0][1:]
    pdb, chain = tag.split("_")
    mapping[f"{pdb.upper()}_{chain}"] = lines[i+1]
with open("sequence_mapping.json", "w") as f:
    json.dump(mapping, f)
print("Wrote sequence_mapping.json")
EOF
```

### Download MSAs, structures, sequences

```bash
apptainer exec --nv \
    --bind /your/root_dir:/data/root \
    egf.sif python3 /opt/EGF_public/write_inputs.py \
        --root_dir /data/root \
        --tag_path /data/root/stcm.json \
        --sequence_dict_path /data/root/sequence_mapping.json \
        --all
```

---

## 4. Running the Container

All commands use `--nv` (GPU pass-through) and `--bind` to expose host data paths inside the container. You can bind any number of directories.

### Convenience alias

```bash
alias egf='apptainer exec --nv \
    --bind /your/alphafold_params:/data/alphafold_params \
    --bind /your/root_dir:/data/root \
    egf.sif python3 /opt/EGF_public'
```

Replace the bind paths with your actual data locations.

---

### 4a. Entropy Guided Folding (EGF) — main experiment

```bash
apptainer exec --nv \
    --bind /your/alphafold_params:/data/alphafold_params \
    --bind /your/root_dir:/data/root \
    egf.sif python3 /opt/EGF_public/run_all_settings.py \
        --config-name=egf \
        --fasta_dir=/data/root/sequences \
        --alignment_dir=/data/root/alignments \
        --template_dir=/data/root/templates \
        --output_dir=/data/root/outputs \
        --structure_dir=/data/root/structures \
        --info_dir=/data/root/outputs \
        --plot_rmsd_path \
        --tag_cluster_mapping_path=/data/root/stcm.json \
        --jax_param_path=/data/alphafold_params/params_{model_config}.npz \
        --crop_msa_num=512 \
        --skip_existing \
        --max_recycling_iters 1 \
        --models_to_run all \
        --data_random_seed 0
```

Single-model variant using `main.py`:

```bash
apptainer exec --nv \
    --bind /your/alphafold_params:/data/alphafold_params \
    --bind /your/root_dir:/data/root \
    egf.sif python3 /opt/EGF_public/main.py \
        --config-name=egf \
        fasta_dir=/data/root/sequences \
        alignment_dir=/data/root/alignments \
        template_dir=/data/root/templates \
        output_dir=/data/root/outputs \
        structure_dir=/data/root/structures \
        guide_config.info_dir=/data/root/outputs \
        guide_config.plot_rmsd_path=true \
        guide_config.tag_cluster_mapping_path=/data/root/stcm.json \
        base.jax_param_path=/data/alphafold_params/params_model_3.npz \
        base.config_preset=model_3 \
        base.crop_msa_num=512 \
        guide_config.skip_existing=true
```

---

### 4b. Ground Truth Guided Folding (GTGF)

First generate ground truth distograms:

```bash
apptainer exec --nv \
    --bind /your/alphafold_params:/data/alphafold_params \
    --bind /your/root_dir:/data/root \
    egf.sif python3 /opt/EGF_public/save_gt_distograms.py \
        --pred_path "/data/root/outputs/model_3/{tag}_int_0.cif" \
        --tag_cluster_mapping /data/root/stcm.json \
        --sequence_dict /data/root/sequence_mapping.json \
        --output_dir /data/root/gt_distograms_3cb \
        --structure_dir /data/root/structures \
        --which_to_save both
```

Then run GTGF:

```bash
apptainer exec --nv \
    --bind /your/alphafold_params:/data/alphafold_params \
    --bind /your/root_dir:/data/root \
    egf.sif python3 /opt/EGF_public/main.py \
        --config-name=gtgf \
        fasta_dir=/data/root/sequences \
        alignment_dir=/data/root/alignments \
        template_dir=/data/root/templates \
        output_dir=/data/root/gtgf_outputs \
        structure_dir=/data/root/structures \
        guide_config.info_dir=/data/root/gtgf_outputs \
        guide_config.plot_rmsd_path=true \
        guide_config.tag_cluster_mapping_path=/data/root/stcm.json \
        "guide_config.iter_0.gt_distances_path=/data/root/gt_distograms_3cb/far/{tag}_dist.npy" \
        "guide_config.iter_0.gt_mask_path=/data/root/gt_distograms_3cb/far/{tag}_dist.npy" \
        base.jax_param_path=/data/alphafold_params/params_model_3.npz \
        base.config_preset=model_3 \
        base.crop_msa_num=512 \
        guide_config.skip_existing=true \
        base.data_random_seed=0
```

---

### 4c. Evaluation

```bash
apptainer exec --nv \
    --bind /your/root_dir:/data/root \
    --bind /your/figures:/figures \
    egf.sif python3 /opt/EGF_public/evaluate.py \
        --pred_file_pattern "/data/root/outputs/model_*/{tag}_int_*.cif" \
        --tag_cluster_mapping /data/root/stcm.json \
        --match_threshold 2.0 \
        --output_path /data/root/results.json \
        --structure_dir=/data/root/structures \
        --pca_reduction \
        --reduce_num 10 \
        --scatter_rmsds \
        --pca_coloring \
        --output_plot_pattern "/figures/{tag}.png"
```

---

### 4d. Figure generation

```bash
# Distogram guidance
apptainer exec --nv \
    --bind /your/root_dir:/data/root \
    --bind /your/figures:/figures \
    egf.sif python3 /opt/EGF_public/figure_generation/distogram_guidance.py \
        --tag_cluster_mapping /data/root/stcm.json \
        --orders_path /data/root/gt_distograms_3cb/orders.json \
        --output_reduction_path /figures/distogram_reduction.png \
        --output_rmsd_path /figures/distogram_rmsd.png \
        --pred_pattern "/data/root/gtgf_outputs/{tag}_int_{step}.cif" \
        --structure_dir /data/root/structures

# Entropy guidance
apptainer exec --nv \
    --bind /your/root_dir:/data/root \
    --bind /your/figures:/figures \
    egf.sif python3 /opt/EGF_public/figure_generation/entropy_guidance.py \
        --tag_cluster_mapping /data/root/stcm.json \
        --distogram_pattern /data/root/egf_distogram_outputs/{tag}_distograms_0.pt \
        --entropy_increase_output_path /figures/entropy_increase.png \
        --entropy_change_output_path /figures/entropy_change.png

# Loss tracking
apptainer exec --nv \
    --bind /your/root_dir:/data/root \
    --bind /your/figures:/figures \
    egf.sif python3 /opt/EGF_public/figure_generation/loss_tracking.py \
        --tag_cluster_mapping /data/root/stcm.json \
        --metrics_pattern /data/root/track_outputs/{tag}_metrics.json \
        --sample_output_path /figures/loss_sample.png \
        --mean_output_path /figures/loss_mean.png
```

---

## 5. GPU Selection

Use `CUDA_VISIBLE_DEVICES` to select which GPUs to use (the code automatically detects multiple GPUs and splits the model across them):

```bash
# Use GPUs 0 and 1 only
CUDA_VISIBLE_DEVICES=0,1 apptainer exec --nv --bind ... egf.sif python3 /opt/EGF_public/run_all_settings.py ...

# Or pass it through Apptainer's env flag
apptainer exec --nv --env CUDA_VISIBLE_DEVICES=0 --bind ... egf.sif python3 ...
```

---

## 6. Interactive Shell

Open an interactive shell inside the container for debugging or exploration:

```bash
apptainer shell --nv \
    --bind /your/alphafold_params:/data/alphafold_params \
    --bind /your/root_dir:/data/root \
    egf.sif

# Inside the container:
Apptainer> python3 /opt/EGF_public/main.py --help
Apptainer> python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## 7. SLURM Job Submission

Example SLURM batch script:

```bash
#!/bin/bash
#SBATCH --job-name=egf_run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=egf_%j.log

module load apptainer   # or: module load singularity

apptainer exec --nv \
    --bind /your/alphafold_params:/data/alphafold_params \
    --bind /your/root_dir:/data/root \
    /path/to/egf.sif python3 /opt/EGF_public/run_all_settings.py \
        --config-name=egf \
        --fasta_dir=/data/root/sequences \
        --alignment_dir=/data/root/alignments \
        --template_dir=/data/root/templates \
        --output_dir=/data/root/outputs \
        --structure_dir=/data/root/structures \
        --info_dir=/data/root/outputs \
        --tag_cluster_mapping_path=/data/root/stcm.json \
        --jax_param_path=/data/alphafold_params/params_{model_config}.npz \
        --crop_msa_num=512 \
        --skip_existing \
        --models_to_run all \
        --data_random_seed 0
```

---

## 8. Troubleshooting

### Build fails: "permission denied" or "fakeroot not configured"

```bash
# Check if fakeroot is available
apptainer config fakeroot --list

# If not, ask your sysadmin or try building as root on a local machine
# and transferring the .sif
sudo apptainer build egf.sif egf.def
```

### Build fails: conda/pip network errors

The build requires internet access. On a login node with limited network:

```bash
# Some clusters block outbound connections from compute nodes; build on login node
# If conda channel is blocked, try using a proxy:
export https_proxy=http://your-proxy:port
apptainer build --fakeroot egf.sif egf.def
```

### GPU not detected at runtime

```bash
# Verify --nv is passed
apptainer exec --nv egf.sif python3 -c "import torch; print(torch.cuda.is_available())"

# Check driver compatibility
nvidia-smi
# Driver must support CUDA 12.1+ (driver version >= 525)
```

### "CUDA out of memory" during folding

EGF requires significant GPU memory for longer sequences. Options:
- Use multiple GPUs: `CUDA_VISIBLE_DEVICES=0,1`
- Reduce MSA depth: `--crop_msa_num=256`
- Run shorter sequences first to estimate memory needs

### First run is slow (CUDA JIT compilation)

If your GPU is newer than sm_80 (e.g. RTX 3090 / sm_86, H100 / sm_90), CUDA will JIT-compile PTX on first run. This is a one-time cost per GPU model. Subsequent runs are fast.

### ModuleNotFoundError for `egf` or `openfold`

The container sets `PYTHONPATH=/opt/EGF_public`. If you override the environment, restore it:

```bash
apptainer exec --nv --env PYTHONPATH=/opt/EGF_public egf.sif python3 /opt/EGF_public/main.py ...
```

---

## 9. What's Inside the Container

| Component | Location |
|---|---|
| EGF source code | `/opt/EGF_public/` |
| Python (3.10) + conda packages | `/opt/conda/` |
| PyTorch 2.6.0 (CUDA 12.1) | site-packages |
| OpenFold (compiled) | site-packages |
| CUDA attention kernel | site-packages (`attn_core_inplace_cuda`) |
| Cython alignment tools | `/opt/EGF_public/tools/` |
| hmmer / hhsuite / kalign2 | `/opt/conda/bin/` |
| openmm / pdbfixer | `/opt/conda/` |

Data (AlphaFold parameters, MSAs, PDB sequences) are **not** included and must be bind-mounted at runtime.
