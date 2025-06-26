Welcome to the codebase for entropy guided folding!

Biorxiv: https://www.biorxiv.org/content/10.1101/2025.04.26.650728v1

Prerequisites:
 - Fresh conda environment with python=3.10
 - GCC
 - Cuda
 - aria2c (you can install with `sudo apt install aria2`)

Reproduction Steps:
# Install requirements
```
pip install requirements.txt
python3 setup.py build_ext --inplace
chmod +x setup_tools.sh
./setup_tools.sh
```
This installation should take less than 30 minutes on a standard computer.

# GPU requirements
Note that in order to run EGF, you must have sufficient GPU memory. Evaluating on longer sequences requires more GPU memory. If the GPU memory on your device is not enough, you will see a "Cuda out of memory" error. This indicates that you either need a GPU with more memory or run on multiple GPUs. The code will automatically detect multiple GPUs and split the model between the GPUs, and you can control which GPUs is uses by setting `CUDA_VISIBLE_DEVICES` (e.g. setting `CUDA_VISIBLE_DEVICES=0,1` forces the program to use GPUs 0 and 1). 

# Data
Make a new root folder where you will store the relevant information.
Make a file inside the folder which is the tag to cluster mapping (the keys are also used as a list of tags, so also called tag_path):
```json
{
    "AAAA_B": ["CCCC_D", "EEEE_F"],
    ...
}
```

Download a list of all PDB sequences from https://www.rcsb.org/downloads/fasta

You can download and process these sequences into a JSON with the following code:
```
wget https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz
gunzip pdb_seqres.txt.gz
```

```python
with open("pdb_seqres.txt", "r") as f:
    pdb_lines = [x.strip() for x in f.readlines()]

sequence_mapping = {}
for i in range(0, len(pdb_lines), 2):
    tag_line, sequence_line = pdb_lines[i], pdb_lines[i+1]
    tag = tag_line.split(" ")[0][1:]
    tag, chain = tag.split("_")
    tag = f"{tag.upper()}_{chain}"
    sequence_mapping[tag] = sequence_line

import json
with open("/your/path/to/sequence_mapping.json", "w") as f:
    json.dump(sequence_mapping, f)
```

Then download the AlphaFold model parameters with the following command:
```
mkdir /content/alphafold_params
chmod +x scripts/download_alphafold_params.sh
scripts/download_alphafold_params.sh /content/alphafold_params/
```

Run `write_inputs.py` to write MSAs, structures, and sequences.
```
python3 write_inputs.py --root_dir /your/root/dir \
                        --tag_path /your/tag/path.json \
                        --sequence_dict_path /your/sequence/dict.json \
                        --all
```
Note that the --all option will download sequence, structure, and MSA. If you would only like to download a subset of these three, you can use the individual --sequence, --structure, and --msa options.

Writing complete data for each protein (i.e. sequence, structure, MSA) should take less than 5 minutes each.
If this command is successful, the program should output progress bars for the MSA, structure, and sequence downloads.

# Entropy Guided Folding
Use `main.py` to run a single model of AF2, and use `run_all_settings.py` to run multiple models at once (most likely all models at once).

Specify config information in the configs directory.
- base: basic AF2 related configuration
- guide_config: modify parameters relating to the entropy guidance
- - Within guide_config there is a top level config and an _iter config. The _iter config controls parameters that change between iterations of recycling, while the top level config controls parameters that stay the same between iterations of recycling.
- Top level configs: These put together base and guide_configs, as well as input folders (which you can specify from the command line).

Sample `run_all_settings.py` command:
```
python3 run_all_settings.py --config-name=egf \
                --fasta_dir=/your/root/dir/sequences \
                --alignment_dir=/your/root/dir/alignments \
                --template_dir=/your/root/dir/templates \
                --output_dir=/your/root/dir/outputs \
                --structure_dir=/your/root/dir/structures \
                --info_dir=/your/root/dir/outputs \
                --plot_rmsd_path \
                --tag_cluster_mapping_path=/your/tag/path.json \
                --jax_param_path=/path/to/alphafold/params/params_{model_config}.npz \
                --crop_msa_num=512 \
                --skip_existing \
                --max_recycling_iters 1 \
                --models_to_run all \
                --data_random_seed 0
```

Sample `main.py` command:
```
python3 main.py --config-name=egf \
                fasta_dir=/your/root/dir/sequences \
                alignment_dir=/your/root/dir/alignments \
                template_dir=/your/root/dir/templates \
                output_dir=/your/root/dir/outputs \
                structure_dir=/your/root/dir/structures \
                guide_config.info_dir=/your/root/dir/outputs \
                guide_config.plot_rmsd_path=true \
                guide_config.tag_cluster_mapping_path=/your/tag/path.json \
                base.jax_param_path=/path/to/alphafold/params/params_model_3.npz \
                base.config_preset=model_3 \
                base.crop_msa_num=512 \
                guide_config.skip_existing=true
```
Running each structure's prediction should take less than 15 minutes per protein.
If this command is successful, the program should output progress bars for structure prediction, tables with intermediate metrics, and save structures to the output paths.

# Other experiments

## Ground truth guided folding
First, generate ground truth distograms using:
```
python3 save_gt_distograms.py --pred_path "/your/root/dir/outputs/model_3/{tag}_int_0.cif" \
                              --tag_cluster_mapping /your/tag/path.json \
                              --sequence_dict /your/sequence/dict.json \
                              --output_dir /your/root/dir/gt_distograms_3cb \
                              --structure_dir /your/root/dir/structures \
                              --which_to_save both
```

Next, use `main.py`
```
python3 main.py --config-name=gtgf \
                fasta_dir=/your/root/dir/sequences \
                alignment_dir=/your/root/dir/alignments \
                template_dir=/your/root/dir/templates \
                output_dir=/your/root/dir/gtgf_outputs \
                structure_dir=/your/root/dir/structures \
                guide_config.info_dir=/your/root/dir/gtgf_outputs \
                guide_config.plot_rmsd_path=true \
                guide_config.tag_cluster_mapping_path=/your/tag/path.json \
                guide_config.iter_0.gt_distances_path="/your/root/dir/gt_distograms_3cb/far/\{tag\}_dist.npy" \
                guide_config.iter_0.gt_mask_path="/your/root/dir/gt_distograms_3cb/far/\{tag\}_dist.npy" \
                base.jax_param_path=/path/to/alphafold/params/params_model_3.npz \
                base.config_preset=model_3 \
                base.crop_msa_num=512 \
                guide_config.skip_existing=true \
                base.data_random_seed=0
```

## Distogram saving for entropy computation
```
python3 main.py --config-name=egf \
                fasta_dir=/your/root/dir/sequences \
                alignment_dir=/your/root/dir/alignments \
                template_dir=/your/root/dir/templates \
                output_dir=/your/root/dir/egf_distogram_outputs \
                structure_dir=/your/root/dir/structures \
                guide_config.info_dir=/your/root/dir/egf_distogram_outputs \
                guide_config.plot_rmsd_path=true \
                guide_config.tag_cluster_mapping_path=/your/tag/path.json \
                base.jax_param_path=/path/to/alphafold/params/params_model_3.npz \
                base.config_preset=model_3 \
                base.crop_msa_num=512 \
                guide_config.skip_existing=true \
                guide_config.save_distograms=true \
                base.data_random_seed=0
```

## Loss tracking
```
python3 main.py --config-name=track \
                fasta_dir=/your/root/dir/sequences \
                alignment_dir=/your/root/dir/alignments \
                template_dir=/your/root/dir/templates \
                output_dir=/your/root/dir/track_outputs_seed0 \
                structure_dir=/your/root/dir/structures \
                guide_config.info_dir=/your/root/dir/track_outputs_seed0 \
                guide_config.plot_rmsd_path=true \
                guide_config.tag_cluster_mapping_path=/your/root/dir/stcm.json \
                base.jax_param_path=/path/to/alphafold/params/params_model_3.npz \
                base.config_preset=model_3 \
                base.crop_msa_num=512 \
                guide_config.skip_existing=true \
                guide_config.iter_0.gt_distances_path="/your/root/dir/gt_distograms_3cb/far/\{tag\}_dist.npy" \
                guide_config.iter_0.gt_mask_path="/your/root/dir/gt_distograms_3cb/far/\{tag\}_dist.npy" \
                base.data_random_seed=0
```

## Figure generation
```
python3 figure_generation/distogram_guidance.py --tag_cluster_mapping /your/tag/path.json \
                              --orders_path /your/root/dir/gt_distograms_3cb/orders.json \
                              --output_reduction_path /your/figure/dir/distogram_reduction.png \
                              --output_rmsd_path /your/figure/dir/distogram_rmsd.png \
                              --pred_pattern "/your/root/dir/gtgf_outputs/{tag}_int_{step}.cif" \
                              --structure_dir /your/root/dir/structures/
```

```
python3 figure_generation/entropy_guidance.py --tag_cluster_mapping /your/tag/path.json \
                              --distogram_pattern /your/root/dir/egf_distogram_outputs/{tag}_distograms_0.pt \
                              --entropy_increase_output_path /your/figure/dir/entropy_increase.png \
                              --entropy_change_output_path /your/figure/dir/entropy_change.png
```


```
python3 figure_generation/loss_tracking.py --tag_cluster_mapping /your/tag/path.json \
                              --metrics_pattern /your/root/dir/track_outputs/{tag}_metrics.json \
                              --sample_output_path /your/figure/dir/loss_sample.png \
                              --mean_output_path /your/figure/dir/loss_mean.png
```
Each of these figure generation commands should take less than 5 minutes.
If successful, these commands will save figures to the output paths specified.

# Evaluation
```
python3 evaluate.py --pred_file_pattern "/your/root/dir/outputs/model_*/{tag}_int_*.cif" \
                    --tag_cluster_mapping /your/tag/path.json \
                    --match_threshold 2.0 \
                    --output_path /your/output/path.json \
                    --structure_dir=/your/root/dir/structures \
                    --pca_reduction \
                    --reduce_num 10 \
                    --scatter_rmsds \
                    --pca_coloring \
                    --output_plot_pattern /your/figure/dir/{tag}.png
```
Important flags:
- --match_threshold: threshold for structures to count as a match (e.g. 2.0, 3.5)
- --pca_reduction: reduce using PCA
- --reduce_num: number of structures to reduce to _per side_ (i.e. total number of output structures will be 2 * reduce_num)
- --scatter_rmsds: whether to draw scatter plots of RMSDs and save them
- --pca_coloring: whether to color scattered points with PCA scores \
- --output_plot_pattern: where to save the output figures

The evaluation command should take less than 5 minutes.
If successful, these commands will output tables with data on RMSD calculations for each evaluated protein, as well overall statistics at the end of the output.

# References

Our code is built on top of the OpenFold Repository: https://github.com/aqlaboratory/openfold

Ahdritz, G. et al. OpenFold: retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization. Nat Methods 21, 1514-1524 (2024).

Our code uses the AlphaFold2 models: https://github.com/google-deepmind/alphafold

Jumper, J. et al. Highly accurate protein structure prediction with AlphaFold. Nature 596, 583-589 (2021).

# Questions
If you have any questions about the code please feel free to file an issue.