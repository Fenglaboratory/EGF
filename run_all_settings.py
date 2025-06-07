import sys
import os
import argparse


def bold(string):
    return f"\033[1m{string}\033[0m"


model_list = [
    "model_1",
    "model_1_ptm",
    "model_2",
    "model_2_ptm",
    "model_3",
    "model_3_ptm",
    "model_4",
    "model_4_ptm",
    "model_5",
    "model_5_ptm",
]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-name", type=str, required=True, help="Name of the config file"
    )
    parser.add_argument(
        "--fasta_dir", type=str, required=True, help="Directory containing fasta files"
    )
    parser.add_argument(
        "--alignment_dir",
        type=str,
        required=True,
        help="Directory containing alignment files",
    )
    parser.add_argument(
        "--template_dir",
        type=str,
        required=True,
        help="Directory containing template files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save output files"
    )
    parser.add_argument(
        "--structure_dir",
        type=str,
        required=True,
        help="Directory containing structure files",
    )

    parser.add_argument(
        "--info_dir",
        type=str,
        required=True,
        help="Directory to save guide config files",
    )
    parser.add_argument(
        "--plot_rmsd_path", action="store_true", help="Whether to plot RMSD path"
    )
    parser.add_argument(
        "--tag_cluster_mapping_path",
        type=str,
        required=True,
        help="Path to tag cluster mapping file",
    )
    parser.add_argument(
        "--jax_param_path",
        type=str,
        required=True,
        help="Path to JAX params file. Should contain {model_config} as a placeholder.",
    )
    parser.add_argument(
        "--crop_msa_num",
        type=int,
        required=True,
        help="Number of MSA sequences to crop",
    )
    parser.add_argument(
        "--skip_existing", action="store_true", help="Skip existing output files"
    )
    parser.add_argument(
        "--max_recycling_iters",
        type=int,
        default=1,
        help="Maximum number of recycling iterations",
    )
    parser.add_argument("--data_random_seed", type=int, default=0)

    parser.add_argument(
        "--models_to_run", type=str, nargs="+", required=True, help="Config preset"
    )

    return parser.parse_args()


bool_to_str = lambda x: "true" if x else "false"


def main():
    args = get_args()

    model_config_list = (
        model_list if "all" in args.models_to_run else args.models_to_run
    )

    for model_config in model_config_list:
        output_dir = os.path.join(args.output_dir, model_config)
        os.makedirs(output_dir, exist_ok=True)

        info_dir = os.path.join(args.info_dir, model_config)
        os.makedirs(info_dir, exist_ok=True)

        cmd = f"""python3 main.py --config-name={args.config_name} \
                    fasta_dir={args.fasta_dir} \
                    alignment_dir={args.alignment_dir} \
                    template_dir={args.template_dir} \
                    output_dir={output_dir} \
                    structure_dir={args.structure_dir} \
                    guide_config.info_dir={info_dir} \
                    guide_config.plot_rmsd_path={bool_to_str(args.plot_rmsd_path)} \
                    guide_config.tag_cluster_mapping_path={args.tag_cluster_mapping_path} \
                    base.jax_param_path={args.jax_param_path.format(model_config=model_config)} \
                    base.config_preset={model_config} \
                    base.crop_msa_num={args.crop_msa_num} \
                    guide_config.skip_existing={args.skip_existing} \
                    guide_config.max_recycling_iters={args.max_recycling_iters} \
                    base.data_random_seed={args.data_random_seed}"""
        print(bold(cmd))
        os.system(cmd)


if __name__ == "__main__":
    main()
