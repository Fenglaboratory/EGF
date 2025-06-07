from egf.folder import Folder
import hydra


@hydra.main(config_path="configs/", config_name="config")
def main(cfg):
    folder = Folder(cfg)
    folder.run(
        fasta_dir=cfg.fasta_dir,
        alignment_dir=cfg.alignment_dir,
        template_dir=cfg.template_dir,
        output_dir=cfg.output_dir,
        structure_dir=cfg.structure_dir,
    )


if __name__ == "__main__":
    main()
