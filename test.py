from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from utils.common import instantiate_from_config, load_state_dict


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/test_refmap_level_2c_oz.yaml")
    parser.add_argument("--data_config", type=str, default="configs/datasetcfg/reference_map_test_range_2c_oz.yaml")
    parser.add_argument("--ckpt", type=str, default="data/checkpoints/process_weight/g1-l1-2c+oz-1102-t165500-c30.771.ckpt")
    parser.add_argument("--save_dir", type=str, default="data/test/refmap-level-test-range-2c-oz")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.lightning.trainer.default_root_dir = args.save_dir
    config.data.params.test_config = args.data_config
    pl.seed_everything(config.lightning.seed, workers=True)

    data_module = instantiate_from_config(config.data)
    model = instantiate_from_config(OmegaConf.load(config.model.config))
    load_state_dict(model, torch.load(args.ckpt, map_location="cpu"), strict=False)

    callbacks = []
    for callback_config in config.lightning.callbacks:
        callbacks.append(instantiate_from_config(callback_config))
    trainer = pl.Trainer(callbacks=callbacks, **config.lightning.trainer)

    model.eval()
    data_module.setup("test")
    data_module.test_dataset.cascade_path = f"{args.save_dir}/samples"
    data_loader = data_module.test_dataloader()
    trainer.test(model, dataloaders=data_loader)

if __name__ == "__main__":
    main()
