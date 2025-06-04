from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from utils.common import instantiate_from_config, load_state_dict


def main() -> None:
    parser = ArgumentParser()

    ## Scale 2x with cascade
    # parser.add_argument("--config", type=str, default="configs/train_refmap_level_2c_wz.yaml")
    # parser.add_argument("--data_config", type=str, default="configs/datasetcfg/reference_map_val_level_2c_1816.yaml")
    # parser.add_argument("--ckpt", type=str, default="data/checkpoints/process_weight/0913-g1-s2-l1+tz-t197000-c31.937.ckpt")
    # parser.add_argument("--save_dir", type=str, default="data/val/refmap-level-val-s2c-1816")

    ## Scale 4x with cascade
    parser.add_argument("--config", type=str, default="configs/train_refmap_level_2c_wz.yaml")
    parser.add_argument("--data_config", type=str, default="configs/datasetcfg/reference_map_val_range_2c_wz.yaml")
    parser.add_argument("--ckpt", type=str, default="data/checkpoints/process_weight/0913-g1-s2-l1+tz-t197000-c31.937.ckpt")
    parser.add_argument("--save_dir", type=str, default="data/val/refmap-range-val-s2c-1815")

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.lightning.trainer.default_root_dir = args.save_dir
    config.data.params.val_config = args.data_config
    pl.seed_everything(config.lightning.seed, workers=True)

    data_module = instantiate_from_config(config.data)
    model = instantiate_from_config(OmegaConf.load(config.model.config))
    load_state_dict(model, torch.load(args.ckpt, map_location="cpu"), strict=False)

    callbacks = []
    for callback_config in config.lightning.callbacks:
        callbacks.append(instantiate_from_config(callback_config))
    trainer = pl.Trainer(callbacks=callbacks, **config.lightning.trainer)

    data_module.setup("validate")
    data_module.val_dataset.cascade_path = f"{args.save_dir}/val/last"
    data_loader = data_module.val_dataloader()

    trainer.validate(model, dataloaders=data_loader)

if __name__ == "__main__":
    main()
