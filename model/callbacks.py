import os
import shutil
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, ProgressBar, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader

from utils.common import instantiate_from_config

from .mixins import ImageLoggerMixin

__all__ = [
    "RichProgressBar",
    "ModelCheckpoint",
    "EarlyStopping",
    "ImageLogger",
    "ReloadCascadeImageOutput",
]

class ImageLogger(Callback):
    """
    Log images during training or validating.
    """
    #TODO: Support validating.
    def __init__(
        self,
        log_every_n_steps: int = 2000,
        max_images_each_step: int = 4,
        log_images_kwargs: Dict[str, Any] = None
    ) -> "ImageLogger":
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.max_images_each_step = max_images_each_step
        self.log_images_kwargs = log_images_kwargs or dict()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert isinstance(pl_module, ImageLoggerMixin)

    @rank_zero_only
    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT,
        batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if pl_module.global_step % self.log_every_n_steps == 0:
            is_train = pl_module.training
            if is_train:
                pl_module.freeze()

            with torch.no_grad():
                # returned images should be: nchw, rgb, [0, 1]
                images: Dict[str, torch.Tensor] = pl_module.log_images(batch, **self.log_images_kwargs)

            # save images
            save_dir = os.path.join(pl_module.logger.save_dir, "image_log", "train")
            os.makedirs(save_dir, exist_ok=True)
            for image_key in images:
                image = images[image_key].detach().cpu()
                if image_key == "hq":
                    image = (image + 1.0) / 2.0
                N = min(self.max_images_each_step, len(image))
                grid = torchvision.utils.make_grid(image[:N], nrow=4)
                # chw -> hwc (hw if gray)
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).numpy()
                grid = (grid * 255).clip(0, 255).astype(np.uint8)
                filename = "{}_step-{:06}_e-{:06}_b-{:06}.png".format(
                    image_key, pl_module.global_step, pl_module.current_epoch, batch_idx
                )
                path = os.path.join(save_dir, filename)
                Image.fromarray(grid).save(path)

            if is_train:
                pl_module.unfreeze()

class MeterlessProgressBar(ProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 0
        return bar

class ReloadCascadeImageOutput(Callback):
    def __init__(
        self,
        data_config: str,
        start_after_n_steps: int = 100000,
        test_every_n_epochs: int = 5,
    ) -> "ReloadCascadeImageOutput":
        super().__init__()
        self.data_config = data_config
        self.start_after_n_steps = start_after_n_steps
        self.test_every_n_epochs = test_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if (
            trainer.global_step >= self.start_after_n_steps and
            trainer.current_epoch >= self.test_every_n_epochs and
            trainer.current_epoch % self.test_every_n_epochs == 0
        ):

            save_dir = os.path.join(pl_module.logger.save_dir, "test")
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir)

            config = OmegaConf.load(self.data_config)

            test_dataset = instantiate_from_config(config["dataset"])
            test_dataloader = DataLoader(dataset=test_dataset, **config["data_loader"])

            # test on cascade level
            # trainer.test(model=pl_module, dataloaders=test_dataloader)

            for idx, batch in enumerate(test_dataloader):
                with torch.no_grad():
                    pl_module.test_inference(batch, idx, save_dir)

            print("ReloadCascadeImageOutput: test done.")