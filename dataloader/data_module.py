from typing import Any, Mapping, Tuple

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset

from dataloader.batch_transform import BatchTransform, IdentityBatchTransform
from utils.common import instantiate_from_config


class BIRDataModule(pl.LightningDataModule):

    def __init__(
        self,
        train_config: str = None,
        val_config: str = None,
        test_config: str = None
    ) -> "BIRDataModule":
        super().__init__()
        self.train_config = OmegaConf.load(train_config) if train_config else None
        self.val_config = OmegaConf.load(val_config) if val_config else None
        self.test_config = OmegaConf.load(test_config) if test_config else None

    def load_dataset(self, config: Mapping[str, Any]) -> Tuple[Dataset, BatchTransform]:
        dataset = instantiate_from_config(config["dataset"])
        batch_transform = (
            instantiate_from_config(config["batch_transform"])
            if config.get("batch_transform") else IdentityBatchTransform()
        )
        return dataset, batch_transform

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset, self.train_batch_transform = self.load_dataset(self.train_config)
            if self.val_config:
                self.val_dataset, self.val_batch_transform = self.load_dataset(self.val_config)
            else:
                self.val_dataset, self.val_batch_transform = None, None
            if self.test_config:
                self.test_dataset, self.test_batch_transform = self.load_dataset(self.test_config)
            else:
                self.test_dataset, self.test_batch_transform = None, None
        elif stage == "validate":
            self.val_dataset, self.val_batch_transform = self.load_dataset(self.val_config)
        elif stage == "test":
            self.test_dataset, self.test_batch_transform = self.load_dataset(self.test_config)
        else:
            raise NotImplementedError(stage)

    def reload_train_dataset(self) -> None:
        self.train_dataset, self.train_batch_transform = self.load_dataset(self.train_config)

    def reload_val_dataset(self) -> None:
        self.val_dataset, self.val_batch_transform = self.load_dataset(self.val_config)

    def reload_test_dataset(self) -> None:
        self.test_dataset, self.test_batch_transform = self.load_dataset(self.test_config)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset, **self.train_config["data_loader"]
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.val_dataset is None:
            return None
        return DataLoader(
            dataset=self.val_dataset, **self.val_config["data_loader"]
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.test_dataset is None:
            return None
        return DataLoader(
            dataset=self.test_dataset, **self.test_config["data_loader"]
        )

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        self.trainer: pl.Trainer

        if self.trainer.training:
            return self.train_batch_transform(batch)
        elif self.trainer.validating or self.trainer.sanity_checking:
            return self.val_batch_transform(batch)
        else:
            raise RuntimeError(
                "Trainer state: \n"
                f"training: {self.trainer.training}\n"
                f"validating: {self.trainer.validating}\n"
                f"testing: {self.trainer.testing}\n"
                f"predicting: {self.trainer.predicting}\n"
                f"sanity_checking: {self.trainer.sanity_checking}"
            )