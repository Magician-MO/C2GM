import os
from argparse import ArgumentParser

import einops
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.common import instantiate_from_config, load_state_dict


def print_data(dataset, dataloader, save_dir):
    print("dataset_len ", dataset.data_len)
    print("batch_size ", dataloader.batch_size)
    print("batch_len ", len(dataloader))
    print("cascade_path ", dataset.cascade_path)

    batch = next(iter(dataloader))
    for k, v in batch.items():
        if k in ["MAP", "RS", "REF"]:
            print(k, v.shape)
        elif k in ["scale", "level", "txt", "name", "ref_type"]:
            print(k, v[0])
        else:
            print(k, v)

    for idx, batch in enumerate(tqdm(dataloader)):
        img_batch = {"MAP": batch["MAP"], "RS": batch["RS"], "REF": batch["REF"]}
        img_name = batch["name"]
        ref_type = batch["ref_type"]
        for image_key in img_batch:
            images = einops.rearrange(img_batch[image_key], "b h w c -> b c h w")

            for i in range(len(images)):
                curr_img = images[i]
                curr_img = curr_img.transpose(0, 1).transpose(1, 2).numpy()
                curr_img = (curr_img * 255).clip(0, 255).astype(np.uint8)
                if image_key == "REF":
                    filename = "{}_{}-{}.png".format(img_name[i], image_key, ref_type[i])
                else:
                    filename = "{}_{}.png".format(img_name[i], image_key)
                path = os.path.join(save_dir, filename)
                Image.fromarray(curr_img).save(path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_refmap_level_cascade_output.yaml")
    parser.add_argument("--save_dir", type=str, default="data/test/test_dataset_cascade")
    parser.add_argument("--cascade_dir", type=str, default="data/exps/refmap-level-train-test1/val/last")

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    data_module = instantiate_from_config(config.data)
    data_module.setup("fit")
    # print("----------train_dataset----------")
    # print_data(data_module.train_dataset, data_module.train_dataloader(), args.save_dir)
    # print("----------val_dataset----------")
    # data_module.val_dataset.cascade_path = args.cascade_dir
    # print_data(data_module.val_dataset, data_module.val_dataloader(), args.save_dir)
    print("----------test_dataset----------")
    print_data(data_module.test_dataset, data_module.test_dataloader(), args.save_dir)
