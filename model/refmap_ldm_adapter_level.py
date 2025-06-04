import copy
import itertools
import math
import os
import shutil
from collections import OrderedDict
from typing import Any, Mapping, Tuple

import cv2
import einops
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torchvision
from PIL import Image
from torch.nn import functional as F
from torch_fidelity import calculate_metrics
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.fid import FrechetInceptionDistance as FID

from dataloader.utils import Tile, Util
from ldm.models.diffusion.ddpm_map import LatentDiffusion
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, Downsample, ResBlock, TimestepEmbedSequential, UNetModel
from ldm.modules.diffusionmodules.util import conv_nd, linear, timestep_embedding, zero_module
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import exists, instantiate_from_config, log_txt_as_img
from utils.common import frozen_module
from utils.fid.fid_score import fid_score

from .adapters_rmap import RefMapAdapter
from .spaced_sampler import SpacedSampler


# Do forward process for UNetModel with prepared "control" tensors
class ControlledUnetModel(UNetModel):
    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        control=None,
        only_mid_control=False,
        **kwargs,
    ):
        # "control" is output of "ControlNet" model
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, context)
            if (i - 1) % 3 == 0 and ((i - 1) / 3 < len(control)):
                h = h + control[int((i - 1) / 3)]
            hs.append(h)

        h = self.middle_block(h, emb, context)

        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
            if i % 3 == 0 and ((3 - i / 3) < len(control)):
                h = h + control[int(3 - i / 3)]

        h = h.type(x.dtype)
        return self.out(h)

class ControlLDM(LatentDiffusion):
    def __init__(
        self,
        merge_type: str,
        source_key: str,
        ref_key: str,
        tile_name_key: str,
        sd_locked: bool,
        only_mid_control: bool,
        learning_rate: float,
        disable_preprocess=False,
        frozen_diff=True,
        *args,
        **kwargs,
    ) -> "ControlLDM":
        super().__init__(*args, **kwargs)
        # instantiate control module
        self.adapter = RefMapAdapter(merge_type=merge_type)
        self.merge_type = merge_type
        self.source_key = source_key
        self.ref_key = ref_key
        self.tile_name_key = tile_name_key
        self.disable_preprocess = disable_preprocess
        self.sd_locked = sd_locked
        self.only_mid_control = only_mid_control
        self.learning_rate = learning_rate
        self.frozen_diff = frozen_diff

        # instantiate preprocess module (SwinIR)
        # self.preprocess_model = instantiate_from_config(preprocess_config)
        # frozen_module(self.preprocess_model)

        if self.frozen_diff:
            self.model.eval()
            # self.model.train = disabled_train
            for name, param in self.model.named_parameters():
                if "attn" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def apply_cond_ref_encoder(self, control_source, control_ref_scale):
        if self.merge_type == "origin":
            cond_latent = self.adapter(control_source * 2 - 1, control_ref_scale)
        else:
            cond_latent = self.adapter(control_source * 2 - 1, control_ref_scale * 2 - 1) # extract merge features
        cond_latent = [cond * self.scale_factor for cond in cond_latent]
        return cond_latent

    def get_input(self, batch, k, bs=None, *args, **kwargs):
        with torch.no_grad():
            x, c = super().get_input(batch, self.target_key, *args, **kwargs)
            # x: target encoded (first stage)
            # c: conditional text (txt)

            source_cond = batch[self.source_key]
            if bs is not None:
                source_cond = source_cond[:bs]
            source_cond = source_cond.to(self.device)
            source_cond = einops.rearrange(source_cond, "b h w c -> b c h w")
            source_cond = source_cond.to(memory_format=torch.contiguous_format).float()

            if self.merge_type != "origin":
                ref_scale_cond = batch[self.ref_key]
                if bs is not None:
                    ref_scale_cond = ref_scale_cond[:bs]
                ref_scale_cond = ref_scale_cond.to(self.device)
                ref_scale_cond = einops.rearrange(ref_scale_cond, "b h w c -> b c h w")
                ref_scale_cond = ref_scale_cond.to(memory_format=torch.contiguous_format).float()
            else:
                ref_scale_cond = None

            source = source_cond

        # apply condition encoder
        cond_latent = self.apply_cond_ref_encoder(source_cond, ref_scale_cond)

        return x, dict(
            c_crossattn=[c],
            cond_latent=[cond_latent],
            source=[source],
        )

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond["c_crossattn"], 1)
        cond_control = cond["cond_latent"][0]

        eps = diffusion_model(
            x=x_noisy,
            timesteps=t,
            context=cond_txt,
            control=cond_control,
            only_mid_control=self.only_mid_control,
        )

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, sample_steps=50):
        log = dict()
        z, c = self.get_input(batch, self.target_key)

        # [-1 ,1]
        log["target"] = einops.rearrange(batch[self.target_key], "b h w c -> b c h w")
        # [0, 1]
        log["source"] = einops.rearrange(batch[self.source_key], "b h w c -> b c h w")
        # [0, 1]
        if self.merge_type != "origin":
            log["ref_scale"] = einops.rearrange(batch[self.ref_key], "b h w c -> b c h w")

        samples = self.sample_log(
            cond=c,
            steps=sample_steps,
        )
        # [0, 1]
        log["samples"] = samples

        return log

    @torch.no_grad()
    def sample_log(self, cond, steps):
        sampler = SpacedSampler(self)
        b, c, h, w = cond["cond_latent"][0][0].shape
        shape = (b, self.channels, h, w)
        samples = sampler.sample(steps, shape, cond)
        return samples

    def configure_optimizers(self):
        lr = self.learning_rate
        params = (
            list(self.adapter.parameters())
            + list(self.model.parameters())
        )
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def on_validation_epoch_start(self):
        self.val_psnr = []
        self.val_ssim = []

        self.val_save_dir = os.path.join(self.logger.save_dir, "val", f"step--{self.global_step}")
        if os.path.exists(self.val_save_dir):
            shutil.rmtree(self.val_save_dir)
        os.makedirs(self.val_save_dir, exist_ok=True)

        self.val_last_dir = os.path.join(self.logger.save_dir, "val", "last")
        os.makedirs(self.val_last_dir, exist_ok=True)

        self.val_target_dir = os.path.join(self.logger.save_dir, "val", "target")
        os.makedirs(self.val_target_dir, exist_ok=True)

        # instantiate metrics
        # self.metric_val_fid = FID(feature=64, normalize=True).set_dtype(torch.float64).to("cpu")
        self.metric_val_psnr = PSNR(data_range=1.0).to("cpu")
        self.metric_val_ssim = SSIM(data_range=1.0).to("cpu")

    def validation_step(self, batch, batch_idx):
        # bchw;[0, 1];tensor
        val_results = self.validation_inference(batch, batch_idx, self.val_save_dir, last_dir=self.val_last_dir, target_dir=self.val_target_dir)

        # calculate psnr and ssim
        target_batch_tensor = val_results["target"].detach().cpu()
        target_batch_tensor = (target_batch_tensor + 1) / 2.0
        sample_batch_tensor = val_results["samples"].detach().cpu()

        this_psnr = self.metric_val_psnr(sample_batch_tensor, target_batch_tensor).item()
        this_ssim = self.metric_val_ssim(sample_batch_tensor, target_batch_tensor).item()

        # update FID metric
        # self.metric_val_fid.update(sample_batch_tensor.double(), real=False)
        # self.metric_val_fid.update(target_batch_tensor.double(), real=True)
        # this_fid = self.metric_val_fid.compute()

        self.val_psnr.append(this_psnr)
        self.val_ssim.append(this_ssim)

    def on_validation_epoch_end(self):
        # calculate average metrics
        psnr = sum(self.val_psnr) / len(self.val_psnr)
        ssim = sum(self.val_ssim) / len(self.val_ssim)

        # compute FID score
        # fid = self.metric_val_fid.compute().item()
        # self.metric_val_fid.reset()
        fid = calculate_metrics(input1=self.val_target_dir, input2=self.val_last_dir, cuda=False, fid=True,
                                isc=False, kid=False, prc=False, verbose=False)['frechet_inception_distance']
        # fid = fid_score(real_path=self.val_target_dir, fake_path=self.val_last_dir, device='cpu')

        # log metrics out
        self.log("val/psnr", psnr)
        self.log("val/ssim", ssim)
        self.log("val/fid", fid)

        # log metrics file
        log_file_path = os.path.join(self.logger.save_dir, "log_metrics.txt")
        with open(log_file_path, "a") as f:
            f.write(f"val-step-{self.global_step}: PSNR={str(psnr)}, SSIM={str(ssim)}, FID={str(fid)}\n")

        # make log dir
        # log_dir = os.path.join(
        #     self.logger.save_dir, f"logs/val-step-{self.global_step}--psnr-{round(psnr, 3)}-ssim-{round(ssim, 3)}--fid-{round(fid, 3)}"
        # )
        # os.makedirs(log_dir, exist_ok=True)


    def validation_inference(self, batch, batch_idx, save_dir, last_dir=None, target_dir=None):
        # bchw;[0, 1];tensor
        val_results = self.log_images(batch)

        # save images
        for image_key in val_results:
            image = val_results[image_key].detach().cpu()
            if image_key == "target":
                image = (image + 1.0) / 2.0

            for i in range(len(image)):
                img_name = os.path.splitext(batch[self.tile_name_key][i])[0]
                curr_img = image[i]
                curr_img = curr_img.transpose(0, 1).transpose(1, 2).numpy()
                curr_img = (curr_img * 255).clip(0, 255).astype(np.uint8)
                filename = "{}_{}.png".format(img_name, image_key)
                path = os.path.join(save_dir, filename)
                Image.fromarray(curr_img).save(path)

                if last_dir and image_key == "samples":
                    filename = "{}.png".format(img_name)
                    path = os.path.join(last_dir, filename)
                    Image.fromarray(curr_img).save(path) # save image overlay
                elif target_dir and image_key == "target":
                    filename = "{}.png".format(img_name)
                    path = os.path.join(target_dir, filename)
                    Image.fromarray(curr_img).save(path) # save image overlay

        return val_results

    def on_test_epoch_start(self):
        string = ','.join([str(x) for x in self.trainer.gpus])
        print(f"Using GPU: {string}")
        os.environ["CUDA_VISIBLE_DEVICES"] = string
        # PSNR, SSIM metrics are set zero
        self.test_psnr = []
        self.test_ssim = []

        # instantiate metrics
        # self.metric_test_fid = FID(feature=64, normalize=True).to(self.device)
        self.metric_test_psnr = PSNR(data_range=1.0).to(self.device)
        self.metric_test_ssim = SSIM(data_range=1.0).to(self.device)

        # set image save dir
        self.test_save_dir = os.path.join(self.logger.save_dir, "samples")
        os.makedirs(self.test_save_dir, exist_ok=True)

        self.test_target_dir = os.path.join(self.logger.save_dir, "target")
        os.makedirs(self.test_target_dir, exist_ok=True)

    def test_step(self, batch, batch_idx):
        # inference
        test_results = self.test_inference(batch, batch_idx, self.test_save_dir, self.test_target_dir)

        # calculate psnr and ssim
        target_batch_tensor = test_results["target"].detach()
        target_batch_tensor = (target_batch_tensor + 1) / 2.0
        sample_batch_tensor = test_results["samples"].detach()

        this_psnr = self.metric_test_psnr(sample_batch_tensor, target_batch_tensor).item()
        this_ssim = self.metric_test_ssim(sample_batch_tensor, target_batch_tensor).item()

        # update FID metric
        # self.metric_test_fid.update(sample_batch_tensor.double(), real=False)
        # self.metric_test_fid.update(target_batch_tensor.double(), real=True)
        # this_fid = self.metric_test_fid.compute()

        self.test_psnr.append(this_psnr)
        self.test_ssim.append(this_ssim)

    def on_test_epoch_end(self):
        # calculate average metrics
        psnr = sum(self.test_psnr) / len(self.test_psnr)
        ssim = sum(self.test_ssim) / len(self.test_ssim)

        # compute FID score
        # fid = self.metric_test_fid.compute().item()
        # self.metric_test_fid.reset()
        fid = calculate_metrics(input1=self.test_target_dir, input2=self.test_save_dir, cuda=True, fid=True,
                                isc=False, kid=False, prc=False, verbose=False)['frechet_inception_distance']

        # log metrics out
        log_file = os.path.join(self.logger.save_dir, "log_test_metrics.txt")
        with open(log_file, "w") as f:
            f.write(f"test metrics {self.global_step}\n")
            f.write(f"PSNR: {str(psnr)}\n")
            f.write(f"SSIM: {str(ssim)}\n")
            f.write(f"FID: {str(fid)}\n")

        # make saving dir
        # log_dir = os.path.join(
        #     self.logger.save_dir, f"logs/test-step-{self.global_step}--psnr-{round(psnr, 3)}-ssim-{round(ssim, 3)}--fid-{round(fid, 3)}"
        # )
        # os.makedirs(log_dir, exist_ok=True)

    def test_inference(self, batch, batch_idx, save_dir=None, target_dir=None):
        os.makedirs(save_dir, exist_ok=True)
        test_results = self.log_images(batch)

        # save samples
        if save_dir:
            images = test_results["samples"].detach().cpu()
            for i in range(len(images)):
                img_name = batch[self.tile_name_key][i]
                curr_img = images[i]
                curr_img = curr_img.transpose(0, 1).transpose(1, 2).numpy()
                curr_img = (curr_img * 255).clip(0, 255).astype(np.uint8)
                filename = "{}.png".format(img_name)
                path = os.path.join(save_dir, filename)
                Image.fromarray(curr_img).save(path)
        # save target
        if target_dir:
            targets = test_results["target"].detach().cpu()
            for i in range(len(targets)):
                img_name = batch[self.tile_name_key][i]
                curr_img = targets[i]
                curr_img = curr_img.transpose(0, 1).transpose(1, 2).numpy()
                curr_img = (curr_img * 255).clip(0, 255).astype(np.uint8)
                filename = "{}.png".format(img_name)
                path = os.path.join(target_dir, filename)
                Image.fromarray(curr_img).save(path)

        return test_results
