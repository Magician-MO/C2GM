import math
import os
import shutil
from argparse import ArgumentParser, Namespace

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from tqdm import tqdm

from ldm.xformers_state import disable_xformers
from utils.common import instantiate_from_config, load_state_dict
from utils.fid.fid_score import fid_score

# from model.Flows.mu_sigama_estimate_normflows import CreateFlow


def parse_args() -> Namespace:
    parser = ArgumentParser()

    # TODO: add help info for these options
    parser.add_argument(
        "--ckpt",
        default="data/checkpoints/process_weight/0917-g1-s4-l1+tz-t189500-c31.208.ckpt",
        type=str,
        help="full checkpoint path",
    )
    parser.add_argument(
        "--style_scale",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--sample_style",
        default=False,
        # default = None,
        type=str,
        help="Whether to perform style sampling from the pretrained normalizing flow model. If true, 'ckpt_flow_mean' and 'ckpt_flow_std' must not be 'None'",
    )
    parser.add_argument(
        "--ckpt_flow_mean",
        default="model/Flows/checkpoints/flow_tanh_mini_mean",
        type=str,
        help="full checkpoint path",
    )
    parser.add_argument(
        "--ckpt_flow_std",
        default="model/Flows/checkpoints/flow_tanh_mini_std",
        type=str,
        help="full checkpoint path",
    )
    parser.add_argument(
        "--model_config",
        default="configs/modelcfg/refmap_level_wc.yaml",
        type=str,
        help="model config path",
    )
    parser.add_argument(
        "--data_config", type=str, default="configs/datasetcfg/reference_map_test_range_4c_AVG.yaml"
    )
    # FlowSampler-sampleFromTrain
    parser.add_argument(
        "--output", type=str, default="data/output/refmap-level-inference"
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda:0", choices=["cpu", "cuda", "mps"]
    )

    return parser.parse_args()


def check_device(device):
    if device == "cuda" or "cuda" in device:
        # check if CUDA is available
        if not torch.cuda.is_available():
            print(
                "CUDA not available because the current PyTorch install was not "
                "built with CUDA enabled."
            )
            device = "cpu"
    else:
        # xformers only support CUDA. Disable xformers when using cpu or mps.
        disable_xformers()
        if device == "mps":
            # check if MPS is available
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print(
                        "MPS not available because the current PyTorch install was not "
                        "built with MPS enabled."
                    )
                    device = "cpu"
                else:
                    print(
                        "MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine."
                    )
                    device = "cpu"
    print(f"using device {device}")
    return device

def main() -> None:
    args = parse_args()
    device = check_device(args.device)

    pl.seed_everything(args.seed)

    # output dir
    sample_dir = os.path.join(args.output, "samples")
    target_dir = os.path.join(args.output, "targets")
    log_file_path = os.path.join(args.output, "log_metrics.txt")

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    test_dataset = instantiate_from_config(OmegaConf.load(args.data_config)["dataset"])
    test_dataset.cascade_path = sample_dir
    test_dataloader = DataLoader(
        dataset=test_dataset, **(OmegaConf.load(args.data_config)["data_loader"])
    )
    print("dataset loaded!!!!!")

    model = instantiate_from_config(OmegaConf.load(args.model_config))

    static_dic = torch.load(args.ckpt, map_location="cpu")
    load_state_dict(model, static_dic, strict=False)

    # if args.ckpt_flow_mean and args.ckpt_flow_std:
    #     flow_mean = CreateFlow(dim=32, num_layers=16, hidden_layers=[16, 64, 64, 32])
    #     static_dic_flow_mean = torch.load(args.ckpt_flow_mean, map_location="cpu")
    #     load_state_dict(flow_mean, static_dic_flow_mean, strict=True)
    #     model.flow_mean = flow_mean

    #     flow_std = CreateFlow(dim=32, num_layers=16, hidden_layers=[16, 64, 64, 32])
    #     static_dic_flow_std = torch.load(args.ckpt_flow_std, map_location="cpu")
    #     load_state_dict(flow_std, static_dic_flow_std, strict=True)
    #     model.flow_std = flow_std

    model.freeze()
    model.to(device)

    # PSNR, SSIM metrics are set zero
    test_psnr = []
    test_ssim = []

    # instantiate metrics
    metric_test_fid = FID(feature=64, normalize=True).to(device)
    metric_test_psnr = PSNR(data_range=1.0).to(device)
    metric_test_ssim = SSIM(data_range=1.0).to(device)

    for idx, batch in enumerate(tqdm(test_dataloader)):
        model.eval()
        test_results = model.test_inference(batch, idx, sample_dir, save_target=True, target_dir=target_dir)
        # test_results = model.test_inference(batch, idx, args.output, args.sample_style, args.style_scale)

        # calculate psnr and ssim
        target_batch_tensor = test_results["target"].detach().to(device)
        target_batch_tensor = (target_batch_tensor + 1) / 2.0
        sample_batch_tensor = test_results["samples"].detach().to(device)

        this_psnr = metric_test_psnr(sample_batch_tensor, target_batch_tensor).item()
        this_ssim = metric_test_ssim(sample_batch_tensor, target_batch_tensor).item()

        # update FID metric
        metric_test_fid.update(sample_batch_tensor.double(), real=False)
        metric_test_fid.update(target_batch_tensor.double(), real=True)
        # this_fid = metric_test_fid.compute()

        test_psnr.append(this_psnr)
        test_ssim.append(this_ssim)

    # calculate average metrics
    psnr = sum(test_psnr) / len(test_psnr)
    ssim = sum(test_ssim) / len(test_ssim)

    # compute FID score
    fid = metric_test_fid.compute().item()
    metric_test_fid.reset()

    # compute FID score 2ND
    fid_2 = fid_score(real_path=target_dir, fake_path=sample_dir, device=device)

    # log metrics
    with open(log_file_path, "w") as f:
        f.write("test metrics\n")
        f.write(f"PSNR: {str(psnr)}\n")
        f.write(f"SSIM: {str(ssim)}\n")
        f.write(f"FID: {str(fid)}\n")
        f.write(f"FID_LACG: {str(fid_2)}\n")

    # # make saving dir
    # save_dir = os.path.join(
    #     args.output, f"log-psnr-{round(psnr, 2)}-ssim-{round(ssim, 2)}--fid-{round(fid, 2)}"
    # )
    # os.makedirs(save_dir, exist_ok=True)


if __name__ == "__main__":
    main()
