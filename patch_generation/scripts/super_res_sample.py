"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""
import sys
sys.path.append("/home/wangzhenyuan/diffu_path/guided-diffusion-main/")
import argparse
import os
from torchvision.utils import make_grid, save_image
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure("./sample_big/")

    logger.log("creating model...")
    if args.feat_cond:
        ssl_dim =384
    else:
        ssl_dim = 2048
        print("No SSL models")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys()), ssl_dim=ssl_dim
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    data,feature = load_data_for_worker(args.base_samples,args.base_features, args.batch_size, args.class_cond,args.feat_cond)

    sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)

    logger.log("creating samples...")
    num_current_samples = 0
    all_images = []
    while num_current_samples < args.num_samples:
        model_kwargs = {}
        model_kwargs["low_res"] = next(data).to(dist_util.dev())
        print(model_kwargs["low_res"].shape)
        if args.feat_cond:
            model_kwargs["feat"] = next(feature).to(dist_util.dev())
            print(model_kwargs["feat"].shape)
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.large_size, args.large_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        samples = sample.contiguous()
        all_images.extend([sample.unsqueeze(0).cpu().numpy() for sample in samples])
        logger.log(f"created {len(all_images)} samples")
        num_current_samples += 1
    arr = np.concatenate(all_images, axis=0)
    np.savez(args.out_dir + '/' + args.name + "_databig" + '.npz', arr)
    save_image(th.FloatTensor(arr).permute(0, 3, 1, 2),
               args.out_dir + '/' + args.name + "big" + str(dist.get_rank()) + '.jpeg', normalize=True, scale_each=True,
               nrow=args.batch_size)
    logger.log("sampling complete")
    # while len(all_images) * args.batch_size < args.num_samples:
    #     model_kwargs = next(data)
    #     model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
    #     sample = diffusion.p_sample_loop(
    #         model,
    #         (args.batch_size, 3, args.large_size, args.large_size),
    #         clip_denoised=args.clip_denoised,
    #         model_kwargs=model_kwargs,
    #     )
    #     sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    #     sample = sample.permute(0, 2, 3, 1)
    #     sample = sample.contiguous()
    #
    #     all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
    #     dist.all_gather(all_samples, sample)  # gather not supported with NCCL
    #     for sample in all_samples:
    #         all_images.append(sample.cpu().numpy())
    #     logger.log(f"created {len(all_images) * args.batch_size} samples")
    #
    # arr = np.concatenate(all_images, axis=0)
    # arr = arr[: args.num_samples]
    # if dist.get_rank() == 0:
    #     shape_str = "x".join([str(x) for x in arr.shape])
    #     out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    #     logger.log(f"saving to {out_path}")
    #     np.savez(out_path, arr)
    #
    # dist.barrier()
    # logger.log("sampling complete")


def load_data_for_worker(base_samples,base_features, batch_size, class_cond,feat_cond):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if class_cond:
            label_arr = obj["arr_1"]
    if feat_cond:
        with bf.BlobFile(base_features, "rb") as f:
            obj_ = np.load(f)
            feature_arr = obj_["arr_0"]

    #rank = dist.get_rank()
    #num_ranks = dist.get_world_size()
    buffer = []
    feature_buffer = []
    image_arr = th.from_numpy(image_arr).float()
    if feat_cond:
        feature_arr = th.from_numpy(feature_arr).float()
    image_arr = image_arr / 127.5 - 1.0
    image_arr = image_arr.permute(0, 3, 1, 2)
    for i in range(0, len(image_arr), batch_size):
        buffer.append(image_arr[i:(i + batch_size)])
        if feat_cond:
            feature_buffer.append(feature_arr[i:(i + batch_size)])
    buffer = iter(buffer)
    feature_buffer = iter(feature_buffer)

    return buffer, feature_buffer



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=8,
        batch_size=8,
        use_ddim=False,
        base_samples="/home/wangzhenyuan/diffu_path/guided-diffusion-main/sample_small/sample_data.npz",
        base_features=None,
        model_path="/home/wangzhenyuan/diffu_path/guided-diffusion-main/log_supp/ema_0.9999_460000.pt",
        out_dir="/home/wangzhenyuan/diffu_path/guided-diffusion-main/sample_big",
        name="sample"
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
