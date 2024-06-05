"""
Train a super-resolution model.
"""
import sys
sys.path.append("/home/wangzhenyuan/diffu_path/guided-diffusion-main/")

import argparse
import torch
import torch.nn.functional as F
from guided_diffusion.get_ssl_models import get_model
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir="./log_supp_220/")
    # Load SSL model
    if args.feat_cond:
        ssl_model = get_model(args.type_model, args.use_head).to(
            dist_util.dev()).eval()  ### 这里改一下###########################################
        ssl_dim = ssl_model(torch.zeros(1, 3, 224, 224).to(dist_util.dev())).size(1)
        print("SSL DIM:", ssl_dim)
        for _, p in ssl_model.named_parameters():
            p.requires_grad_(False)
    else:
        ssl_model = None
        ssl_dim = 2048
        print("No SSL models")

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys()), ssl_dim=ssl_dim
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    # data = load_superres_data(
    #     args.data_dir,
    #     args.batch_size,
    #     large_size=args.large_size,
    #     small_size=args.small_size,
    #     class_cond=args.class_cond,
    # )
    data = load_superres_data(
        args,
        ssl_model=ssl_model,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def load_superres_data(args, ssl_model=None):
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.large_size,
        class_cond=args.class_cond,
    )
    # for large_batch, model_kwargs in data:
    #     model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
    #     yield large_batch, model_kwargs
    for batch, batch_big, model_kwargs in data:
        model_kwargs["low_res"] = F.interpolate(batch, args.small_size, mode="area")
        # We add the conditioning in conditional mode
        if ssl_model is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(args.use_fp16):
                    # we always use an image of size 224x224 for conditioning
                    model_kwargs["feat"] = ssl_model(batch_big.to(dist_util.dev())).detach()
            yield batch, model_kwargs
        else:
            yield batch, model_kwargs


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        type_model="my_model",
        use_head=False
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
