"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import sys
sys.path.append("/home/wangzhenyuan/diffu_path/guided-diffusion-main/")
import argparse
import os
from torchvision.utils import make_grid, save_image
import numpy as np
import torch
import torch.distributed as dist
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.get_ssl_models import get_model
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure("./sample_small/")
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

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()), ssl_dim=ssl_dim
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu"),strict=False
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("Load data...")
    # data = load_data(
    #     data_dir=args.data_dir,
    #     batch_size=1,
    #     image_size=args.image_size,
    #     class_cond=args.class_cond,
    #     deterministic=True,
    #     random_flip=False,
    # )

    index = np.load("/home/wangzhenyuan/pathology/multi_modality/prototype_index.npy")
    prototype = torch.tensor(np.load("/home/wangzhenyuan/pathology/multi_modality/prototype_220.npy"))
    for ind in index:
        logger.log("sampling...")

        # all_images = []
        all_gen = []
        # all_feats = []
        sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
        num_current_samples = 0
        while num_current_samples < args.num_samples:
            #batch_small, batch, cond = next(data)
            #batch = batch[0:1].repeat(args.batch_size, 1, 1, 1).to(dist_util.dev())
            model_kwargs = {}
            with torch.no_grad():
                if ssl_model is not None:
                    #feat = ssl_model(batch).detach()
                    model_kwargs["feat"] = prototype[ind].repeat(args.batch_size, 1).to(dist_util.dev())
                    #all_feats.extend([featt.unsqueeze(0).cpu().numpy() for featt in feat])
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            #batch = ((batch_small[0:1] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            #batch = batch.permute(0, 2, 3, 1)
            #batch = batch.contiguous()
            #all_images.extend([sample.unsqueeze(0).cpu().numpy() for sample in batch])

            sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            sample = sample.permute(0, 2, 3, 1)
            samples = sample.contiguous()

            #all_images.extend([sample.unsqueeze(0).cpu().numpy() for sample in samples])
            all_gen.extend([sample.unsqueeze(0).cpu().numpy() for sample in samples])
            logger.log(f"created {len(all_gen) } samples")
            num_current_samples += 1
        # if ssl_model is not None:
        #     feat_all = np.concatenate(all_feats, axis=0)
        #     np.savez(args.out_dir + '/' + args.name + "_feature" + '.npz', feat_all)

        arr = np.concatenate(all_gen, axis=0)
        #arr_all = np.concatenate(all_images, axis=0)
        #print("feat_all:",feat_all.shape)
        #print("arr_all:",arr.shape)
        np.savez(args.out_dir+'/'+args.name+str(ind)+"_data"+'.npz', arr)
        #save_image(torch.FloatTensor(arr_all).permute(0,3,1,2), args.out_dir+'/'+args.name+str(dist.get_rank())+'.jpeg', normalize=True, scale_each=True, nrow=args.batch_size+1)
        save_image(torch.FloatTensor(arr).permute(0, 3, 1, 2),
                   args.out_dir + '/' + args.name+ str(ind) + '.jpeg', normalize=True, scale_each=True,
                   nrow=args.batch_size)

    logger.log("sampling complete")



    # all_images = []
    # all_labels = []
    # while len(all_images) * args.batch_size < args.num_samples:
    #     model_kwargs = {}
    #     if args.class_cond:
    #         classes = th.randint(
    #             low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
    #         )
    #         model_kwargs["y"] = classes
    #     sample_fn = (
    #         diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    #     )
    #     sample = sample_fn(
    #         model,
    #         (args.batch_size, 3, args.image_size, args.image_size),
    #         clip_denoised=args.clip_denoised,
    #         model_kwargs=model_kwargs,
    #     )
    #     sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    #     sample = sample.permute(0, 2, 3, 1)
    #     sample = sample.contiguous()
    #
    #     gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
    #     dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
    #     all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
    #     if args.class_cond:
    #         gathered_labels = [
    #             th.zeros_like(classes) for _ in range(dist.get_world_size())
    #         ]
    #         dist.all_gather(gathered_labels, classes)
    #         all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
    #     logger.log(f"created {len(all_images) * args.batch_size} samples")
    #
    # arr = np.concatenate(all_images, axis=0)
    # arr = arr[: args.num_samples]
    # if args.class_cond:
    #     label_arr = np.concatenate(all_labels, axis=0)
    #     label_arr = label_arr[: args.num_samples]
    # if dist.get_rank() == 0:
    #     shape_str = "x".join([str(x) for x in arr.shape])
    #     out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    #     logger.log(f"saving to {out_path}")
    #     if args.class_cond:
    #         np.savez(out_path, arr, label_arr)
    #     else:
    #         np.savez(out_path, arr)
    #
    # dist.barrier()
    # logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="1.txt",
        clip_denoised=True,
        num_samples=12,
        batch_size=16,
        use_ddim=False,
        type_model="my_model",
        use_head=False,
        model_path="/home/wangzhenyuan/diffu_path/guided-diffusion-main/log_220_p/ema_0.9999_400000.pt",
        out_dir="/home/wangzhenyuan/diffu_path/guided-diffusion-main/sample_small",
        name="sample_prototype"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
