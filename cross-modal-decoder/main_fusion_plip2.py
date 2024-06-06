import argparse
import os
import sys
import json
from pathlib import Path
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from datasets import Mydataset_plip2
import vision_transformer as vits
from vision_transformer import DINOHead
from model_fusion_plip import MultiCropWrapper,fusionblock2
#from irene import MultiCropWrapper,fusionblock2
import utils
import math
import time
import datetime
import open_clip
from transformers import CLIPProcessor, CLIPModel
def get_args_parser():
    parser = argparse.ArgumentParser('multi-modality', add_help=False)
    parser.add_argument('--data_path', default='/home/wangzhenyuan/pathology/multi_modality/xian_all.csv', type=str,
                        help='Please specify path to the training data.')
    parser.add_argument('--checkpoint',default='checkpoint_1018_180.pth', type=str, help='prototype checkpoint path')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--saveckp_freq', default=50, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument("--lr", default=2e-05, type=float, help="""Learning rate at the end of
            linear warmup (highest LR used during training).""")
    parser.add_argument('--min_lr', type=float, default=1e-06, help="""Target LR at the
            end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument("--warmup_epochs", default=20, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--epochs', default=1000, type=int, help='Number of epochs of training.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help="""Initial value of the
            weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.1, help="""Final value of the
            weight decay. We use a cosine schedule for WD and using a larger decay by
            the end of training improves performance for ViTs.""")
    parser.add_argument('--output_dir', default="/home/wangzhenyuan/pathology/multi_modality/log_1000", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--noise_ratio', type=float, default=0.2, help="""training noise of image and text modality.""")
    parser.add_argument("--depth", default=1, type=int,
                        help="Number of transformer encoder.")
    parser.add_argument('--gate', default=True, type=utils.bool_flag,
                        help="Whether to use gated transformer encoder layer (Default: True)")
    parser.add_argument('--rand', default=True, type=utils.bool_flag,
                        help="Whether to random choice disease (Default: True)")
    return parser



def train_fusion(args):
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    # data
    data = Mydataset_plip2(root=args.data_path,randomm=args.rand)
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True
        )
    print(f"Data loaded: there are {len(data)} image-text-disease pairs.")

    # model
    model = vits.__dict__['vit_small'](patch_size=16, num_classes=0)
    state_dict = torch.load(args.checkpoint, map_location="cpu")  ### checkpoint
    head = DINOHead(384,
                    65536,
                    use_bn=True,
                    norm_last_layer=False,
                    predictor=True, )
    student = MultiCropWrapper(model, head)
    state_dict_new = state_dict["student"]
    state_dict_new = {k.replace("module.", ""): v for k, v in state_dict_new.items()}
    student.load_state_dict(state_dict_new, strict=True)
    student.eval()
    with torch.no_grad():
        w = student.prototypes.weight.data.clone()
        w = nn.functional.normalize(w, dim=1, p=2)
        student.prototypes.weight.copy_(w)
        prototypes = student.prototypes.weight
        p = torch.zeros(prototypes.shape[1])
        p = p.unsqueeze(0)
        prototype_all = torch.cat((prototypes, p), 0).cuda()


    disease_model = CLIPModel.from_pretrained("vinid/plip")
    disease_model.eval()

    model_fusion = fusionblock2(prototype_all=prototype_all, text_model=disease_model, disease_model=disease_model, depth=args.depth, noise_ratio=args.noise_ratio, gated=args.gate)
    model_fusion = model_fusion.cuda()

    # =================== freeze backbone ====================================
    for name, param in model_fusion.named_parameters():
        if name.startswith("text_model") or name.startswith("disease_model"):
            param.requires_grad = False
        else:
            continue

    params_groups = utils.get_params_groups(model_fusion)

    optimizer = torch.optim.AdamW(params_groups) # lr is set by scheduler

    lr_schedule = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )

    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )

    print("optimizer and schedulers ready.")

    print("Starting modality fusion training !")
    start_epoch = 0
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(epoch, data_loader, optimizer, lr_schedule, wd_schedule,model_fusion, args)

        save_dict = {
            'model_fusion': model_fusion.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }

        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))

        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(epoch,data_loader,optimizer,lr_schedule,wd_schedule,model_fusion,args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (tokens_disease,tokens_description,img,img_num) in enumerate(metric_logger.log_every(data_loader, 5, header)):
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        tokens_disease["input_ids"] = tokens_disease["input_ids"].squeeze(1).cuda(non_blocking=True)
        tokens_disease["attention_mask"] = tokens_disease["attention_mask"].squeeze(1).cuda(non_blocking=True)
        tokens_description["input_ids"] = tokens_description["input_ids"].squeeze(1).cuda(non_blocking=True)
        tokens_description["attention_mask"] = tokens_description["attention_mask"].squeeze(1).cuda(non_blocking=True)

        loss, loss_cosine, loss_clip, _, _,_ = model_fusion(img_p=img.cuda(non_blocking=True), text=tokens_description, img_num=img_num.cuda(non_blocking=True), disease=tokens_disease)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_cosine=loss_cosine.item())
        metric_logger.update(loss_clip=loss_clip.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}






if __name__ == '__main__':
    parser = argparse.ArgumentParser('multi-modality', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_fusion(args)
