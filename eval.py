import os
import numpy as np
import shutil
import torch
import torch.utils.data.distributed

from torch.utils.data import DataLoader

from evenerf.data_loaders import dataset_dict
from evenerf.render_image import render_single_image
from evenerf.model import EVENeRF
from evenerf.sample_ray import RaySamplerSingleImage
from utils import img_HWC2CHW, colorize, img2psnr, lpips, get_ssim, show_attention_map
import config
import torch.distributed as dist
from evenerf.projection import Projector
from evenerf.data_loaders.create_training_dataset import create_training_dataset
import imageio
import tqdm


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


@torch.no_grad()
def eval(args):

    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, "out", args.expname)
    print("outputs will be saved to {}".format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # save the args and config files
    f = os.path.join(out_folder, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, "config.txt")
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    if args.run_val == False:
        # create training dataset
        dataset, sampler = create_training_dataset(args)
        # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
        # please use distributed parallel on multiple GPUs to train multiple target views per batch
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            worker_init_fn=lambda _: np.random.seed(),
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=True if sampler is None else False,
        )
        iterator = iter(loader)
    else:
        # create validation dataset
        dataset = dataset_dict[args.eval_dataset](args, "validation", scenes=args.eval_scenes)
        loader = DataLoader(dataset, batch_size=1)
        iterator = iter(loader)

    # Create GNT model
    model = EVENeRF(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
    )
    
    # create projector
    projector = Projector(device=device)

    psnr_scores = []
    lpips_scores = []
    ssim_scores = []
    for indx, data in enumerate(tqdm.tqdm(loader, leave=True)):
        if args.local_rank == 0:
            tmp_ray_sampler = RaySamplerSingleImage(data, device, render_stride=args.render_stride)
            H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
            gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
            psnr_curr_img, lpips_curr_img, ssim_curr_img = log_view(
                indx,
                args,
                model,
                tmp_ray_sampler,
                projector,
                gt_img,
                render_stride=args.render_stride,
                prefix="val/" if args.run_val else "train/",
                out_folder=out_folder,
                vis_depth=args.vis_depth,
            )
            psnr_scores.append(psnr_curr_img)
            lpips_scores.append(lpips_curr_img)
            ssim_scores.append(ssim_curr_img)
            torch.cuda.empty_cache()

    print("Average PSNR: ", np.mean(psnr_scores))
    print("Average LPIPS: ", np.mean(lpips_scores))
    print("Average SSIM: ", np.mean(ssim_scores))

    txt_filename = os.path.join(out_folder, "result.txt")
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("Average PSNR: " + str(np.mean(psnr_scores)))
        f.write("\n")
        f.write("Average LPIPS: " + str(np.mean(lpips_scores)))
        f.write("\n")
        f.write("Average SSIM: " + str(np.mean(ssim_scores)))


@torch.no_grad()
def log_view(
    global_step,
    args,
    model,
    ray_sampler,
    projector,
    gt_img,
    render_stride=1,
    prefix="",
    out_folder="",
    vis_depth=False,
):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()
        if model.feature_net is not None:
            featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
        else:
            featmaps = [None, None]
        ret = render_single_image(
            ray_sampler=ray_sampler,
            ray_batch=ray_batch,
            model=model,
            projector=projector,
            chunk_size=args.chunk_size,
            N_samples=args.N_samples,
            inv_uniform=args.inv_uniform,
            det=True,
            white_bkgd=args.white_bkgd,
            render_stride=render_stride,
            featmaps=featmaps,
        )

    average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))

    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        average_im = average_im[::render_stride, ::render_stride]

    # rgb_gt = img_HWC2CHW(gt_img)
    average_im = img_HWC2CHW(average_im)

    rgb_pred = ret["outputs_render"]["rgb"].detach().cpu()
    rgb_show = img_HWC2CHW(rgb_pred)  

    rgb_show = rgb_show.permute(1, 2, 0).numpy()
    filename = os.path.join(out_folder, prefix[:-1] + "_{:03d}_render.png".format(global_step))
    imageio.imwrite(filename, rgb_show)

    if vis_depth:
        depth_pred = ret["outputs_render"]["depth"].detach().cpu()
        depth_show = img_HWC2CHW(colorize(depth_pred, cmap_name="jet"))
        depth_show = depth_show.permute(1, 2, 0).detach().cpu().numpy()
        filename = os.path.join(
            out_folder, prefix[:-1] + "_{:03d}_depth.png".format(global_step)
        )
        imageio.imwrite(filename, depth_show)

    # write scalar
    pred_rgb = torch.clip(rgb_pred, 0.0, 1.0)
    lpips_curr_img = lpips(pred_rgb*2.-1., gt_img*2.-1., format="HWC").item()
    ssim_curr_img = get_ssim(pred_rgb, gt_img)
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    print(prefix + "psnr_image: ", psnr_curr_img)
    print(prefix + "lpips_image: ", lpips_curr_img)
    print(prefix + "ssim_image: ", ssim_curr_img)
    return psnr_curr_img, lpips_curr_img, ssim_curr_img


if __name__ == "__main__":
    parser = config.config_parser()
    parser.add_argument("--run_val", action="store_true", help="run on val set")
    args = parser.parse_args()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    eval(args)
