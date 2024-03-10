import torch
from collections import OrderedDict
from evenerf.render_ray import render_rays


def render_single_image(
    ray_sampler,
    ray_batch,
    model,
    projector,
    chunk_size,
    N_samples,
    inv_uniform=False,
    det=False,
    white_bkgd=False,
    render_stride=1,
    featmaps=None,
):
    """
    :param ray_sampler: RaySamplingSingleImage for this view
    :param model:  {'entangled_net': , ...}
    :param chunk_size: number of rays in a chunk
    :param N_samples: samples along each ray 
    :param inv_uniform: if True, uniformly sample inverse depth 
    :return: {'outputs_render': {'rgb': numpy, 'depth': numpy, ...}}
    """

    all_ret = OrderedDict([("outputs_render", OrderedDict())])

    N_rays = ray_batch["ray_o"].shape[0]

    for i in range(0, N_rays, chunk_size):
        chunk = OrderedDict()
        for k in ray_batch:
            if k in ["camera", "depth_range", "src_rgbs", "src_cameras"]:
                chunk[k] = ray_batch[k]
            elif ray_batch[k] is not None:
                chunk[k] = ray_batch[k][i : i + chunk_size]
            else:
                chunk[k] = None

        ret = render_rays(
            chunk,
            model,
            featmaps,
            projector=projector,
            N_samples=N_samples,
            inv_uniform=inv_uniform,
            det=det,
            white_bkgd=white_bkgd,
        )

        # cache chunk results on cpu
        if i == 0:
            for k in ret["outputs_render"]:
                if ret["outputs_render"][k] is not None:
                    all_ret["outputs_render"][k] = []

        for k in ret["outputs_render"]:
            if ret["outputs_render"][k] is not None:
                all_ret["outputs_render"][k].append(ret["outputs_render"][k].cpu())

    rgb_strided = torch.ones(ray_sampler.H, ray_sampler.W, 3)[::render_stride, ::render_stride, :]
    # merge chunk results and reshape
    for k in all_ret["outputs_render"]:
        if k == "random_sigma":
            continue
        tmp = torch.cat(all_ret["outputs_render"][k], dim=0).reshape(
            (rgb_strided.shape[0], rgb_strided.shape[1], -1)
        )
        all_ret["outputs_render"][k] = tmp.squeeze()

    return all_ret
