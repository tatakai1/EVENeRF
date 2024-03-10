import torch.nn as nn
import torch
from utils import img2mse

class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch, scalars_to_log):
        """
        training criterion
        """
        pred_rgb = outputs["rgb"]
        if "mask" in outputs:
            pred_mask = outputs["mask"].float()
        else:
            pred_mask = None
        gt_rgb = ray_batch["rgb"]
        rgb_loss = img2mse(pred_rgb, gt_rgb, pred_mask)
  
        loss = rgb_loss + 0

        return loss, rgb_loss, scalars_to_log
