"""
Some utils for blended latent diffusion.

I just wanted to keep them separate from the utils.py in ldm and such.

@author Johannes Ackermann
"""

import logging
import os
from typing import Tuple

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torchvision.utils import make_grid
import kornia

logger = logging.getLogger()


def get_dilated_mask(
    mask: torch.Tensor,
    dilation_kernel_size: int,
) -> torch.Tensor:
    """
    Dilates (extends) the mask by convultion with a kernel of size `dilation_kernel_size`x`dilation_kernel_size`.
    If applied to the downsampled mask (as is done in the paper), this should be done with sizes, 7->5->3->1 for
    equal parts of the diffusion process. (of course, dilation with size 1 leaves the mask unchanged)
    """
    kernel = torch.ones([dilation_kernel_size, dilation_kernel_size],
                        dtype=mask.dtype).to(mask.device)
    if len(mask.shape) == 2:
        mask = mask[None][None]
    dilated_mask = torch.nn.functional.conv2d(mask, kernel[None][None], padding='same')
    dilated_mask[dilated_mask >= 1.0] = 1.0
    return dilated_mask


def get_repaint_schedule(original_schedule: int, repaint_steps: int, repaint_jump: int):
    """
    Generates a schedule as in repaint, that takes a given denoising schedule and repeats
    it in such a way that each set of `repaint_jump` denoising steps is repeated `repaint_steps` times.
    example if we had 10 steps in total, repaint_steps = 1 and repaint_jump = 2:
    10-9-8-10-9-8-7-6-5-7-6-5-4-3-5-4-3-5-4-3...

    This code became pretty ugly, but essentially it's just zigzagging through `original_schedule`.
    """
    # i thought this could be done with repeat(reshape()), but not quite :'(

    if repaint_jump == 0:
        schedule = np.repeat(original_schedule, repaint_steps + 1)
    else:
        n_step_orig = len(original_schedule)
        schedule = [original_schedule[:repaint_jump - 1]]
        for idx_jump_level in range(1, n_step_orig // repaint_jump):
            for idx_rep in range(repaint_steps):
                if idx_rep == repaint_steps - 1:
                    schedule.append(
                        original_schedule[idx_jump_level * (repaint_jump) - 1:(idx_jump_level + 1) * (repaint_jump) - 1])
                else:
                    schedule.append(
                        original_schedule[idx_jump_level * (repaint_jump) - 1:(idx_jump_level + 1) * (repaint_jump)])
        schedule.append(original_schedule[repaint_jump * (n_step_orig // repaint_jump) - 1:])
        schedule = np.concatenate(schedule)
    return schedule


def tensor_to_pil(tensor: torch.Tensor):
    # if not tensor.min() < 0.0:
    #     logger.warn('Image should be scaled to [-1.0, 1.0]')
    tensor = tensor.cpu().numpy()[0]
    tensor = (tensor + 1.0) / 2.0
    tensor = 255. * rearrange(tensor, 'c h w -> h w c')
    return Image.fromarray(tensor.astype(np.uint8))


def get_alpha_masks(crops_x: Tuple[int], crops_y: Tuple[int], target_imagesize: Tuple[int], overlap: int):
    """
    Generates the alpha masks later used for blending.
    """
    alpha_mask_full = np.zeros([len(crops_x)] + list(target_imagesize))
    for idx, (crop_x, crop_y) in enumerate(zip(crops_x, crops_y)):
        alpha_mask_full[idx, crop_x[0]:crop_x[1], crop_y[0]:crop_y[1]] = 1.0
        alpha_func = 1.0 / (1 + np.exp(-np.linspace(-5.0, 5.0, overlap // 2)))  # sigmoid blending
        alpha_mask_x = np.tile(alpha_func, [crop_y[1] - crop_y[0], 1]).T
        alpha_mask_y = np.tile(alpha_func, [crop_x[1] - crop_x[0], 1])
        if not crop_x[0] == 0:
            alpha_mask_full[idx, crop_x[0]:crop_x[0] + overlap // 2, crop_y[0]:crop_y[1]
                            ] = alpha_mask_full[idx, crop_x[0]:crop_x[0] + overlap // 2, crop_y[0]:crop_y[1]] * alpha_mask_x
        if not crop_y[0] == 0:
            alpha_mask_full[idx, crop_x[0]:crop_x[1], crop_y[0]:crop_y[0] + overlap //
                            2] = alpha_mask_full[idx, crop_x[0]:crop_x[1], crop_y[0]:crop_y[0] + overlap // 2] * alpha_mask_y
        if not crop_x[1] == target_imagesize[0]:
            alpha_mask_full[idx, crop_x[1] - overlap // 2:crop_x[1], crop_y[0]:crop_y[1]] = alpha_mask_full[idx,
                                                                                                            crop_x[1] - overlap // 2:crop_x[1], crop_y[0]:crop_y[1]] * alpha_mask_x[::-1, :]
        if not crop_y[1] == target_imagesize[1]:
            alpha_mask_full[idx, crop_x[0]:crop_x[1], crop_y[1] - overlap // 2:crop_y[1]] = alpha_mask_full[idx,
                                                                                                            crop_x[0]:crop_x[1], crop_y[1]-overlap // 2:crop_y[1]] * alpha_mask_y[:, ::-1]

    alpha_mask_seg = []
    for idx, (crop_x, crop_y) in enumerate(zip(crops_x, crops_y)):
        alpha_mask_seg.append(alpha_mask_full[idx, crop_x[0]:crop_x[1], crop_y[0]:crop_y[1]])

    return alpha_mask_seg


def get_result_grid(source_img, all_samples, mask) -> Image:
    """
    Saves a grid of images visualizing the original image, the mask and multiple samples
    """
    mask = mask.cpu()
    h = all_samples[0].shape[2]
    w = all_samples[0].shape[3]
    source_img_mask_vis = (0.8 * source_img.clone().cpu() + 1.0) / 2.0
    source_img_mask_vis[0, 1] += mask[0, 0] * 0.2
    source_img_mask_vis = source_img_mask_vis.clamp(-1.0, 1.0)
    source_img = torch.nn.functional.interpolate(source_img, size=[h, w])
    source_img_mask_vis = torch.nn.functional.interpolate(source_img_mask_vis, size=[h, w])
    # additionally, save as grid
    vis_rows = []
    for sample_row in all_samples:
        vis_rows.append(torch.cat([(source_img.cpu() + 1.0) / 2.0,
                        source_img_mask_vis, all_samples[0].cpu()], 0))

    grid = torch.stack(vis_rows, 0)

    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=(len(all_samples[0]) + 2))

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()

    pil_result = Image.fromarray(grid.astype(np.uint8))
    return pil_result


def debug_save_img(image: torch.Tensor, name='debug'):
    assert len(image.shape) == 4
    # if not image.min() < 0.0:
    #     logger.warn('Image should be scaled to [-1.0, 1.0]')
    image = torch.clamp((image[0] + 1.0) / 2.0, min=0.0, max=1.0)
    image = 255.0 * rearrange(image.cpu().numpy(), 'c h w -> h w c')
    Image.fromarray(image.astype(np.uint8)).save(f'{name}.png')  # debugging
    logger.info(f'saved debug output to {name}.png')


def is_notebook() -> bool:
    """
    Based on 
    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def display_or_save(pil_img: Image, folder: str, name: str) -> None:
    """
    If running in a jupyter notebook uses `display` to show the image, if not running in a jupyter notebook
    saves the image to the given folder with the given name.
    """
    if is_notebook():
        from IPython.display import display
        print(name)
        display(pil_img)
    else:
        if folder is not None:
            fp = os.path.join(folder, name + '.png')
            pil_img.save(fp)
            logger.info(f'saved image to {fp}')

def sharpen(image: torch.Tensor, unsharpen_masking: bool = True, kernel_size: int = 11, sigma = 7.0) -> torch.Tensor:
    if not unsharpen_masking:
        raise NotImplementedError('Only unsharp masking is supported currently.')
    else: 
        logger.info(f'sharpening with sigma {sigma}')
        return kornia.filters.unsharp_mask(image, (kernel_size,kernel_size), (sigma,sigma)).clamp(-1.0,1.0)

