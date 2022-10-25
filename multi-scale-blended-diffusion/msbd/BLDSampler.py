"""
This module is based on the original latent diffusion models code base, specifically on `ldm/models/diffusion/ddim.py`.
It mainly just implements the sampler for blended latent diffusion by changing the sampling in `blended_ddim_sampling` to incorporate the mask and
replace the unmasked area by a noisy version of the original image after each timestep.
It also implements SDEdit and repaint by starting at an intermediate timestep and repeating diffusion steps multiple times.
"""
import logging

import numpy as np
import torch
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from PIL import Image
from tqdm.auto import tqdm

from msbd.msbd_utils import get_dilated_mask, get_repaint_schedule

logger = logging.getLogger()


class BlendedDiffusionSampler(DDIMSampler):
    def __init__(self, model: LatentDiffusion, schedule="linear"):  # this just adds the typehint for easier coding
        super().__init__(model, schedule)

    @torch.no_grad()
    def blended_diffusion_sampling(
        self,
        source_img,  # LATENT, not pixels
        mask,
        num_ddim_steps,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.,
        x0=None,
        temperature=1.,
        noise_dropout=0.,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.,
        unconditional_conditioning=None,
        dilate_mask=False,
        repaint_steps=4,
        repaint_jump=1,
        start_timestep=1.0,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(
                        f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=num_ddim_steps, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.blended_ddim_sampling(
            conditioning, size,
            source_img=source_img,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask, x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            dilate_mask=dilate_mask,
            repaint_steps=repaint_steps,
            repaint_jump=repaint_jump,
            start_timestep=start_timestep
        )
        return samples, intermediates

    @torch.no_grad()
    def decode_and_save_latent(self, latent: torch.Tensor, fp: str):
        """
        Function for debugging to decode and save any input image latent tensor.
        """
        if len(latent.shape) == 3:
            latent = latent[None]
        if latent.device != self.model.device:
            latent = latent.to(self.model.device)
        decoded = self.model.decode_first_stage(latent)[0]

        decoded = torch.clamp((decoded+1.0)/2.0, min=0.0, max=1.0).cpu().numpy()
        decoded = 255. * rearrange(decoded, 'c h w -> h w c')
        Image.fromarray(decoded.astype(np.uint8)).save(fp)
        logging.info(f'Saved sample to {fp}')

    def q_sample_start_end(self, x_t_start: torch.Tensor, t_end: int, t_start: int = 0, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Samples the forward diffusion process from any starting point in the diffusion process, i.e. x_t_end ~ q(x_t_end|x_t_start).

        Samples from the forward process as defined in DDPM, but should be fine to work with DDIM sampling.
        """
        if noise is None:
            noise = torch.randn_like(x_t_start)

        alpha_cumprod_start_end = self.model.alphas_cumprod[t_end] / \
            self.model.alphas_cumprod[t_start]
        return torch.sqrt(alpha_cumprod_start_end) * x_t_start + torch.sqrt(1 - alpha_cumprod_start_end) * noise

    @torch.no_grad()
    def blended_ddim_sampling(
            self, cond, shape, source_img,
            x_T=None, ddim_use_original_steps=False,
            callback=None, timesteps=None, quantize_denoised=False,
            mask=None, x0=None, img_callback=None, log_every_t=100,
            temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
            unconditional_guidance_scale=1., unconditional_conditioning=None,
            dilate_mask=False, repaint_steps=0, repaint_jump=1, start_timestep=1.0):
        """
        `source_img` is the image that we wil edit, such that the content of `mask` matches `cond`.
        This is done by starting at gaussian noise and then doing the following iteratively:

        1. do one denoising step on img_t
        2. edit_img = denoise(img_t, cond, t)
        3. noised_source_img = forward_process(source_img, t)
        4. img_{t-1} = mask * edit_img + (1 - mask) * noised_source_img
        5. goto 2 till t=0

        `start_timestep` is the ratio of where to start in the diffusion process as in SDEdit.
        1.0 means do the full diffusion from gaussian noise
        """
        device = self.model.betas.device
        b = shape[0]

        mask = mask.to(device)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(
                min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        if x_T is not None:
            edit_img = x_T
            logger.warn('starting diffusion from preset x_T, ignoring sdedit')
        else:
            if 0.0 < start_timestep < 1.0 and int(len(self.ddim_timesteps) * start_timestep):  # second condition ensures at least one timestep
                #SDEdit
                if source_img.shape[0] == 1:
                    # if a single image is given this function will make b different samples
                    source_img = source_img.repeat(b, 1, 1, 1)

                timesteps = timesteps[:int(len(timesteps) * start_timestep)]
                logger.info(f'Using SDEdit ratio {start_timestep}, starting at {timesteps[-1]}/1000')
                edit_img = self.q_sample_start_end(source_img, t_end=timesteps[-1], t_start=0)
            elif start_timestep == 1.0:
                edit_img = torch.randn(shape, device=device)
            elif start_timestep == 0.0 or int(len(self.ddim_timesteps) * start_timestep):
                logger.warn('Start timestep is 0.0, or rounded down to zero, returning original image')
                return source_img, [source_img]
              

        intermediates = {'x_inter': [edit_img], 'pred_x0': [edit_img]}
        time_range = reversed(
            range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)

        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(
            f"Running blended DDIM Sampling with {total_steps} timesteps with {repaint_steps} repaint steps and {repaint_jump} repaint jumps")

        if repaint_steps:
            # overwrites timerange with a zigzag line
            index_schedule = get_repaint_schedule(
                np.arange(len(time_range) - 1, -1, -1), repaint_steps, repaint_jump)
            time_range = get_repaint_schedule(time_range, repaint_steps, repaint_jump)
            logger.info(f'repaint schedule: {time_range}', )
        else:
            index_schedule = np.arange(len(time_range) - 1, -1, -1)

        for i, step in enumerate(tqdm(time_range, desc='DDIM Sampler', total=len(time_range))):
            index = index_schedule[i]
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # ts is the timestep in the original 1000 step DDPM model
            # index is the timestep for the DDIM sampler using only as many timesteps as specified
            edit_img, pred_x0 = self.p_sample_ddim(edit_img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                                   quantize_denoised=quantize_denoised, temperature=temperature,
                                                   noise_dropout=noise_dropout, score_corrector=score_corrector,
                                                   corrector_kwargs=corrector_kwargs,
                                                   unconditional_guidance_scale=unconditional_guidance_scale,
                                                   unconditional_conditioning=unconditional_conditioning)

            source_img_noised = self.model.q_sample(source_img, ts)
            # maybe this should not be ts, but the next step?

            mask_expanded = self.get_mask(mask, dilate_mask, shape, total_steps, i)
            edit_img = (1 - mask_expanded) * source_img_noised + mask_expanded * edit_img

            if repaint_steps and i > 0 and i < len(time_range) - 1:
                # if doing repaint, jump back if the next element in the schedule is noisier than the current one
                # this is the noise level/timestep AFTER the most recent denoising step
                t_current = timesteps[index_schedule[i] - 1]
                # this is the noise level/timestep we need for the next denoising step
                t_next_step = timesteps[index_schedule[i + 1]]
                if t_next_step > t_current:
                    edit_img = self.q_sample_start_end(
                        edit_img, t_end=t_next_step, t_start=t_current)

            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            # logger.warn('doing a lot of logging of intermediates!')
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(edit_img)
                intermediates['pred_x0'].append(pred_x0)

        return edit_img, intermediates

    def get_mask(self, mask, dilate_mask, shape, total_steps, current_step):
        """Implements mask dilation, extending the mask at the start of the diffusion process but slowly shrinking it to the original size.
        Not really used for our multi-stage blended diffusion.
        """
        if dilate_mask:
            sampling_progress_ratio = current_step / total_steps
            if sampling_progress_ratio < 0.25:
                kernel_size = 7
            elif sampling_progress_ratio < 0.5:
                kernel_size = 5
            elif sampling_progress_ratio < 0.75:
                kernel_size = 3
            else:
                kernel_size = 1

            mask_dilated = get_dilated_mask(mask, kernel_size)
            mask_expanded = mask_dilated.expand(*shape)
        else:
            mask_expanded = mask.expand(*shape)
        return mask_expanded
