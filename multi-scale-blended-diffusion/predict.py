import os
import sys
import subprocess

from argparse import Namespace
from PIL import Image

from cog import BasePredictor, Path, Input

sys.path.append(os.getcwd())
from msbd.MSBDGenerator import MSBDGenerator

args = Namespace(blended_upscale=True,
                 clip_reranking=True,
                 conditional_upscale=True,
                 ddim_steps=50,
                 decoder_optimization=True,
                 decoderopt_it=100,
                 dilate_mask=False,
                 first_stage_batch=5,
                 first_stage_size=512,
                 fp16=True,
                 grid_overlap=128,
                 grid_startstep=0.25,
                 lowpass_reference='matching',
                 max_edgelen=768,
                 outdir='outputs',
                 repaint_jump=0,
                 repaint_steps=5,
                 seed=-1,
                 start_timestep=1.0,
                 straight_to_grid=False,
                 upscale_startstep=0.4,
                 upscaling_mode='esrgan',
                 use_stablediffusion=True)


class Predictor(BasePredictor):
    def setup(self):
        subprocess.run(["mkdir", "$HOME/.cache/clip"])
        subprocess.run(["mv", "ViT-L-14.pt", "$HOME/.cache/clip"])
        self.generator = MSBDGenerator(
            use_fp16=True,
            stable_diffusion=True,
            max_edgelen=args.max_edgelen,
            first_stage_batchsize=args.first_stage_batch
        )

    def predict(
            self,
            input_image: Path = Input(description="image to edit"),
            input_mask: Path = Input(description="Mask indicating edit area"),
            edit_prompt: str = Input(description="Prompt for generation inside the editing area"),
            margin_mult: float = Input(description="", default=1.4),

    ) -> Path:
        input_image = str(input_image)
        input_mask = str(input_mask)
        edit_prompt = str(edit_prompt)
        margin_mult = float(margin_mult)
        result = self.generator.multi_scale_generation(
            pil_img=Image.open(input_image).convert('RGB'),
            pil_mask=Image.open(input_mask).convert("L"),
            prompt=edit_prompt,
            ddim_steps=50,
            decoder_optimization=args.decoder_optimization,
            clip_reranking=args.clip_reranking,
            margin=margin_mult,
            seed=args.seed,
            repaint_steps=args.repaint_steps,
            start_timestep=args.start_timestep,
            upscaling_start_step=args.upscale_startstep,
            upscaling_mode=args.upscaling_mode,
            straight_to_grid=args.straight_to_grid,
            grid_upscaling_start_step=args.grid_startstep,
            log_folder=args.outdir,
            lowpass_reference=args.lowpass_reference,
            blended_upscale=args.blended_upscale,
            conditional_upscale=args.conditional_upscale,
            grid_overlap=args.grid_overlap,
            first_stage_size=args.first_stage_size
        )

        out_path = 'output.png'
        result.save(out_path)
        return Path(out_path)
