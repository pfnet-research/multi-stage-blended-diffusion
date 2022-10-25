"""
This module can be used to batch process the files in a given input folder, along with a txt file containing the prompts and filenames.
See inputs/inputs.txt for an example of the samples we use in our publication.

For interactive use we recommend to use the notebook `InteractiveEditing.ipynb`.

@author: Johannes Ackermann
"""

import argparse
import sys
import os
import logging

from PIL import Image

sys.path.append(os.getcwd())
from msbd.MSBDGenerator import MSBDGenerator

logger = logging.getLogger()
logging.getLogger().setLevel(logging.INFO)


def main(args):

    if args.input_list is not None and not args.input_list == 'None':
        prompts = []
        image_fps = []
        margin_mults = []
        with open(args.input_list, 'r') as f:
            lines = f.readlines()
            print('Reading from ', args.input_list)
        for line in lines:
            line = line.strip()  # remove newline
            if line.startswith('#'):
                print('skipping line ', line)
                continue
            prompts.append(line.split(';')[0])
            margin_mults.append(float((line.split(';')[2])))
            image_fp = os.path.join(
                os.path.dirname(args.input_list),
                line.split(';')[1].replace(' ', '')
            )
            assert os.path.exists(image_fp), f'could not find image {image_fp}'
            mask_fp = os.path.splitext(image_fp)[0] + '_mask.png'
            assert os.path.exists(mask_fp), f'could not find mask {mask_fp}'
            image_fps.append(image_fp)
            print(image_fps[-1], prompts[-1], margin_mults[-1])
    else:
        image_fps = ['inputs/marunouchi.png']
        assert os.path.exists(image_fps[0])
        prompts = ['Statue of Roman Emperor, Canon 5D Mark 3, 35mm, flickr']
        margin_mults = [1.2]

    generator = MSBDGenerator(
        use_fp16=args.fp16,
        stable_diffusion=True,
        max_edgelen=args.max_edgelen,
        first_stage_batchsize=args.first_stage_batch
    )

    for prompt, image_fp, margin_mult in zip(prompts, image_fps, margin_mults):
        result = generator.multi_scale_generation(
            pil_img = Image.open(image_fp).convert('RGB'),
            pil_mask = Image.open(os.path.splitext(image_fp)[0] + '_mask.png'), 
            prompt=prompt,
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
        out_fp = os.path.splitext(image_fp)[0] + '_output.jpg'
        result.save(out_fp)
        print(f'Output saved to{out_fp}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-list", type=str, nargs="?",
                        default='inputs/inputs.txt', help="path to a list of prompts and file_paths ")
    def str2bool(v):
        # from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser.add_argument("--prompt", type=str, nargs="?",
                        default='Oil painting of Mt. Fuji, by Paul Sandby', help="the prompt to render")
    parser.add_argument("--outdir", type=str, nargs="?",
                        help="dir to write results to", default="outputs")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="number of ddim sampling steps",)

    # Added to original LDM
    parser.add_argument("--fp16", type=str2bool, default=True,
                        help="run inference in mixed precision",)
    parser.add_argument("--use-stablediffusion", type=str2bool, default=True,
                        help="load the stable diffusion model, if False uses the LDM `text2img-large` model instead",)

    # decoder optimization
    parser.add_argument("--decoder-optimization", type=str2bool, default=True,
                        help="optimize the weights of the decoder for each image",)
    parser.add_argument("--decoderopt-it", type=int, default=100,
                        help="iterations of decoder finetuning, ignored if not using decoder optimization",)

    parser.add_argument("--dilate-mask", type=str2bool, default=False,
                        help="dilates the mask and shrinks it to original size over the diffusion timesteps, use for small or masks with fine details",)

    parser.add_argument("--seed", type=int, default=-1,
                        help="seed for everything, set to -1 to seed randomly",)

    # repaint
    parser.add_argument("--repaint-steps", type=int, default=5,
                        help="repetitions of each denoising step for repainting, set to 0 to disable repaint",)
    parser.add_argument("--repaint-jump", type=int, default=0,
                        help="jumps size in the repaint steps, jump size in DDIM steps, not DDPM steps, default=0, i.e. repeats the current step `repaint-step` times,",)

    parser.add_argument("--start-timestep", type=float, default=1.0,
                        help="SDEdit-like relative timestep to start the diffusion process from, i.e. 1.0 to start from pure noise, 0.5 = T/2",)
    parser.add_argument("--upscale-startstep", type=float, default=0.4,
                        help="start step for upscaling stages except final gridlike upscaling stage",)
    parser.add_argument("--grid-startstep", type=float, default=0.25,
                        help="start step for upscaling in the grid stage",)
    parser.add_argument("--clip-reranking", type=str2bool, default=True,
                        help="re-rerank first-stage outputs by clip similarity.",)
    parser.add_argument("--upscaling-mode", type=str, default='esrgan',
                        help="interpolation mode in the upscaling, 'esrgan', 'sharpen', 'bilinear', or 'bicubic' (or anything supported by torch functional interpolation)",)
    parser.add_argument("--straight-to-grid", type=str2bool, default=False,
                        help="after the first stage immediately go to the grid stage without intermediate steps",)
    parser.add_argument("--lowpass-reference", type=str, default='matching',
                        help=" 'matching', 'half', or 'no' .",)
    parser.add_argument("--conditional-upscale", type=str2bool, default=True,
                        help="do the upscaling with text conditioning",)
    parser.add_argument("--blended-upscale", type=str2bool, default=True,
                        help="do the upscaling with a reference image",)
    parser.add_argument("--grid-overlap", type=int, default=128,
                        help="overlap between different grid regions in pixels. Must be multiple of 64.",)
    parser.add_argument("--max-edgelen", type=int, default=12 * 64,
                        help="Maximum edge length of square images processable by the used GPU. Default value requires 25GB of VRAM.",)
    parser.add_argument("--first-stage-size", type=int, default=512,
                        help="Resolution to be used in the first stage. Default is 512 for stable diffusion, largest possible on V100 is 960",)
    parser.add_argument("--first-stage-batch", type=int, default=5,
                        help="Batch size for first stage. If clip-reranking is enabled, the image with the highest clip similarity is chosen, else the first one is used in subsequent stages",)

    arglist = parser.parse_args()
    main(arglist)
