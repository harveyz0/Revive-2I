import argparse, os, re
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from contextlib import nullcontext
from einops import repeat
from torch import autocast
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts, logger
from transformers import logging

logging.set_verbosity_error()


def process_source_img_dir(source_img_dir="data/skull2dog/testA",
                           config="optimizedSD/v1-inference.yaml",
                           input_prompt="A realistic photo of a dog",
                           unet_bs=1,
                           turbo=False,
                           height=512,
                           width=512,
                           device="cuda",
                           seed=42,
                           ckpt="models/ldm/stable-diffusion-v1/model.ckpt",
                           outpath="outputs/txt-guid-i2i-samples",
                           precision="autocast",
                           from_file=None,
                           n_samples=1,
                           ddim_eta=0.0,
                           ddim_steps=100,
                           strength=0.95,
                           n_iter=1,
                           scale=7.5,
                           sampler="ddim",
                           iterate_seed=False,
                           output_format="png"):
    print('Processing {} images from {}'.format(
        len(os.listdir(source_img_dir)), source_img_dir))
    os.makedirs(outpath, exist_ok=True)
    model, modelCS, modelFS = build_models(ckpt, config, device, unet_bs,
                                           turbo)
    sample_path, seeds = None, None
    for filename in os.listdir(source_img_dir):
        if not (filename.endswith('.jpg') or filename.endswith('.png')):
            print(f"INFO : Skipping file {filename}")
            continue
        image_file = os.path.join(source_img_dir, filename)
        assert os.path.isfile(image_file), "Image not found"
        source_image = load_img(image_file, height, width).to(device)

        # model = instantiate_from_config(config.modelUNet)
        # _, _ = model.load_state_dict(sd, strict=False)
        # model.eval()
        # model.cdevice = opt.device
        # model.unet_bs = opt.unet_bs
        # model.turbo = opt.turbo

        # modelCS = instantiate_from_config(config.modelCondStage)
        # _, _ = modelCS.load_state_dict(sd, strict=False)
        # modelCS.eval()
        # modelCS.cond_stage_model.device = opt.device

        # modelFS = instantiate_from_config(config.modelFirstStage)
        # _, _ = modelFS.load_state_dict(sd, strict=False)
        # modelFS.eval()
        # del sd
        if device != "cpu" and precision == "autocast":
            model.half()
            modelCS.half()
            modelFS.half()
            source_image = source_image.half()

        batch_size = n_samples
        data = get_prompts(filename, batch_size, from_file, input_prompt)
        modelFS.to(device)

        source_image = repeat(source_image, "1 ... -> b ...", b=batch_size)
        init_latent = modelFS.get_first_stage_encoding(
            modelFS.encode_first_stage(source_image))  # move to latent space

        if device != "cpu":
            mem = torch.cuda.memory_allocated(device=device) / 1e6
            modelFS.to("cpu")
            while torch.cuda.memory_allocated(device=device) / 1e6 >= mem:
                time.sleep(1)

        assert 0.0 <= strength <= 1.0, "can only work with strength in [0.0, 1.0]"
        t_enc = int(strength * ddim_steps)
        print(f"target t_enc is {t_enc} steps")

        if precision == "autocast" and device != "cpu":
            precision_scope = autocast
        else:
            precision_scope = nullcontext

        seeds = ""
        with torch.no_grad():
            for n in trange(n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):

                    sample_path = os.path.join(
                        outpath, "_".join(re.split(":| ", prompts[0])))[:150]
                    os.makedirs(sample_path, exist_ok=True)
                    base_count = len(os.listdir(sample_path))

                    with precision_scope("cuda"):
                        modelCS.to(device)
                        uc = None
                        if scale != 1.0:
                            uc = modelCS.get_learned_conditioning(batch_size *
                                                                  [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        subprompts, weights = split_weighted_subprompts(
                            prompts[0])
                        if len(subprompts) > 1:
                            c = torch.zeros_like(uc)
                            totalWeight = sum(weights)
                            # normalize each "sub prompt" and add it
                            for i in range(len(subprompts)):
                                weight = weights[i]
                                # if not skip_normalize:
                                weight = weight / totalWeight
                                c = torch.add(c,
                                              modelCS.get_learned_conditioning(
                                                  subprompts[i]),
                                              alpha=weight)
                        else:
                            c = modelCS.get_learned_conditioning(prompts)

                        if device != "cpu":
                            mem = torch.cuda.memory_allocated(
                                device=device) / 1e6
                            modelCS.to("cpu")
                            while torch.cuda.memory_allocated(
                                    device=device) / 1e6 >= mem:
                                time.sleep(1)

                        # encode (scaled latent)
                        z_enc = model.stochastic_encode(
                            init_latent,
                            torch.tensor([t_enc] * batch_size).to(device),
                            seed,
                            ddim_eta,
                            ddim_steps,
                        )
                        # decode it
                        samples_ddim = model.sample(
                            t_enc,
                            c,
                            z_enc,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc,
                            sampler=sampler)

                        modelFS.to(device)
                        print("saving images")
                        for i in range(batch_size):

                            x_samples_ddim = modelFS.decode_first_stage(
                                samples_ddim[i].unsqueeze(0))
                            x_sample = torch.clamp(
                                (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_sample = 255.0 * rearrange(
                                x_sample[0].cpu().numpy(), "c h w -> h w c")
                            # Image.fromarray(x_sample.astype(np.uint8)).save(
                            #     os.path.join(sample_path, "seed_" + str(opt.seed) + "_" + f"{base_count:05}.{opt.format}")
                            # )
                            # replace .jpg with ''
                            new_filename = filename.replace(
                                ".jpg", "") + '_' + str(
                                    seed) + "_" + f"{base_count:05}.{output_format}"
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(sample_path, new_filename))
                            seeds += str(seed) + ","
                            # opt.seed += 1 # stopped incrememnting seed to keep the same seed for each prompt
                            if iterate_seed:
                                seed += 1
                            base_count += 1

                        if device != "cpu":
                            mem = torch.cuda.memory_allocated(
                                device=device) / 1e6
                            modelFS.to("cpu")
                            while torch.cuda.memory_allocated(
                                    device=device) / 1e6 >= mem:
                                time.sleep(1)

                        del samples_ddim
                        print("memory_final = ",
                              torch.cuda.memory_allocated(device=device) / 1e6)
    return sample_path, seeds


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


def load_img(path, h0, w0):

    image = Image.open(path).convert("RGB")
    w, h = image.size

    print(f"loaded input image of size ({w}, {h}) from {path}")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    w, h = map(lambda x: x - x % 64,
               (w, h))  # resize to integer multiple of 32

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def get_prompts(filename, batch_size, from_file=None, input_prompt=None):
    data = []
    if not from_file:
        assert input_prompt is not None
        if input_prompt == 'class':
            if 'boston' in filename:
                prompt = 'A photo of the head of a boston terrier dog'
            elif 'boxer' in filename:
                prompt = 'A photo of the head of a boxer dog'
            elif 'chi' in filename:
                prompt = 'A photo of the head of a chihuahua dog'
            elif 'dane' in filename:
                prompt = 'A photo of the head of a great dane dog'
            elif 'pek' in filename:
                prompt = 'A photo of the head of a pekingese dog'
            elif 'rot' in filename:
                prompt = 'A photo of the head of a rottweiler dog'
            else:
                prompt = 'A photo of the head of a dog'

        elif input_prompt == 'short class':
            if 'boston' in filename:
                prompt = 'boston terrier'
            elif 'boxer' in filename:
                prompt = 'boxer'
            elif 'chi' in filename:
                prompt = 'chihuahua'
            elif 'dane' in filename:
                prompt = 'great dane'
            elif 'pek' in filename:
                prompt = 'pekingese'
            elif 'rot' in filename:
                prompt = 'rottweiler'
            else:
                prompt = 'dog'

        elif input_prompt == 'no photo':
            if 'boston' in filename:
                prompt = 'boston terrier head'
            elif 'boxer' in filename:
                prompt = 'boxer head'
            elif 'chi' in filename:
                prompt = 'chihuahua head'
            elif 'dane' in filename:
                prompt = 'great dane head'
            elif 'pek' in filename:
                prompt = 'pekingese head'
            elif 'rot' in filename:
                prompt = 'rottweiler head'
            else:
                prompt = 'dog head'

        else:
            prompt = input_prompt
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {from_file}")
        with open(from_file, "r") as f:
            prompts = f.read().splitlines()
            data = batch_size * list(prompts)
            data = list(chunk(sorted(data), batch_size))
    return data


def build_models(ckpt, config, device, unet_bs, turbo):
    state_dict = load_model_from_config(f"{ckpt}")
    li, lo = [], []
    for key, value in state_dict.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        state_dict["model1." + key[6:]] = state_dict.pop(key)
    for key in lo:
        state_dict["model2." + key[6:]] = state_dict.pop(key)

    config = OmegaConf.load(f"{config}")

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.cdevice = device
    model.unet_bs = unet_bs
    model.turbo = turbo

    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(state_dict, strict=False)
    modelCS.eval()
    modelCS.cond_stage_model.device = device

    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(state_dict, strict=False)
    modelFS.eval()
    return model, modelCS, modelFS


def get_args(config="optimizedSD/v1-inference.yaml",
             ckpt="models/ldm/stable-diffusion-v1/model.ckpt",
             data_directory="data/skull2dog/testA"):
    parser = argparse.ArgumentParser()

    parser.add_argument("prompt",
                        type=str,
                        nargs="?",
                        default="A realistic photo of a dog",
                        help="the prompt to render")

    parser.add_argument("--outdir",
                        type=str,
                        nargs="?",
                        help="dir to write results to",
                        default="outputs/txt-guid-i2i-samples")

    parser.add_argument("--source-img",
                        type=str,
                        nargs="?",
                        help="path to the input image")

    parser.add_argument(
        "--source-img-dir",
        type=str,
        nargs="?",
        help=
        "path to the input image directory (if source-img is not specified)",
        default=data_directory)

    parser.add_argument("--ckpt",
                        type=str,
                        default=ckpt,
                        help="path to the model checkpoint to use")

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.95,
        help=
        "strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )

    parser.add_argument("--config",
                        type=str,
                        default=config,
                        help="path to the config file to use")

    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help=
        "do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="do not save individual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling)",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help=
        "how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help=
        "unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument("--iterate_seed",
                        action="store_true",
                        help="iterate the seed for each sample")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="CPU or GPU (cuda/cuda:0/cuda:1/...)",
    )
    parser.add_argument(
        "--unet_bs",
        type=int,
        default=1,
        help=
        "Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Reduces inference time on the expense of 1GB VRAM",
    )
    parser.add_argument("--precision",
                        type=str,
                        help="evaluate at this precision",
                        choices=["full", "autocast"],
                        default="autocast")
    parser.add_argument(
        "--format",
        type=str,
        help="output image format",
        choices=["jpg", "png"],
        default="png",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        help="sampler",
        choices=["ddim"],
        default="ddim",
    )
    return parser.parse_args()


def main():
    opt = get_args()

    if opt.ckpt:
        ckpt = opt.ckpt
    if opt.config:
        config = opt.config

    tic = time.time()

    if opt.seed is None:
        opt.seed = randint(0, 1000000)
    seed_everything(opt.seed)

    # Logging
    logger(vars(opt), log_csv="logs/img2img_logs.csv")

    if opt.source_img:
        raise NotImplementedError("Took this out")
        # process_source_img(opt)
    elif opt.source_img_dir:
        sample_path, seeds = process_source_img_dir(
            source_img_dir=opt.source_img_dir,
            config=config,
            input_prompt=opt.prompt,
            unet_bs=opt.unet_bs,
            turbo=opt.turbo,
            height=opt.H,
            width=opt.W,
            device=opt.device,
            seed=opt.seed,
            ckpt=ckpt,
            outpath=opt.outdir,
            precision=opt.precision,
            from_file=opt.from_file,
            n_samples=opt.n_samples,
            ddim_eta=opt.ddim_eta,
            ddim_steps=opt.ddim_steps,
            strength=opt.strength,
            n_iter=opt.n_iter,
            scale=opt.scale,
            sampler=opt.sampler,
            iterate_seed=opt.iterate_seed,
            output_format=opt.format)
    toc = time.time()

    time_taken = (toc - tic) / 60.0

    print(("Samples finished in {0:.2f} minutes and exported to " +
           sample_path + "\n Seeds used = " + seeds[:-1]).format(time_taken))


if __name__ == '__main__':
    main()
