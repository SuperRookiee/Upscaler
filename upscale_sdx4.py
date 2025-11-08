import argparse
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
from tqdm import tqdm

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def load_images(directory: Path) -> Iterable[Path]:
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    )


def configure_pipeline(device: torch.device) -> StableDiffusionUpscalePipeline:
    dtype = torch.float16 if device.type != "cpu" else torch.float32
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        torch_dtype=dtype,
    )

    if device.type == "mps":
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        if hasattr(pipe, "upcast_vae"):
            pipe.upcast_vae()
        pipe.to(device)
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae.to(device, dtype=torch.float32)
        if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "fallback_on_cpu"):
            torch.backends.mps.fallback_on_cpu(True)
    else:
        pipe = pipe.to(device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

    return pipe


def autocast_context(device: torch.device):
    if device.type in {"mps", "cuda"}:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", default="results")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pipe = configure_pipeline(device)

    images = list(load_images(input_dir))
    if not images:
        print("no images found in", input_dir)
        return

    prompt = args.prompt.strip() or "high-resolution photo"

    for path in tqdm(images, desc="Upscaling"):
        with Image.open(path) as img:
            lr_image = img.convert("RGB")
        with torch.inference_mode(), autocast_context(device):
            result = pipe(
                prompt=prompt,
                image=lr_image,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
            )

        out_path = output_dir / path.name
        result.images[0].save(out_path)

    print("done ->", output_dir)


if __name__ == "__main__":
    main()
