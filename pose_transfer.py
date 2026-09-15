"""pose_transfer.py

Core PoseTransfer class:
- extracts pose from a reference image using MediaPipe
- renders a pose map image suitable for ControlNet (OpenPose-style skeleton)
- uses Hugging Face Diffusers StableDiffusionControlNetPipeline + ControlNet

This is a practical, minimal implementation. For production or large-scale
use, add proper error handling, model caching, scheduler choices, and
resource management specific to your environment.
"""
from __future__ import annotations
import os
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel,
                       UniPCMultistepScheduler)
from transformers import logging as transformers_logging

# reduce noisy transformers logs
transformers_logging.set_verbosity_error()

# Local helper functions
from utils.image_utils import load_image, save_image, make_pose_map_from_landmarks, resize_to


DEFAULT_SD_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_CONTROLNET = "lllyasviel/sd-controlnet-openpose"


class PoseTransfer:
    def __init__(self,
                 device: Optional[str] = None,
                 sd_model: str = DEFAULT_SD_MODEL,
                 controlnet_model: str = DEFAULT_CONTROLNET,
                 dtype: Optional[torch.dtype] = None,
                 cache_dir: Optional[str] = None,
                 enable_attention_slicing: bool = True):
        """Initialize the pose transfer pipeline.

        device: "cuda" or "cpu" or None (auto-detect)
        sd_model: Hugging Face repo id for Stable Diffusion base model
        controlnet_model: Hugging Face repo id for ControlNet (openpose)
        dtype: torch.float16 recommended for CUDA; float32 for CPU
        cache_dir: optional local cache directory for models
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if dtype is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.torch_dtype = dtype

        self.sd_model = sd_model
        self.controlnet_model = controlnet_model
        self.cache_dir = cache_dir

        # load controlnet
        print(f"Loading ControlNet model '{controlnet_model}' ...")
        controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=self.torch_dtype, cache_dir=self.cache_dir)

        print(f"Loading Stable Diffusion model '{sd_model}' with ControlNet ...")
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model,
            controlnet=controlnet,
            torch_dtype=self.torch_dtype,
            cache_dir=self.cache_dir,
        )

        # recommended scheduler
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)

        # performance / memory options
        if enable_attention_slicing:
            try:
                self.pipeline.enable_attention_slicing()
            except Exception:
                pass

        # move pipeline to device
        self.pipeline.to(self.device)

    def extract_pose_map(self, reference_image_path: str, size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """Load reference image and return an OpenPose-style pose map PIL image.

        This uses MediaPipe to detect pose keypoints and draws a simple
        skeleton on transparent background (white background output for ControlNet).
        """
        img = load_image(reference_image_path)
        # produce pose map using helper which uses MediaPipe
        pose_map = make_pose_map_from_landmarks(img, size=size)
        return pose_map

    def transfer_pose(self,
                      target_image: str,
                      reference_image: str,
                      prompt: str = "person",
                      num_steps: int = 30,
                      guidance_scale: float = 7.5,
                      output_size: Tuple[int, int] = (512, 512),
                      negative_prompt: Optional[str] = None,
                      generator: Optional[torch.Generator] = None) -> Image.Image:
        """Perform pose transfer and return a PIL Image result.

        target_image: path to target person image (used as base / init image)
        reference_image: path containing desired pose
        """
        # prepare images
        target = load_image(target_image).convert("RGB")
        target = resize_to(target, output_size)

        pose_map = self.extract_pose_map(reference_image, size=output_size).convert("RGB")

        # The StableDiffusionControlNetPipeline for image-to-image has arguments:
        # prompt, image (init image), control_image (controlnet conditioning image)
        print("Running the diffusion pipeline ... this may take a while.")

        with torch.autocast(self.device.type) if self.device.type == "cuda" else torch.cpu.amp.autocast(enabled=False):
            outputs = self.pipeline(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                image=target,
                control_image=pose_map,
                negative_prompt=negative_prompt,
                generator=generator,
                height=output_size[1],
                width=output_size[0],
            )

        if hasattr(outputs, "images"):
            result = outputs.images[0]
            return result
        # fallback
        return Image.fromarray(np.array(outputs[0]))


if __name__ == "__main__":
    # quick smoke test (no model downloads) - ensure imports and basic flows work
    print("PoseTransfer module loaded. Run pose_transfer_cli.py to use the CLI.")
