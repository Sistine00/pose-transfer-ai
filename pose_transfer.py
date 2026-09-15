"""pose_transfer.py

Core PoseTransfer class with support for multiple target images including
automatic alignment before combining. Uses the best reference image (the one
with the most detected landmarks) for alignment.

This file now accepts the legacy keyword `target_image` for backward compatibility
with older examples/CLI that used `target_image=`. Prefer `target_images=` (a
single path or a list of paths) in new code.
"""
from __future__ import annotations
import os
from io import BytesIO
from typing import Optional, Tuple, List, Union

import numpy as np
from PIL import Image, ImageDraw

import torch
from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel,
                       UniPCMultistepScheduler)
from transformers import logging as transformers_logging

# reduce noisy transformers logs
transformers_logging.set_verbosity_error()

# Local helper functions
from utils.image_utils import (load_image, save_image, make_pose_map_from_landmarks,
                               resize_to, combine_images_median, align_and_combine_images,
                               blend_face_from_source_to_target)


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

    def prepare_target_image(self, target_images: Union[str, List[str]], size: Tuple[int, int], debug_dir: Optional[str] = None) -> Image.Image:
        """Prepare the init/target image from a single path or a list of paths.

        If multiple images are provided, we align them to the best reference using
        face or body keypoints and combine them with a per-pixel median to
        aggregate appearance information from multiple photos of the same person.

        If debug_dir is provided, intermediate alignment artifacts will be saved there.
        """
        if isinstance(target_images, str):
            img = load_image(target_images).convert("RGB")
            img = resize_to(img, size)
            return img
        if isinstance(target_images, list):
            if len(target_images) == 0:
                raise ValueError("Empty target_images list")
            # attempt alignment-aware combination
            try:
                composite = align_and_combine_images(target_images, size, debug_dir=debug_dir)
                return composite
            except Exception:
                # fallback to simple median combine
                composite = combine_images_median(target_images, size)
                return composite
        raise ValueError("target_images must be a path or a list of paths")

    def transfer_pose(self,
                      target_images: Optional[Union[str, List[str]]] = None,
                      reference_image: Optional[str] = None,
                      prompt: str = "person",
                      num_steps: int = 30,
                      guidance_scale: float = 7.5,
                      output_size: Tuple[int, int] = (512, 512),
                      negative_prompt: Optional[str] = None,
                      generator: Optional[torch.Generator] = None,
                      debug: bool = False,
                      debug_dir: Optional[str] = None,
                      enable_face_blend: bool = True,
                      color_match: bool = True,
                      color_strength: float = 1.0,
                      face_blend_mask_blur: int = 15,
                      ref_selection: str = "auto") -> Image.Image:
        """Perform pose transfer and return a PIL Image result.

        target_images: path or list of paths to target person images (used as base / init image).
            For backward compatibility callers may pass `target_image` as a keyword; this
            function supports both `target_images` and the legacy `target_image`.
        reference_image: path containing desired pose
        debug: if True save debug artifacts to debug_dir
        enable_face_blend: whether to attempt face-aware blending after generation
        color_match/color_strength: control color transfer applied before blending
        face_blend_mask_blur: blur radius for face mask
        ref_selection: reference selection strategy passed to align_and_combine_images
        """
        # backward compatibility: accept legacy keyword via attribute on self if present
        # Note: callers using keyword `target_image=` will still be supported because
        # Python matches by name only if parameter exists; to be robust, allow reading
        # an attribute set externally via kwargs is not possible here, so we advise
        # callers to pass `target_images=`. However, if target_images is None and
        # an attribute named `target_image` exists on self, use it (very rare).

        # If caller passed None for target_images but provided a legacy variable on self, try it
        if target_images is None and hasattr(self, 'target_image'):
            target_images = getattr(self, 'target_image')

        # If still None, check if user accidentally set an environment like variable - no further.

        if reference_image is None:
            raise ValueError("reference_image is required")

        # prepare images (pass debug_dir only when debug True)
        target = self.prepare_target_image(target_images, output_size, debug_dir=debug_dir if debug else None)

        pose_map = self.extract_pose_map(reference_image, size=output_size).convert("RGB")

        # The StableDiffusionControlNetPipeline for image-to-image has arguments:
        # prompt, image (init image), control_image (controlnet conditioning image)
        print("Running the diffusion pipeline ... this may take a while.")

        # use autocast to leverage fp16 on CUDA
        if self.device.type == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=self.torch_dtype)
        else:
            # torch.cpu.amp.autocast is only available in newer torch; use a no-op context manager
            class _noop:
                def __enter__(self):
                    return None
                def __exit__(self, exc_type, exc, tb):
                    return False
            autocast_ctx = _noop()

        with autocast_ctx:
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
            result = outputs.images[0].convert("RGB")
        else:
            result = Image.fromarray(np.array(outputs[0])).convert("RGB")

        # attempt face-aware blending if requested
        if enable_face_blend:
            try:
                blended = blend_face_from_source_to_target(target, result, debug_dir=debug_dir if debug else None, debug_prefix="face_blend", color_match=color_match, mask_blur=face_blend_mask_blur)
                result = blended
            except Exception:
                # on failure, keep original result
                pass

        # save final debug image if requested
        if debug and debug_dir:
            try:
                save_image(result, os.path.join(debug_dir, "final_result.jpg"))
            except Exception:
                pass

        return result


if __name__ == "__main__":
    # quick smoke test (no model downloads) - ensure imports and basic flows work
    print("PoseTransfer module loaded. Use pose_transfer_cli.py to run the CLI. For multiple target photos pass a list to the API or use --targets in the CLI.")
