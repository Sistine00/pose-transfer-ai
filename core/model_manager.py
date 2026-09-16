"""
模型管理器 - 处理多个 ControlNet 模型和 Stable Diffusion 版本
支持模型切换、缓存和版本管理
"""
from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Literal
from pathlib import Path
import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    DDIMScheduler,
    PNDMScheduler,
)

logger = logging.getLogger(__name__)

PoseEstimator = Literal["openpose", "dwpose", "depth", "canny"]
SDVersion = Literal["v1.5", "v2", "sdxl"]


class ModelManager:
    """管理多个 ControlNet 和 Stable Diffusion 模型"""
    
    # 预定义模型映射
    SD_MODELS = {
        "v1.5": "runwayml/stable-diffusion-v1-5",
        "v2": "stabilityai/stable-diffusion-2",
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    }
    
    CONTROLNET_MODELS = {
        "openpose": "lllyasviel/sd-controlnet-openpose",
        "dwpose": "bdsqlsz/control_v11p_sd15_dwpose",
        "depth": "lllyasviel/sd-controlnet-depth",
        "canny": "lllyasviel/sd-controlnet-canny",
    }
    
    SCHEDULERS = {
        "unipc": UniPCMultistepScheduler,
        "ddim": DDIMScheduler,
        "pndm": PNDMScheduler,
    }
    
    def __init__(
        self,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
        enable_attention_slicing: bool = True,
        enable_memory_efficient: bool = True,
    ):
        """初始化模型管理器
        
        Args:
            device: "cuda" 或 "cpu"，默认自动检测
            dtype: torch.float16 或 torch.float32
            cache_dir: 模型缓存目录
            enable_attention_slicing: 启用注意力切片以节省内存
            enable_memory_efficient: 启用内存优化
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.cache_dir = cache_dir or "./models_cache"
        self.enable_attention_slicing = enable_attention_slicing
        self.enable_memory_efficient = enable_memory_efficient
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.pipeline: Optional[StableDiffusionControlNetPipeline] = None
        self.current_sd_version: Optional[SDVersion] = None
        self.current_controlnet: Optional[PoseEstimator] = None
        
        logger.info(f"ModelManager initialized on device: {self.device}, dtype: {self.dtype}")
    
    def load_pipeline(
        self,
        sd_version: SDVersion = "v1.5",
        controlnet_model: PoseEstimator = "openpose",
        scheduler: str = "unipc",
    ) -> StableDiffusionControlNetPipeline:
        """加载 Stable Diffusion + ControlNet 管道
        
        Args:
            sd_version: Stable Diffusion 版本
            controlnet_model: ControlNet 姿态检测模型类型
            scheduler: 调度器类型 (unipc, ddim, pndm)
            
        Returns:
            加载好的 diffusers 管道
        """
        # 如果已经加载相同配置，直接返回
        if (self.pipeline is not None and 
            self.current_sd_version == sd_version and 
            self.current_controlnet == controlnet_model):
            logger.info(f"Pipeline already loaded: SD {sd_version} + {controlnet_model}")
            return self.pipeline
        
        logger.info(f"Loading pipeline: SD {sd_version} + {controlnet_model} ControlNet")
        
        # 获取模型ID
        sd_model_id = self.SD_MODELS.get(sd_version, self.SD_MODELS["v1.5"])
        controlnet_id = self.CONTROLNET_MODELS.get(controlnet_model, self.CONTROLNET_MODELS["openpose"])
        
        # 加载 ControlNet
        logger.info(f"Loading ControlNet: {controlnet_id}")
        controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
        )
        
        # 选择合适的 Pipeline 类
        if sd_version == "sdxl":
            PipelineClass = StableDiffusionXLControlNetPipeline
        else:
            PipelineClass = StableDiffusionControlNetPipeline
        
        # 加载 Stable Diffusion 模型
        logger.info(f"Loading Stable Diffusion: {sd_model_id}")
        self.pipeline = PipelineClass.from_pretrained(
            sd_model_id,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
        )
        
        # 设置调度器
        scheduler_class = self.SCHEDULERS.get(scheduler, UniPCMultistepScheduler)
        self.pipeline.scheduler = scheduler_class.from_config(self.pipeline.scheduler.config)
        logger.info(f"Set scheduler to: {scheduler}")
        
        # 性能优化
        if self.enable_attention_slicing:
            try:
                self.pipeline.enable_attention_slicing()
                logger.info("Enabled attention slicing")
            except Exception as e:
                logger.warning(f"Could not enable attention slicing: {e}")
        
        if self.enable_memory_efficient:
            try:
                self.pipeline.enable_model_cpu_offload()
                logger.info("Enabled memory efficient inference")
            except Exception as e:
                logger.warning(f"Could not enable memory efficient mode: {e}")
        
        # 移到设备
        self.pipeline.to(self.device)
        logger.info(f"Pipeline moved to device: {self.device}")
        
        self.current_sd_version = sd_version
        self.current_controlnet = controlnet_model
        
        return self.pipeline
    
    def switch_controlnet(
        self,
        controlnet_model: PoseEstimator,
    ) -> StableDiffusionControlNetPipeline:
        """切换 ControlNet 模型，保持 Stable Diffusion 模型不变"""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline first.")
        
        logger.info(f"Switching ControlNet to: {controlnet_model}")
        
        controlnet_id = self.CONTROLNET_MODELS.get(controlnet_model, self.CONTROLNET_MODELS["openpose"])
        controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
        )
        
        self.pipeline.controlnet = controlnet
        self.current_controlnet = controlnet_model
        logger.info(f"ControlNet switched to: {controlnet_model}")
        
        return self.pipeline
    
    def get_available_sd_versions(self) -> list[str]:
        """获取可用的 Stable Diffusion 版本"""
        return list(self.SD_MODELS.keys())
    
    def get_available_controlnets(self) -> list[str]:
        """获取可用的 ControlNet 模型"""
        return list(self.CONTROLNET_MODELS.keys())
    
    def clear_cache(self) -> None:
        """清除模型缓存"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self.current_sd_version = None
            self.current_controlnet = None
        
        # 清空 GPU 缓存
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("Model cache cleared")
