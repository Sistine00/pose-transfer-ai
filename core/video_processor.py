"""
视频处理模块 - 支持视频文件的逐帧姿态转移
支持批处理、跳帧、进度显示
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VideoProcessor:
    """处理视频文件，提取帧并进行姿态转移"""
    
    def __init__(self, output_fps: int = 30, max_frames: Optional[int] = None):
        """初始化视频处理器
        
        Args:
            output_fps: 输出视频 FPS
            max_frames: 最大处理帧数限制
        """
        self.output_fps = output_fps
        self.max_frames = max_frames
    
    def extract_frames(
        self,
        video_path: str,
        skip_frames: int = 1,
        output_dir: Optional[str] = None,
    ) -> List[Image.Image]:
        """从视频文件提取帧
        
        Args:
            video_path: 输入视频路径
            skip_frames: 跳帧数（1表示提取所有帧，2表示每隔1帧取1帧）
            output_dir: 可选的输出目录，用于保存提取的帧
            
        Returns:
            PIL Image 列表
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Extracting frames from: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video FPS: {fps}, Total frames: {total_frames}")
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        pbar = tqdm(total=min(total_frames, self.max_frames or total_frames), desc="Extracting frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 应用跳帧策略
            if frame_count % skip_frames == 0:
                # 检查帧数限制
                if self.max_frames and extracted_count >= self.max_frames:
                    break
                
                # 转换为 RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                frames.append(pil_image)
                
                # 保存帧
                if output_dir:
                    frame_path = os.path.join(output_dir, f"frame_{extracted_count:06d}.jpg")
                    pil_image.save(frame_path, quality=95)
                
                extracted_count += 1
                pbar.update(1)
            
            frame_count += 1
        
        pbar.close()
        cap.release()
        
        logger.info(f"Extracted {extracted_count} frames from video")
        return frames
    
    def create_video_from_frames(
        self,
        frames: List[Image.Image],
        output_path: str,
        fps: Optional[int] = None,
    ) -> None:
        """从帧列表创建视频文件
        
        Args:
            frames: PIL Image 列表
            output_path: 输出视频路径
            fps: 输出 FPS（如果为None，使用 self.output_fps）
        """
        if not frames:
            raise ValueError("No frames provided")
        
        fps = fps or self.output_fps
        output_fps = fps
        
        logger.info(f"Creating video: {output_path} at {output_fps} FPS")
        
        # 获取第一帧的尺寸
        first_frame = np.array(frames[0])
        height, width = first_frame.shape[:2]
        
        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        if not out.isOpened():
            raise RuntimeError(f"Failed to create video writer for: {output_path}")
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        for frame in tqdm(frames, desc="Writing video"):
            # 转换为 BGR 格式
            frame_array = np.array(frame)
            if len(frame_array.shape) == 2:  # 灰度图
                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_GRAY2BGR)
            else:  # RGB 图
                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            
            out.write(frame_array)
        
        out.release()
        logger.info(f"Video saved: {output_path}")
    
    def batch_process_video(
        self,
        video_path: str,
        process_fn,
        reference_image: str,
        prompt: str,
        output_dir: str = "./video_output",
        skip_frames: int = 1,
        **process_kwargs
    ) -> str:
        """批量处理视频的所有帧
        
        Args:
            video_path: 输入视频路径
            process_fn: 处理函数（如 pt.transfer_pose）
            reference_image: 参考姿态图像
            prompt: 提示词
            output_dir: 输出目录
            skip_frames: 跳帧数
            **process_kwargs: 传递给 process_fn 的额外参数
            
        Returns:
            输出视频路径
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        frames_dir = os.path.join(output_dir, "processed_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # 提取帧
        frames = self.extract_frames(video_path, skip_frames=skip_frames)
        logger.info(f"Processing {len(frames)} frames...")
        
        # 处理每一帧
        processed_frames = []
        for idx, frame in enumerate(tqdm(frames, desc="Processing frames")):
            # 临时保存当前帧
            temp_frame_path = os.path.join(frames_dir, f"temp_frame_{idx:06d}.jpg")
            frame.save(temp_frame_path)
            
            # 处理帧
            try:
                result = process_fn(
                    target_images=temp_frame_path,
                    reference_image=reference_image,
                    prompt=prompt,
                    **process_kwargs
                )
                processed_frames.append(result)
                
                # 保存处理后的帧
                result_path = os.path.join(frames_dir, f"processed_{idx:06d}.jpg")
                result.save(result_path)
            except Exception as e:
                logger.error(f"Error processing frame {idx}: {e}")
                # 使用原始帧作为回退
                processed_frames.append(frame)
            
            # 清理临时文件
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
        
        # 生成输出视频
        output_video_path = os.path.join(output_dir, "output.mp4")
        self.create_video_from_frames(processed_frames, output_video_path)
        
        logger.info(f"Video processing complete: {output_video_path}")
        return output_video_path
    
    def get_video_info(self, video_path: str) -> dict:
        """获取视频信息"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
        }
        
        cap.release()
        return info
