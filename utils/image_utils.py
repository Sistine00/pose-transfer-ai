"""Image utility helpers used by the pose transfer code.

Contains:
- load_image, save_image
- simple MediaPipe-based pose extraction and drawing
- resizing helpers
"""
from __future__ import annotations
from typing import Tuple

import io
from PIL import Image, ImageDraw
import numpy as np
import cv2

try:
    import mediapipe as mp
except Exception:
    mp = None


def load_image(path: str) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGB")


def save_image(img: Image.Image, path: str, quality: int = 95) -> None:
    img.save(path, quality=quality)


def resize_to(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.LANCZOS)


def make_pose_map_from_landmarks(img: Image.Image, size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """Create a simple pose map from an image using MediaPipe pose estimation.

    The output is an RGB image (white background) with black lines marking the skeleton.
    This is compatible with ControlNet's openpose conditioning which expects a drawing-like input.
    """
    if mp is None:
        raise RuntimeError("mediapipe is required for pose extraction. Install with `pip install mediapipe`")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

    # convert to OpenCV image
    w, h = img.size
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    results = pose.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    # create blank white canvas
    canvas = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    if results.pose_landmarks is None:
        # no landmarks found, return blank canvas
        return canvas

    # draw landmarks scaled to target size
    landmarks = results.pose_landmarks.landmark

    # list of landmark indices to connect (simple subset similar to openpose)
    skeleton_edges = [
        (11, 12),  # shoulders
        (12, 14), (14, 16),  # right arm
        (11, 13), (13, 15),  # left arm
        (24, 23), (23, 11), (24, 12),  # hips to shoulders
        (24, 26), (26, 28),  # right leg
        (23, 25), (25, 27),  # left leg
    ]

    # helper to convert normalized landmark to pixel coords
    def lm_to_xy(lm):
        return (int(lm.x * size[0]), int(lm.y * size[1]))

    # draw keypoints
    for i, lm in enumerate(landmarks):
        if lm.visibility < 0.2:
            continue
        x, y = lm_to_xy(lm)
        r = max(2, int(size[0] / 200))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 0))

    # draw skeleton edges
    for a, b in skeleton_edges:
        if a < len(landmarks) and b < len(landmarks):
            la = landmarks[a]
            lb = landmarks[b]
            if la.visibility < 0.2 or lb.visibility < 0.2:
                continue
            ax, ay = lm_to_xy(la)
            bx, by = lm_to_xy(lb)
            draw.line((ax, ay, bx, by), fill=(0, 0, 0), width=max(2, int(size[0] / 150)))

    pose.close()
    return canvas
