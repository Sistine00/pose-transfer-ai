"""Image utility helpers used by the pose transfer code.

Contains:
- load_image, save_image
- simple MediaPipe-based pose extraction and drawing
- resizing helpers
- combining multiple target images into a single appearance composite
- automatic alignment utilities (face and body keypoints)
- debug visualizations saving to disk
- face mask extraction and face-aware blending (seamlessClone)
"""
from __future__ import annotations
from typing import Tuple, List, Optional

import os
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, quality=quality)


def resize_to(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.LANCZOS)


def combine_images_median(paths: List[str], size: Tuple[int, int]) -> Image.Image:
    """Combine multiple images into a single composite using per-pixel median.

    This is a simple way to aggregate appearance information from multiple
    photos of the same person. It does not perform geometric alignment; for
    best results provide images with similar framing (face/body centered).
    """
    imgs = []
    for p in paths:
        im = load_image(p).convert("RGB")
        im = resize_to(im, size)
        imgs.append(np.array(im, dtype=np.uint8))

    if len(imgs) == 0:
        raise ValueError("No images provided to combine")
    if len(imgs) == 1:
        return Image.fromarray(imgs[0])

    stack = np.stack(imgs, axis=0)  # (N, H, W, C)
    median = np.median(stack.astype(np.float32), axis=0)
    median = np.clip(median, 0, 255).astype(np.uint8)
    return Image.fromarray(median)


# --- Alignment utilities & debug helpers ---

def _detect_face_landmark_points(img: Image.Image, indices: List[int] = [33, 263, 1]) -> Optional[List[Tuple[float, float]]]:
    """Detect facial landmark points using MediaPipe FaceMesh.

    Returns pixel coordinates for the requested landmark indices or None if not found.
    Default indices: [33 (left eye outer), 263 (right eye outer), 1 (nose tip)].
    """
    if mp is None:
        return None
    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False) as face_mesh:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = face_mesh.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = img.size[1], img.size[0]
        pts = []
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                pts.append((lm.x * w, lm.y * h))
            else:
                return None
        return pts


def _count_face_landmarks(img: Image.Image) -> int:
    """Return the number of detected face landmarks (approx)."""
    if mp is None:
        return 0
    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False) as face_mesh:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = face_mesh.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return 0
        landmarks = results.multi_face_landmarks[0].landmark
        return len(landmarks)


def _detect_face_landmarks_all(img: Image.Image) -> Optional[List[Tuple[float, float]]]:
    """Return all face landmarks (x,y) using MediaPipe FaceMesh or None."""
    if mp is None:
        return None
    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False) as face_mesh:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = face_mesh.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = img.size[1], img.size[0]
        pts = [(lm.x * w, lm.y * h) for lm in landmarks]
        return pts


def _detect_pose_points(img: Image.Image, indices: List[int] = [11, 12, 23]) -> Optional[List[Tuple[float, float]]]:
    """Detect body pose keypoints using MediaPipe Pose.

    Default indices: [11 left shoulder, 12 right shoulder, 23 left hip] — we use three points to compute a similarity transform.
    """
    if mp is None:
        return None
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3) as pose:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = pose.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks is None:
            return None
        landmarks = results.pose_landmarks.landmark
        h, w = img.size[1], img.size[0]
        pts = []
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                pts.append((lm.x * w, lm.y * h))
            else:
                return None
        return pts


def _count_pose_landmarks(img: Image.Image) -> int:
    """Return the number of detected pose landmarks with reasonable visibility."""
    if mp is None:
        return 0
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3) as pose:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = pose.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks is None:
            return 0
        count = 0
        for lm in results.pose_landmarks.landmark:
            if getattr(lm, 'visibility', 1.0) > 0.2:
                count += 1
        return count


def _estimate_transform(src_pts: List[Tuple[float, float]], dst_pts: List[Tuple[float, float]]) -> Optional[np.ndarray]:
    """Estimate a 2x3 affine (similarity) transform from src_pts to dst_pts.

    Uses cv2.estimateAffinePartial2D which estimates a similarity (scale+rotation+translation)
    when provided with at least 2 points; 3 points preferred.
    """
    if len(src_pts) < 2 or len(dst_pts) < 2:
        return None
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    try:
        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M is None:
            return None
        return M
    except Exception:
        return None


def _warp_image(img: Image.Image, M: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    arr = np.array(img)
    warped = cv2.warpAffine(arr, M, (size[0], size[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


def _draw_landmarks_overlay(img: Image.Image, face_landmarks: Optional[List[Tuple[float, float]]], pose_landmarks: Optional[List[Tuple[float, float]]]) -> Image.Image:
    """Return an image with landmarks drawn on top for visualization."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    if face_landmarks:
        for (x, y) in face_landmarks:
            r = max(2, int(out.size[0] / 200))
            draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0), width=2)
    if pose_landmarks:
        for (x, y) in pose_landmarks:
            r = max(2, int(out.size[0] / 200))
            draw.ellipse((x - r, y - r, x + r, y + r), outline=(0, 0, 255), width=2)
    return out


def _create_face_mask_from_landmarks(img: Image.Image, landmarks: List[Tuple[float, float]], blur: int = 15) -> np.ndarray:
    """Create a binary face mask from a list of face landmarks.

    Uses convex hull of landmarks and optional Gaussian blur to smooth edges.
    Returns mask as uint8 single-channel array with values 0 or 255.
    """
    h, w = img.size[1], img.size[0]
    pts = np.array(landmarks, dtype=np.int32)
    # compute convex hull
    hull = cv2.convexHull(pts)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if blur > 0:
        k = max(1, int(blur // 2) * 2 + 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def blend_face_from_source_to_target(source: Image.Image, target: Image.Image, debug_dir: Optional[str] = None, debug_prefix: str = "face_blend") -> Image.Image:
    """Blend the face region from source onto target using OpenCV seamlessClone.

    - Detect face landmarks on the source image (face must be present).
    - Create a face mask from landmarks and compute center for cloning.
    - Use cv2.seamlessClone to blend the source face into the target image.
    - If face detection fails, returns target unchanged.

    If debug_dir is provided, save mask and intermediate visualizations.
    """
    if mp is None:
        # cannot detect face; return target unchanged
        return target

    src = source.convert("RGB")
    tgt = target.convert("RGB")

    face_landmarks = _detect_face_landmarks_all(src)
    if face_landmarks is None:
        return target

    # create mask
    mask = _create_face_mask_from_landmarks(src, face_landmarks, blur=15)

    # compute center for seamlessClone as centroid of hull
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return target
    center = (int(np.mean(xs)), int(np.mean(ys)))

    src_np = cv2.cvtColor(np.array(src), cv2.COLOR_RGB2BGR)
    tgt_np = cv2.cvtColor(np.array(tgt), cv2.COLOR_RGB2BGR)

    # ensure mask is 3-channel for seamlessClone
    mask_3c = mask

    try:
        mixed = cv2.seamlessClone(src_np, tgt_np, mask, center, cv2.NORMAL_CLONE)
    except Exception:
        # fallback to normal clone with 3-channel mask
        try:
            mixed = cv2.seamlessClone(src_np, tgt_np, mask, center, cv2.MIXED_CLONE)
        except Exception:
            return target

    blended = Image.fromarray(cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB))

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        # save mask visualization
        mask_vis = Image.fromarray(mask).convert("RGB")
        save_image(mask_vis, os.path.join(debug_dir, f"{debug_prefix}_mask.jpg"))
        # save source overlay
        save_image(_draw_landmarks_overlay(src, face_landmarks, None), os.path.join(debug_dir, f"{debug_prefix}_source_overlay.jpg"))
        # save blended result
        save_image(blended, os.path.join(debug_dir, f"{debug_prefix}_blended.jpg"))

    return blended


def align_and_combine_images(paths: List[str], size: Tuple[int, int], debug_dir: Optional[str] = None) -> Image.Image:
    """Align multiple images to the best reference (the one with the most detected landmarks) and combine them using per-pixel median.

    If debug_dir is provided, save intermediate resized images, overlays, warped images and the final composite into that directory.
    """
    if len(paths) == 0:
        raise ValueError("No images provided")

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    # load and resize all images first
    imgs = [resize_to(load_image(p).convert("RGB"), size) for p in paths]

    # compute landmark counts for each image
    face_counts = [(_count_face_landmarks(img) if mp is not None else 0) for img in imgs]
    pose_counts = [(_count_pose_landmarks(img) if mp is not None else 0) for img in imgs]

    # choose reference by max of (face_count * 1000 + pose_count) so face detection is preferred
    scores = [fc * 1000 + pc for fc, pc in zip(face_counts, pose_counts)]
    best_idx = int(np.argmax(np.array(scores)))

    ref_img = imgs[best_idx]

    # prepare reference landmarks
    ref_face_pts = _detect_face_landmark_points(ref_img) if face_counts[best_idx] > 0 else None
    ref_pose_pts = _detect_pose_points(ref_img) if pose_counts[best_idx] > 0 else None

    if debug_dir:
        # save the selected reference and an overlay
        save_image(ref_img, os.path.join(debug_dir, f"ref_selected_{best_idx}.jpg"))
        overlay = _draw_landmarks_overlay(ref_img, _detect_face_landmarks_all(ref_img), ref_pose_pts)
        save_image(overlay, os.path.join(debug_dir, f"ref_selected_{best_idx}_overlay.jpg"))

    aligned_images = []

    # include all images: if an image is the reference, include as-is (resized)
    for i, img in enumerate(imgs):
        # save resized
        if debug_dir:
            save_image(img, os.path.join(debug_dir, f"img_{i}_resized.jpg"))
            # overlays on resized
            face_pts = _detect_face_landmarks_all(img)
            pose_pts = _detect_pose_points(img)
            overlay = _draw_landmarks_overlay(img, face_pts, pose_pts)
            save_image(overlay, os.path.join(debug_dir, f"img_{i}_resized_overlay.jpg"))

        if i == best_idx:
            aligned_images.append(np.array(img, dtype=np.uint8))
            continue

        M = None
        # try face alignment first (if reference has face and this image has face)
        if ref_face_pts is not None:
            src_face = _detect_face_landmark_points(img)
            if src_face is not None:
                M = _estimate_transform(src_face, ref_face_pts)
        # else try pose alignment if available
        if M is None and ref_pose_pts is not None:
            src_pose = _detect_pose_points(img)
            if src_pose is not None:
                M = _estimate_transform(src_pose, ref_pose_pts)

        if M is not None:
            try:
                warped = _warp_image(img, M, size)
                aligned_images.append(np.array(warped, dtype=np.uint8))
                if debug_dir:
                    save_image(warped, os.path.join(debug_dir, f"img_{i}_warped.jpg"))
                    # overlay on warped (draw reference landmarks for visualization)
                    warped_overlay = _draw_landmarks_overlay(warped, _detect_face_landmarks_all(ref_img), ref_pose_pts)
                    save_image(warped_overlay, os.path.join(debug_dir, f"img_{i}_warped_overlay.jpg"))
                continue
            except Exception:
                # fallback to unaligned resized
                pass

        aligned_images.append(np.array(img, dtype=np.uint8))

    # combine via median
    stack = np.stack(aligned_images, axis=0)
    median = np.median(stack.astype(np.float32), axis=0)
    median = np.clip(median, 0, 255).astype(np.uint8)
    composite = Image.fromarray(median)

    if debug_dir:
        save_image(composite, os.path.join(debug_dir, "combined_median.jpg"))

    return composite


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
        if getattr(lm, 'visibility', 1.0) < 0.2:
            continue
        x, y = lm_to_xy(lm)
        r = max(2, int(size[0] / 200))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 0))

    # draw skeleton edges
    for a, b in skeleton_edges:
        if a < len(landmarks) and b < len(landmarks):
            la = landmarks[a]
            lb = landmarks[b]
            if getattr(la, 'visibility', 1.0) < 0.2 or getattr(lb, 'visibility', 1.0) < 0.2:
                continue
            ax, ay = lm_to_xy(la)
            bx, by = lm_to_xy(lb)
            draw.line((ax, ay, bx, by), fill=(0, 0, 0), width=max(2, int(size[0] / 150)))

    pose.close()
    return canvas
