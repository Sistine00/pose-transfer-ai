"""Command-line interface for PoseTransfer

Example:
python pose_transfer_cli.py \
  --target examples/target_person.jpg \
  --reference examples/reference_pose.jpg \
  --output out.jpg \
  --prompt "A young woman wearing the original clothes"
"""
import argparse
from pathlib import Path
import torch

from pose_transfer import PoseTransfer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, help="Path to target person image")
    p.add_argument("--reference", required=True, help="Path to reference pose image")
    p.add_argument("--output", default="output.jpg", help="Output image path")
    p.add_argument("--prompt", default="person", help="Text prompt describing the target person")
    p.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    p.add_argument("--steps", type=int, default=30, help="Inference steps")
    p.add_argument("--scale", type=float, default=7.5, help="Guidance scale")
    p.add_argument("--height", type=int, default=512, help="Output height")
    p.add_argument("--width", type=int, default=512, help="Output width")
    return p.parse_args()


def main():
    args = parse_args()
    args.output = Path(args.output)

    pt = PoseTransfer(device=args.device)

    result = pt.transfer_pose(
        target_image=args.target,
        reference_image=args.reference,
        prompt=args.prompt,
        num_steps=args.steps,
        guidance_scale=args.scale,
        output_size=(args.width, args.height),
    )

    result.save(str(args.output))
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
