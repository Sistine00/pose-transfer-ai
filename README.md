# Pose Transfer AI

一个强大的AI工具，可以让一个人物摆出另一个人物的姿态。使用 Stable Diffusion 和 ControlNet 技术生成高质量的人物姿态转移图像。

## 安装与注意事项

- requirements.txt 已在 repo 中列出依赖。重要：请将 mediapipe 安装为 0.10.14，因为 1.0.0+ 的版本移除了 `mp.solutions` API，本项目依赖该 API。

## 使用示例（更新）

Python API 示例：

```python
from pose_transfer import PoseTransfer

pt = PoseTransfer(device="cuda")
result = pt.transfer_pose(
    target_images=["path/to/target1.jpg", "path/to/target2.jpg"],  # 支持单张字符串或多张路径列表
    reference_image="path/to/reference_pose.jpg",
    prompt="一个年轻女性，穿着原始衣服",
)
result.save("out.jpg")
```

命令行示例：

```bash
python pose_transfer_cli.py \
  --targets target1.jpg target2.jpg \
  --reference reference_pose.jpg \
  --output result.jpg \
  --prompt "一个年轻女性" \
  --debug --debug-dir examples/debug
```

