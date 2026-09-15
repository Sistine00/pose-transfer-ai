# Pose Transfer AI

一个强大的AI工具，可以让一个人物摆出另一个人物的姿态。使用 Stable Diffusion 和 ControlNet 技术生成高质量的人物姿态转移图像。

## 功能特性

✨ **姿态检测** - 从参考照片自动提取人物姿态
✨ **姿态转移** - 将参考姿态应用到目标人物
✨ **高质量生成** - 基于 Stable Diffusion 生成逼真的图像
✨ **简单易用** - 简洁的 Python API 和命令行工具

## 系统要求

- Python 3.8+
- CUDA 11.0+ (NVIDIA GPU) 或 CPU (较慢)
- 显存 6GB+ (推荐 8GB+)
- 8GB+ RAM

## 安装

### 1. 克隆项目
```bash
git clone https://github.com/Sistine00/pose-transfer-ai.git
cd pose-transfer-ai
```

### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

## 快速开始

### 准备图像
1. 准备两张图片：
   - `target_person.jpg` - 目标人物照片
   - `reference_pose.jpg` - 参考姿态照片

### 使用 Python API

```python
from pose_transfer import PoseTransfer

# 初始化
pt = PoseTransfer(device="cuda")  # 或 "cpu"

# 加载图像
target_image = "path/to/target_person.jpg"
reference_image = "path/to/reference_pose.jpg"

# 执行姿态转移
result = pt.transfer_pose(
    target_image=target_image,
    reference_image=reference_image,
    prompt="一个年轻女性，穿着原始衣服"
)

# 保存结果
result.save("output_pose_transfer.jpg")
```

### 使用命令行

```bash
python pose_transfer_cli.py \
    --target target_person.jpg \
    --reference reference_pose.jpg \
    --output result.jpg \
    --prompt "一个年轻女性"
```

## 工作原理

1. **姿态检测** - 使用 OpenPose 从参考图像提取人物骨骼信息
2. **姿态图生成** - 将骨骼信息转换为可视化姿态图
3. **条件生成** - 使用 ControlNet 根据姿态图和目标人物生成新图像
4. **图像优化** - 混合原始纹理和生成的内容，确保自然效果

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `target_image` | 目标人物照片路径 | 必需 |
| `reference_image` | 参考姿态照片路径 | 必需 |
| `prompt` | 人物描述提示词 | "person" |
| `output` | 输出图像路径 | "output.jpg" |
| `device` | 计算设备 | "cuda" |
| `num_steps` | 生成步数(越多越慢但质量更好) | 50 |
| `guidance_scale` | 提示词引导强度 | 7.5 |

## 常见问题

### Q: 显存不足怎么办？
A: 可以尝试以下方法：
- 减小图像分辨率
- 减少 `num_steps` 参数
- 使用 `--enable-attention-slicing` 选项
- 更新显卡驱动

### Q: 生成速度太慢？
A: 
- 使用 GPU 加速（NVIDIA CUDA）
- 减少 `num_steps` 参数
- 使用更小的模型版本

### Q: 生成效果不理想？
A:
- 调整 `prompt` 参数，更详细描述人物
- 增加 `num_steps` 获得更好质量
- 确保输入图像清晰，照明均匀
- 提高参考照片中姿态的清晰度

## 项目结构

```
pose-transfer-ai/
├── pose_transfer.py          # 核心模块
├── pose_transfer_cli.py      # 命令行工具
├── models/                   # 预训练模型
├── utils/                    # 工具函数
├── examples/                 # 使用示例
├── requirements.txt          # 依赖列表
└── README.md                 # 本文件
```

## 示例

查看 `examples/` 目录获取更多使用示例。

## 许可证

MIT License - 详见 LICENSE 文件

## 贡献

欢迎提交 Issue 和 Pull Request！

## 相关资源

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)

## 联系方式

有问题？欢迎在 GitHub Issues 中提问！
