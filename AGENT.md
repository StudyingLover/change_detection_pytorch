# Change Detection PyTorch Project Design

## 项目概述

这是一个基于 PyTorch 的变化检测（Change Detection）深度学习框架，参考 CASP 项目架构，支持多种分割模型和数据集。

## 目录结构

```
change_detection_pytorch/
├── core/               # 底层组件库（可复用基础模块）
├── models/             # 常用分割模型（用 core 搭建成品）
├── applications/       # 完整解决方案（用 core + models 组合）
├── datasets/           # 数据集加载器
└── ...
```

## 分层设计原则

### 1. core/ - 底层组件库

**定位**：最底层，提供可复用的基础模块，不依赖其他内部模块。

```
core/
├── base/          # 共享基类（SegmentationModel, Decoder, Heads 等）
├── encoders/      # 编码器/骨干网络（ResNet, MixVisionTransformer 等）
├── ops/           # 可选算子（DCNv3 等高级算子，非必需）
├── losses/        # 损失函数（Dice, Focal, BCE 等）
└── utils/         # 工具函数（指标、调度器等）
```

- `base/` - 所有模型的父类定义
- `encoders/` - 特征提取器，被 `models/` 内所有模型使用
- `ops/` - 可选的高级算子，按需引入（如 DCNv3 用于 CASP）
- `losses/` - 共享损失函数
- `utils/` - 共享工具函数

### 2. models/ - 独立分割模型

**定位**：中间层，基于 core 搭建的独立分割模型，可直接使用。

```
models/
├── unet/           # U-Net
├── unetplusplus/   # U-Net++
├── deeplabv3/      # DeepLabV3, DeepLabV3+
├── fpn/            # Feature Pyramid Network
├── pspnet/         # Pyramid Scene Parsing Network
├── linknet/        # LinkNet
├── manet/          # MAnet
├── pan/            # PAN
├── upernet/        # UPerNet
└── stanet/        # STANet
```

**依赖关系**：
```
models/* = core/base + core/encoders + core/losses + core/utils
```

### 3. applications/ - 完整解决方案

**定位**：最上层，基于 core 和 models 组合的完整模型/方案。

```
applications/
└── casp/           # CASP 模型（Change Detection with Spatial Perception）
```

**依赖关系**：
```
applications/casp = core/encoders + core/ops + core/base
                = MixVisionTransformer + DCNv3 + SAFM + SPAligned + DCAlign
```

## 模块导入规范

### 从内部模块导入

```python
# 编码器
from change_detection_pytorch.core.encoders import get_encoder
from change_detection_pytorch.core.encoders.mix_transformer import MixVisionTransformer

# 算子
from change_detection_pytorch.core.ops.dcnv3.modules import DCNv3

# 基类
from change_detection_pytorch.core.base import SegmentationModel, Decoder

# 损失函数
from change_detection_pytorch.core.losses import DiceLoss

# 独立模型（通过顶层包）
from change_detection_pytorch import Unet, DeepLabV3, FPN

# 组合模型（通过 applications，自动注册）
from change_detection_pytorch.applications import get_model
model = get_model('casp', in_ch=3, pretrained=True)
```

### 模型内部导入

```python
# models/ 内模块导入 core 组件（使用绝对路径）
from change_detection_pytorch.core.encoders import get_encoder
from change_detection_pytorch.core.base import SegmentationModel

# applications/ 内模块导入 core 组件
from change_detection_pytorch.core.ops.dcnv3.modules import DCNv3
from change_detection_pytorch.core.encoders.mix_transformer import MixVisionTransformer
```

## 数据集结构

```
datasets/
├── __init__.py           # 导出所有数据集类
├── custom.py             # 基础自定义数据集类
├── transforms/           # 数据增强
├── LEVIR_CD.py           # LEVIR-CD 数据集
├── WHU_CD.py             # WHU-CD 数据集
├── GZ_CD.py              # GZ-CD 数据集
└── SVCD.py              # SVCD 数据集
```

**数据集使用示例**：
```python
from change_detection_pytorch.datasets import LEVIR_CD_Dataset, WHU_CD_Dataset, GZ_CD_Dataset

train_dataset = LEVIR_CD_Dataset(
    img_dir='path/to/train',
    ann_dir='path/to/train/label',
    size=256
)
```

## 训练流程

使用 `train_app.py` 训练已注册模型：

```bash
# 训练 CASP 模型
python train_app.py --model casp --dataset WHU-CD --data_dir ./data

# 训练其他模型
python train_app.py --model unet --dataset LEVIR-CD --encoder_name resnet50
```

查看可用模型：
```python
from change_detection_pytorch.applications import list_models
print(list_models())  # ['casp', ...]
```

## 添加新模块指南

### 添加新算子到 ops/

1. 在 `core/ops/` 下创建新目录
2. 实现算子模块
3. 在 `core/ops/__init__.py` 中导出

### 添加新模型到 models/

1. 在 `models/` 下创建新目录
2. 实现模型（继承 `core.base.SegmentationModel`）
3. 在 `change_detection_pytorch/__init__.py` 中添加导出

### 添加新应用到 applications/

**使用装饰器自动注册：**

1. 在 `applications/` 下创建新目录 `your_model/`
2. 在 `model.py` 中用 `@register_model('your_model')` 装饰器注册模型类

```python
# applications/your_model/model.py
from change_detection_pytorch.applications import register_model

@register_model('your_model')
class YourModel(nn.Module):
    def __init__(self, in_ch=3, pretrained=False):
        ...
```

**无需修改任何 `__init__.py`**，自动扫描会注册模型。

**使用：**
```python
from change_detection_pytorch.applications import get_model
model = get_model('your_model', in_ch=3, pretrained=True)
```

## 设计原则总结

1. **分层清晰**：`core` < `models` < `applications`，依赖关系单向向上
2. **可复用性**：`core` 作为基础库，被多个模块共享
3. **可选性**：`core/ops/` 是可选的高级算子，非必需组件
4. **独立性**：`models/` 下的模型可独立使用，不依赖 `applications/`
5. **组合性**：`applications/` 基于 `core` 和 `models` 组合构建

## 技术栈

- **框架**：PyTorch
- **算子**：DCNv3（来自 InternImage 的可变形卷积）
- **编码器**：MixVisionTransformer（SegFormer backbone）、ResNet 系列等
- **数据增强**：albumentations
