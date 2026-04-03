"""
Applications 模块 - 组合模型/完整应用自动注册

用法:
    from change_detection_pytorch.applications import get_model

    # 在 model.py 中用装饰器注册
    @register_model('casp')
    class CASP(nn.Module):
        ...

    # 训练脚本中通过字符串加载
    model = get_model('casp', in_ch=3, pretrained=True)
"""

from typing import Dict, Type
import os
import importlib
import sys

# 注册表
_MODELS: Dict[str, Type] = {}


def register_model(name: str):
    """装饰器：注册模型到全局注册表

    用法:
        @register_model('casp')
        class CASP(nn.Module):
            ...
    """
    def decorator(cls):
        _MODELS[name] = cls
        return cls
    return decorator


def get_model(name: str, **kwargs):
    """根据名称获取已注册的模型

    Args:
        name: 模型名称 (如 'casp')
        **kwargs: 传递给模型的参数

    Returns:
        模型实例

    Raises:
        KeyError: 如果模型名称未注册
    """
    if name not in _MODELS:
        raise KeyError(
            f"Model '{name}' not found. Available models: {list(_MODELS.keys())}"
        )
    return _MODELS[name](**kwargs)


def list_models():
    """返回所有已注册的模型名称"""
    return list(_MODELS.keys())


# 自动扫描并导入子模块，触发装饰器注册
_APPS_DIR = os.path.dirname(os.path.abspath(__file__))
for _item in os.listdir(_APPS_DIR):
    _item_path = os.path.join(_APPS_DIR, _item)
    if os.path.isdir(_item_path) and _item not in ('__pycache__',):
        # 导入子模块的 __init__.py
        _module_name = f"{__name__}.{_item}"
        importlib.import_module(_module_name)
        # 尝试导入 model.py（子模块的主要模型文件）
        try:
            importlib.import_module(f"{_module_name}.model")
        except ImportError:
            pass
