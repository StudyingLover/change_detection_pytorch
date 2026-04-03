# Lazy import to avoid loading all encoders (which require pretrainedmodels)
# Users should import specific models directly: from change_detection_pytorch.models.unet import Unet

from .core import encoders
from .core import utils
from .core import losses
from . import datasets

from .__version__ import __version__

from typing import Optional
import torch
