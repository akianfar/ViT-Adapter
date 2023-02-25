import sys
sys.path.append(r'segmentation/mmcv_custom')
sys.path.append(r'segmentation/mmseg_custom')
from .core import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .models import *  # noqa: F401,F403
