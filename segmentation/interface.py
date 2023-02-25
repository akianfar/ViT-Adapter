# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
import cv2
import os.path as osp

CONFIG = r'my_submission/my/segmentation/configs/my/my_city.py'
CHECKPOINT = r'my_submission/my/segmentation/work_dirs/my_city/iter_20000.pth'
       
def segment_single_image(img):
    model = init_segmentor(CONFIG, checkpoint=None, device='cuda:0')
    checkpoint = load_checkpoint(model, CHECKPOINT, map_location='cpu')
    result = inference_segmentor(model, img)
    return result