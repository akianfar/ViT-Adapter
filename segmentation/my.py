import sys
sys.path.append(r'segmentation/mmcv_custom')
sys.path.append(r'segmentation/mmseg_custom')
import mmcv

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes

CONFIG = r'segmentation/configs/my/my_city.py'
CHECKPOINT = r'~/iter_20000.pth'
PALETTE = \
    ([ 148, 218, 255 ],  # light blue
    [  85,  85,  85 ],  # almost black
    [ 200, 219, 190 ],  # light green
    [ 166, 133, 226 ],  # purple    
    [ 255, 171, 225 ],  # pink
    [  40, 150, 114 ],  # green
    [ 234, 144, 133 ],  # orange
    [  89,  82,  96 ],  # dark gray
    [ 255, 255,   0 ],  # yellow
    [ 110,  87, 121 ],  # dark purple
    [ 205, 201, 195 ],  # light gray
    [ 212,  80, 121 ],  # medium red
    [ 159, 135, 114 ],  # light brown
    [ 102,  90,  72 ],  # dark brown
    [ 255, 255, 102 ],  # bright yellow
    [ 251, 247, 240 ])  # almost white
    
class MyModel:
    def __init__(self):
        pass
           
    def segment_single_image(self, img):
        print("init model")
        model = init_segmentor(CONFIG, checkpoint=None, device='cuda:0')
        print("load checkpoint")
        checkpoint = load_checkpoint(model, CHECKPOINT, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = get_classes(PALETTE)
        print("inference segmentor")
        result = inference_segmentor(model, img)
        return result