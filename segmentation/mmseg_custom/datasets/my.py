from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class MyDataset(CustomDataset):
    """My dataset.
    """
    CLASSES = ('WATER', 'ASPHALT', 'GRASS', 'HUMAN', 'ANIMAL', 'HIGH_VEGETATION', 
               'GROUND_VEHICLE', 'FAÇADE', 'WIRE', 'GARDEN_FURNITURE', 'CONCRETE', 
               'ROOF', 'GRAVEL', 'SOIL', 'PRIMEAIR_PATTERN', 'SNOW')

    PALETTE = [[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
               [180, 165, 180], [90, 120, 150], [
                   102, 102, 156], [128, 64, 255],
               [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96],
               [230, 150, 140], [128, 64, 128], [
                   110, 110, 110], [244, 35, 232]]

    def __init__(self, **kwargs):
        super(MyDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)