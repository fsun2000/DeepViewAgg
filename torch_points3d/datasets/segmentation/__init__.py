IGNORE_LABEL: int = -1

from .shapenet import ShapeNet, ShapeNetDataset
from .s3dis import S3DISFusedDataset, S3DIS1x1Dataset, S3DISOriginalFused, S3DISSphere
from .scannet import ScannetDataset, Scannet
from .scannet_inference import ScannetDataset_Inference, Scannet_Inference
from .kitti360 import KITTI360Dataset
