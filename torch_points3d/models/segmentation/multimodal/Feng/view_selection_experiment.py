import logging
from abc import ABC

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL
# from torch_points3d.applications.multimodal.Feng.mvfusion import MVFusionEncoder
from torch_points3d.applications.multimodal.Feng.view_selection_experiment import ViewSelectionExpEncoder

# from sklearn.neighbors import NearestNeighbors
from pykeops.torch import LazyTensor

log = logging.getLogger(__name__)


class ViewSelectionExp(BaseModel, ABC):

    _MODALITY_VIEW_LOSS = None

    def __init__(self, option, model_type, dataset, modules):
        # ViewSelectionExp should not be directly instantiated, child classes should
        # be used instead
        if not hasattr(self, '_HAS_HEAD'):
            raise NotImplementedError

        # BaseModel init
        super().__init__(option)

        # UnwrappedUnetBasedModel init
        option['backbone']['transformer']['n_classes'] = dataset.num_classes
        self.backbone = ViewSelectionExpEncoder(option, model_type, dataset, modules)
        self._modalities = self.backbone._modalities

        # Segmentation head init
        if self._HAS_HEAD:
            self.head = nn.Sequential(nn.Linear(dataset.num_classes,
                                                dataset.num_classes))
        self.loss_names = ["loss_seg"]


    def get_seen_points(self, mm_data):
        ### Select seen points
        csr_idx = mm_data.modalities['image'][0].view_csr_indexing
        dense_idx_list = torch.arange(mm_data.modalities['image'][0].num_points).repeat_interleave(csr_idx[1:] - csr_idx[:-1])
        # take subset of only seen points without re-indexing the same point
        mm_data = mm_data[dense_idx_list.unique()]
        return mm_data

    def set_input(self, data, device):
        # Get only seen points
        data = self.get_seen_points(data)
    
        self.input = data.to(self.device)

        if hasattr(data, 'batch') and data.batch is not None:
            self.batch_idx = data.batch.squeeze()
        else:
            self.batch_idx = None

        if hasattr(data, 'y') and data.y is not None:
            self.labels = data.y.to(self.device)
        else:
            self.labels = None

    def forward(self, *args, **kwargs):      
        data = self.backbone(self.input)
        features = data.x
        seen_mask = data.seen
        
        if features.device != self.device:
            features = features.to(self.device)
        
        if seen_mask is None:
            seen_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=self.device)

    
        logits = self.head(features) if self._HAS_HEAD else features
        self.output = F.log_softmax(logits, dim=-1)

        if not self.training and not torch.all(seen_mask):    # Skip if all points were seen
            if torch.any(seen_mask):
                print("Currently propagating predictions to unseen points in validation run")
                # If the module is in eval mode, propagate the output of the
                # nearest seen point to unseen points
                # K-NN search with KeOps
                xyz_query_keops = LazyTensor(data.pos[~seen_mask][:, None, :])
                xyz_search_keops = LazyTensor(data.pos[seen_mask][None, :, :])
                d_keops = ((xyz_query_keops - xyz_search_keops) ** 2).sum(dim=2)
                nn_idx = d_keops.argmin(dim=1)
                del xyz_query_keops, xyz_search_keops, d_keops

                self.output[~seen_mask] = self.output[seen_mask][nn_idx].squeeze()
            else:
                # If no points were seen, do not compute the loss on 
                # unseen data points
                self.labels[~seen_mask] = IGNORE_LABEL

        else:
            # If the module is in training mode, do not compute the loss
            # on the unseen data points
            self.labels[~seen_mask] = IGNORE_LABEL

        # Compute the segmentation loss
        if self.labels is not None:

            # Based on the 3D pointwise predictions
            if self._MODALITY_VIEW_LOSS is None:
                pred = self.output
                target = self.labels


            self.loss_seg = F.nll_loss(pred, target, ignore_index=IGNORE_LABEL)

    def backward(self):
        self.loss_seg.backward()


class ViewSelectionExp_model(ViewSelectionExp):
    _HAS_HEAD = True


# class No3DLogitFusion(No3D):
#     _HAS_HEAD = False


# class No3DImageFeatureFusion(No3D):
#     _HAS_HEAD = True
#     _MODALITY_VIEW_LOSS = 'image'


# class No3DImageLogitFusion(No3D):
#     _HAS_HEAD = False
#     _MODALITY_VIEW_LOSS = 'image'
