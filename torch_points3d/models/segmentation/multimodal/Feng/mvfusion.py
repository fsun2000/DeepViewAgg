import logging
from abc import ABC

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.applications.multimodal.Feng.mvfusion import MVFusionEncoder

# from sklearn.neighbors import NearestNeighbors
from pykeops.torch import LazyTensor

log = logging.getLogger(__name__)


class MVFusion(BaseModel, ABC):

    _MODALITY_VIEW_LOSS = None

    def __init__(self, option, model_type, dataset, modules):
#         # No3D should not be directly instantiated, child classes should
#         # be used instead
#         if not hasattr(self, '_HAS_HEAD'):
#             raise NotImplementedError

        # BaseModel init
        super().__init__(option)

        # UnwrappedUnetBasedModel init
        option['backbone']['transformer']['n_classes'] = dataset.num_classes
        self.backbone = MVFusionEncoder(option, model_type, dataset, modules)
        self._modalities = self.backbone._modalities

        # Segmentation head init
        if self._HAS_HEAD:
            if option['backbone']['transformer']['feat_downproj_dim'] is not None:
                self.head = nn.Linear(option['backbone']['transformer']['feat_downproj_dim'],
                                                    dataset.num_classes)
            else:
                self.head = nn.Linear(option['backbone']['transformer']['embed_dim'],
                                                    dataset.num_classes)
        self.loss_names = ["loss_seg"]

        self.MAX_SEEN_POINTS = option['backbone']['transformer']['max_n_points']

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
        
        if data.data.mvfusion_input.shape[0] > self.MAX_SEEN_POINTS \
            and self.training is True:
            print("self.training is True -> culling max n seen points", flush=True)
            # 1. get seen points
            # 2. remove them from mvfusion_input
            # 3. remove the removed points from seen points
            csr_idx = data.modalities['image'][0].view_csr_indexing
            seen_mask = csr_idx[1:] > csr_idx[:-1]
            keep_idx = torch.round(
                torch.linspace(0, seen_mask.sum()-1, self.MAX_SEEN_POINTS)).long()
            keep_idx_mask = torch.zeros(seen_mask.sum(), dtype=torch.bool, device=keep_idx.device)
            keep_idx_mask[keep_idx] = True
            seen_mask[seen_mask.clone()] = keep_idx_mask
            
            # Take slice
            data = data[keep_idx_mask]
            
    
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

        
    
        # Feng: directly use the logits output from DVA_cls_5_fusion_7 class with MLP_head inside
        logits = self.head(features) if self._HAS_HEAD else features
        self.output = F.log_softmax(logits, dim=-1)

        if not self.training and not torch.all(seen_mask):    # Skip if all points were seen
            if torch.any(seen_mask):
                print("Currently propagating predictions to unseen points in validation run")
                
                
                # If the module is in eval mode, propagate the output of the
                # nearest seen point to unseen points
                # nn_search = NearestNeighbors(
                #     n_neighbors=1, algorithm="kd_tree").fit(
                #     data.pos[seen_mask].detach().cpu().numpy())
                # _, nn_idx = nn_search.kneighbors(
                #     data.pos[~seen_mask].detach().cpu().numpy())
                # nn_idx = torch.LongTensor(nn_idx)

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
                

#             # Based on a modality's view-wise predictions
#             else:
#                 view_features = data[self._MODALITY_VIEW_LOSS].last_view_x_mod
#                 csr_idx = data[self._MODALITY_VIEW_LOSS].last_view_csr_idx
#                 view_logits = self.head(view_features) if self._HAS_HEAD \
#                     else view_features
#                 pred = F.log_softmax(view_logits, dim=-1)
#                 target = torch.repeat_interleave(self.labels,
#                     csr_idx[1:] - csr_idx[:-1])


            
            self.loss_seg = F.nll_loss(pred, target, ignore_index=IGNORE_LABEL)

    def backward(self):
        self.loss_seg.backward()


class MVFusion_model(MVFusion):
    _HAS_HEAD = True
    

class MVFusion_model_no_head(MVFusion):
    _HAS_HEAD = False


# class No3DLogitFusion(No3D):
#     _HAS_HEAD = False


# class No3DImageFeatureFusion(No3D):
#     _HAS_HEAD = True
#     _MODALITY_VIEW_LOSS = 'image'


# class No3DImageLogitFusion(No3D):
#     _HAS_HEAD = False
#     _MODALITY_VIEW_LOSS = 'image'
