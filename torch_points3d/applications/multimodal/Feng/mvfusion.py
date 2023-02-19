import logging
from abc import ABC
import time

import numpy as np
import torch
from torch_points3d.models.base_architectures.backbone import MVFusionBackboneBasedModel
from torch_points3d.applications.utils import extract_output_nc
from torch_points3d.core.common_modules.base_modules import MLP
from torch_points3d.core.multimodal.data import MMData
from torch_geometric.data import Batch
from torch.utils.checkpoint import checkpoint

### Feng
from torch_points3d.modules.multimodal.pooling import BimodalCSRPool
import sys
sys.path.append("/home/fsun/thesis/modeling")
from compare_methods import DVA_cls_5_fusion_7


log = logging.getLogger(__name__)


class MVFusionEncoder(MVFusionBackboneBasedModel, ABC):
    """Encoder structure for multimodal models without 3D data.

    Inspired from torchpoints_3d.applications.sparseconv3d.
    
    
    Feng modification: this structure creates multi-view fused features for each 3D point,
        based on its Mask2Former view-wise predictions and the viewing conditions.
    """

    def __init__(
            self, model_config, model_type, dataset, modules, *args, **kwargs):
        # UnwrappedUnetBasedModel init
        super(MVFusionEncoder, self).__init__(
            model_config, model_type, dataset, modules)
        
        # Make sure the model is multimodal and has no 3D. Note that
        # the BackboneBasedModel.__init__ carries most of the required
        # initialization.
        assert self.is_multimodal, \
            f"No3DEncoder should carry at least one non-3D modality."
        assert self.no_3d_conv, \
            f"No3DEncoder should not have 3D-specific modules."

        # Recover size of output features
        default_output_nc = kwargs.get("default_output_nc", None)
        if not default_output_nc:
            mod_out_nc_list = [extract_output_nc(getattr(
                model_config.down_conv, m)) for m in self.modalities]
            assert all(o == mod_out_nc_list[0] for o in mod_out_nc_list), \
                f"Expected all modality branches outputs to have the same " \
                f"feature size but got {mod_out_nc_list} sizes instead."
            default_output_nc = mod_out_nc_list[0]
            
        self._output_nc = default_output_nc
        self._checkpointing = model_config['backbone']['transformer']['checkpointing']

        # Set the MLP head if any
#         self._has_mlp_head = True
#         if "output_nc" in kwargs:
#             self._has_mlp_head = True
#         print("self._has_mlp_head is true by default in applications/.../mvfusion.py")
#             self._output_nc = kwargs["output_nc"]

#         # Manually adjusted MLP size because the MVFusion config without 3d does not have output_nc 
#         self.mlp = MLP(
#             [model_config['backbone']['transformer']['embed_dim'], model_config['backbone']['transformer']['n_classes']], activation=torch.nn.Identity(model_config['backbone']['transformer']['embed_dim']),
#             bias=False)
            
        # modules
        self.transformerfusion = DVA_cls_5_fusion_7(model_config['backbone']['transformer'])
        
        self.n_views = model_config.backbone.transformer.n_views
        self.n_classes = model_config.backbone.transformer.n_classes
        self.MAX_SEEN_POINTS = model_config.backbone.transformer.max_n_points
        
        print("WARNING: input points clipped at ", self.MAX_SEEN_POINTS, flush=True)

        if self._checkpointing:
            print("checkpointing enabled for model forward pass!", flush=True)
        
    @property
    def has_mlp_head(self):
        return self._has_mlp_head

    @property
    def output_nc(self):
        return self._output_nc


    def forward(self, data, *args, **kwargs):
        """Run forward pass. Expects a MMData object for input, with
        3D Data and multimodal data and mappings. Although the
        No3DEncoder model does not apply any convolution modules
        directly on the 3D points, it still requires a 3D points Data
        object with a 'pos' attribute as input, to be able to output
        these very same points populated with modality-generated
        features.

        Parameters
        -----------
        data: MMData object

        Returns
        --------
        data: Data object
            - pos [N, 3] (coords or real pos if xyz is in data)
            - x [N, output_nc]
        """

        """  Feng: temporarily skip this as we do not use 3d data, and we only have 1 ImageSetting
        # Apply ONLY atomic-level pooling which is in `down_modules`
        for i in range(len(self.down_modules)):
            mm_data_dict = self.down_modules[i](mm_data_dict)
        """    
        ### multi-view mapping & M2F feature fusion using Transformer 
        
        # Features from only seen point-image matches are included in 'x'
        viewing_feats = data.data.mvfusion_input[:, :, :-1]
        m2f_feats = data.data.mvfusion_input[:, :, -1]
                
        # One hot features of M2F preds
        m2f_feats = torch.nn.functional.one_hot(m2f_feats.squeeze().long(), self.n_classes)
    
        ### Multi-view fusion of M2F and viewing conditions using Transformer
        # TODO: remove assumption that pixel validity is the 1st feature
        invalid_pixel_mask = viewing_feats[:, :, 0] == 0  
            
            
        # get logits
        if self._checkpointing:
            out_scores = checkpoint(self.transformerfusion, invalid_pixel_mask, viewing_feats.requires_grad_(), m2f_feats)
        else:
            out_scores = self.transformerfusion(invalid_pixel_mask, viewing_feats, m2f_feats)
            
        
        csr_idx = data.modalities['image'][0].view_csr_indexing
        x_seen_mask = csr_idx[1:] > csr_idx[:-1]
        
        
        # Discard the modalities used in the down modules, only
        # 3D point features are expected to be used in subsequent
        # modules. Restore the input Data object equipped with the
        # proper point positions and modality-generated features.
        
        out = Batch(
            x=out_scores, 
            pos=data.pos.to(self.device), 
            seen=x_seen_mask.to(self.device))
        out=out.to(self.device)
        

        # TODO: this always passes the modality feature maps in the
        #  output dictionary. May not be relevant at inference time,
        #  when we would rather save memory and discard unnecessary
        #  data. Besides, this behavior for NoDEncoder is not consistent
        #  with the multimodal UNet, need to consider homogenizing.
        for m in self.modalities:
            # Feng:
            out[m] = data.modalities[m]

#         # Apply the MLP head, if any
#         if self.has_mlp_head:
#             # Apply to the 3D features
#             out.x = self.mlp(out.x)

        return out


