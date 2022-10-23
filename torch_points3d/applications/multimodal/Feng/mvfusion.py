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

        # Set the MLP head if any
        self._has_mlp_head = False
        if "output_nc" in kwargs:
            self._has_mlp_head = True
            print("self._has_mlp_head is true in applications/.../mvfusion.py")
            self._output_nc = kwargs["output_nc"]
            self.mlp = MLP(
                [default_output_nc, self.output_nc], activation=torch.nn.ReLU(),
                bias=False)
            
        # modules
        self.fusion = DVA_cls_5_fusion_7(model_config['transformer'])
        
        self.n_views = model_config['transformer'].n_views
        self.n_classes = model_config['transformer']['n_classes']
        
    @property
    def has_mlp_head(self):
        return self._has_mlp_head

    @property
    def output_nc(self):
        return self._output_nc

#     def _set_input(self, data: MMData):
#         """Unpack input data from the dataloader and perform necessary
#         pre-processing steps.

#         Parameters
#         -----------
#         data: MMData object
#         """
# #         print("temprarily skip moving data to device in applications")
# #         print("data before to device", data)
#         data = data.to(self.device)
# #         print("data after to device: ", data)
        
#         self.input = {
#             'x_3d': getattr(data, 'x', None),   # Feng: adjusted from original 'getattr(data, 'x', None)', now it properly moves to gpu
#             'x_seen': None,
#             'modalities': data.modalities}
#         if data.pos is not None:
#             self.xyz = data.pos

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
           
        ### Feng: this has been moved to dataset __getitem__
#         # Take subset of only seen points
#         # idx mapping from each pixel to point
#         # NOTE: each point is contained multiple times if it has multiple correspondences
#         im_data = data.modalities['image']
#         dense_idx_list = [
#                     torch.arange(im.num_points, device=im_data.device).repeat_interleave(
#                         im.view_csr_indexing[1:] - im.view_csr_indexing[:-1])
#                     for im in im_data]
#         # take subset of only seen points without re-indexing the same point
#         ### TODO: this converts data from MMBatch to MMData class when slicing
#         ### Does this cause any errors downstream?
#         data = data[dense_idx_list[0].unique()]

        
        """  Feng: temporarily skip this as we do not use 3d data, and we only have 1 ImageSetting
        # Apply ONLY atomic-level pooling which is in `down_modules`
        for i in range(len(self.down_modules)):
            mm_data_dict = self.down_modules[i](mm_data_dict)
        """
    
#         # Gather 9 (valid and invalid) viewing conditions for each point
#         # Invalid viewing conditions serve as padding
        
#         s = time.time()
#         viewing_feats, m2f_feats = self.get_view_dependent_features(data)
#         print(time.time() - s, flush=True)

#         # Mask2Former predictions per view as feature
#         # Adjust previously used label mapping [0, 21] with 0 being invalid, to [-1, 20].
#         # As M2F model does not produce 0 preds, updated labels are within [0, 19]
#         m2f_feats = m2f_feats - 1   
        
    
        print("data.x.shape in mvfusion:", data.x.shape, flush=True)
        viewing_feats = data.x[:-1]
        m2f_feats = data.x[-1:]
        m2f_feats = torch.nn.functional.one_hot(m2f_feats.squeeze().long(), self.n_classes)
    
        ### Multi-view fusion of M2F and viewing conditions using Transformer
        # TODO: remove assumption that pixel validity is the 1st feature
        invalid_pixel_mask = viewing_feats[:, :, 0] == 0.
        
        fusion_input = {
            'invalid_pixels_mask': invalid_pixel_mask.to(self.device),
            'viewing_features': viewing_feats.to(self.device),
            'one_hot_mask_labels': m2f_feats.to(self.device)
        }
                        
        # get logits
        out_scores = self.fusion(fusion_input)
        
        csr_idx = data.modalities['image'][0].view_csr_indexing
            
        # Discard the modalities used in the down modules, only
        # 3D point features are expected to be used in subsequent
        # modules. Restore the input Data object equipped with the
        # proper point positions and modality-generated features.
        csr_idx = data.modalities['image'][0].view_csr_indexing
        out = Batch(
            x=out_scores, 
            pos=data.pos.to(self.device), 
            seen=(csr_idx[1:] > csr_idx[:-1]).to(self.device))
        out=out.to(self.device)
        

        # TODO: this always passes the modality feature maps in the
        #  output dictionary. May not be relevant at inference time,
        #  when we would rather save memory and discard unnecessary
        #  data. Besides, this behavior for NoDEncoder is not consistent
        #  with the multimodal UNet, need to consider homogenizing.
        for m in self.modalities:
            # x_mod = getattr(mm_data_dict['modalities'][m], 'last_view_x_mod', None)
            # if x_mod is not None:
            #     out[m] = mm_data_dict['modalities'][m]
            #####out[m] = mm_data_dict['modalities'][m]
            
            # Feng:
            out[m] = data.modalities[m]

        # Apply the MLP head, if any
        if self.has_mlp_head:
            # Apply to the 3D features
            out.x = self.mlp(out.x)

            # Apply to the last modality-based view-level features
            for m in [mod for mod in self.modalities if mod in out.keys]:
                out[m].last_view_x_mod = self.mlp(out[m].last_view_x_mod)

        return out

    def get_view_dependent_features(self, mm_data):
        n_views = self.n_views

        image_data = mm_data.modalities['image']
        csr_idx = image_data.view_cat_csr_indexing

        viewing_conditions = image_data[0].mappings.values[2]
        
        assert len(image_data) == 1
        m2f_mapped_feats = image_data[0].get_mapped_m2f_features(interpolate=True)
        
        

        # Add pixel validity as first feature
        viewing_conditions = torch.cat((torch.ones(viewing_conditions.shape[0], 1).to(viewing_conditions.device),
                                        viewing_conditions), dim=1)

        # Calculate amount of empty views. There should be n_points * n_views filled view conditions in total.
        n_seen = csr_idx[1:] - csr_idx[:-1]
        unfilled_points = n_seen[n_seen < n_views]
        n_views_to_fill = int(len(unfilled_points) * n_views - sum(unfilled_points))

        # generate random viewing conditions
        random_invalid_views = viewing_conditions[np.random.choice(range(len(viewing_conditions)), size=n_views_to_fill, replace=True)]
        # set pixel validity to invalid
        random_invalid_views[:, 0] = 0
        random_m2f_preds = m2f_mapped_feats[np.random.choice(range(len(viewing_conditions)), size=n_views_to_fill, replace=True)]


        # concat viewing conditions and random invalid views, then index the tensor such that each point
        # either has 9 valid subsampled views, or is filled to 9 views with random views
        combined_tensor = torch.cat((viewing_conditions, random_invalid_views), dim=0)
        combined_m2f_tensor = torch.cat((m2f_mapped_feats, random_m2f_preds), dim=0)
        
        
        unused_invalid_view_idx = len(viewing_conditions)
        combined_idx = []
        for i, n in enumerate(n_seen):
            if n < n_views:
                n_empty_views = n_views -  n
                combined_idx += list(range(csr_idx[i], csr_idx[i+1])) + \
                                list(range(unused_invalid_view_idx, unused_invalid_view_idx + n_empty_views))
                unused_invalid_view_idx += n_empty_views
            elif n > n_views:
                sampled_idx = sorted(np.random.choice(range(csr_idx[i], csr_idx[i+1]), size=n_views, replace=False))
                combined_idx += sampled_idx
            else:
                combined_idx += list(range(csr_idx[i], csr_idx[i+1]))

        # re-index tensor for MVFusion format
        combined_tensor = combined_tensor[combined_idx]
        combined_m2f_tensor = combined_m2f_tensor[combined_idx]
        
        return combined_tensor.reshape(mm_data.num_points, n_views, -1), combined_m2f_tensor.reshape(mm_data.num_points, n_views)
