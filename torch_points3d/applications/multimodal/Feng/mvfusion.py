import logging
from abc import ABC

import torch
from torch_points3d.models.base_architectures.backbone import MVFusionBackboneBasedModel
from torch_points3d.applications.utils import extract_output_nc
from torch_points3d.core.common_modules.base_modules import MLP
from torch_points3d.core.multimodal.data import MMData
from torch_geometric.data import Batch

from torch_points3d.modules.multimodal.pooling import BimodalCSRPool


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
            
        # pooling modules
        self.atomic_pooling = BimodalCSRPool(mode='max', save_last=False)

    @property
    def has_mlp_head(self):
        return self._has_mlp_head

    @property
    def output_nc(self):
        return self._output_nc

    def _set_input(self, data: MMData):
        """Unpack input data from the dataloader and perform necessary
        pre-processing steps.

        Parameters
        -----------
        data: MMData object
        """
        print("temprarily skip moving data to device in applications")
        data = data#.to(self.device)

        self.input = {
            'x_3d': getattr(data.data, 'x', None),   # Feng: adjusted from original 'getattr(data, 'x', None)', now it properly moves to gpu
            'x_seen': None,
            'modalities': data.modalities}
        if data.pos is not None:
            self.xyz = data.pos

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
        print("mm_data_dict: ", data)

        self._set_input(data)
        mm_data_dict = self.input
        
        
        print("self.down_modules: ", self.down_modules)
        
        # Apply ONLY atomic-level pooling which is in `down_modules`
        for i in range(len(self.down_modules)):
            mm_data_dict = self.down_modules[i](mm_data_dict)
            
        
    
        # Feng:
        # 1. do view-sampling per point
        # 2. run viewing conditions through Attention Transformer
        # 3. save those in mm_data_dict['modalities'][modality]?
        print("pooled mm_data_dict: ", mm_data_dict)
        print(mm_data_dict.modalities['image'])
            
            
            
        raise NotImplementedError

        # Discard the modalities used in the down modules, only
        # 3D point features are expected to be used in subsequent
        # modules. Restore the input Data object equipped with the
        # proper point positions and modality-generated features.
        out = Batch(
            x=mm_data_dict['x_3d'], pos=self.xyz, seen=mm_data_dict['x_seen'])

        # TODO: this always passes the modality feature maps in the
        #  output dictionary. May not be relevant at inference time,
        #  when we would rather save memory and discard unnecessary
        #  data. Besides, this behavior for NoDEncoder is not consistent
        #  with the multimodal UNet, need to consider homogenizing.
        for m in self.modalities:
            # x_mod = getattr(mm_data_dict['modalities'][m], 'last_view_x_mod', None)
            # if x_mod is not None:
            #     out[m] = mm_data_dict['modalities'][m]
            out[m] = mm_data_dict['modalities'][m]

#         # Apply the MLP head, if any
#         if self.has_mlp_head:
#             # Apply to the 3D features
#             out.x = self.mlp(out.x)

#             # Apply to the last modality-based view-level features
#             for m in [mod for mod in self.modalities if mod in out.keys]:
#                 out[m].last_view_x_mod = self.mlp(out[m].last_view_x_mod)

        return out
