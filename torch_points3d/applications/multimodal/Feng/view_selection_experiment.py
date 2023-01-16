import logging
from abc import ABC
import time

import numpy as np
import torch
# from torch_points3d.models.base_architectures.backbone import MVFusionBackboneBasedModel
from torch_points3d.applications.utils import extract_output_nc
from torch_points3d.core.common_modules.base_modules import MLP
from torch_points3d.core.multimodal.data import MMData
from torch_geometric.data import Batch
from torch.utils.checkpoint import checkpoint
import copy
from torch import nn
from torch_points3d.core.common_modules.base_modules import Identity
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.models.base_architectures import ModalityFactory, get_factory
from torch_points3d.models.base_model import BaseModel
from torch_points3d.modules.multimodal.modules import MultimodalBlockDown, \
    UnimodalBranch, IdentityBranch, UnimodalBranchOnlyAtomicPool
from torch_points3d.utils.config import is_list, fetch_arguments_from_list, \
    fetch_modalities, getattr_recursive
from torch_points3d.core.multimodal.data import MODALITY_NAMES
import logging
import sys
import torch
import torch.nn.functional as F
from torch_scatter import segment_csr, scatter_min, scatter_max
from torch_points3d.core.common_modules import MLP
import math


### Feng
import sys
sys.path.append("/home/fsun/thesis/modeling")
from compare_methods import DVA_attention_weighted_M2F_preds

SPECIAL_NAMES = ["block_names"]

log = logging.getLogger(__name__)

# ------------------------- MVFusion backbone ----------------------------#
class ViewSelectionExpBackboneBasedModel(BaseModel, ABC):
    """
    Create a backbone-based generator: this is simply an encoder (can be
    used in classification, regression, metric learning and so on).
    """

    def __init__(self, opt, model_type, dataset: BaseDataset, modules_lib):

        """Construct a backbone generator (It is a simple down module)
        Parameters:
            opt - options for the network generation
            model_type - type of the model to be generated
            modules_lib - all modules that can be used in the backbone


        opt is expected to contains the following keys:
        * down_conv
        """
        opt = copy.deepcopy(opt)
        super().__init__(opt)
        self._spatial_ops_dict = {"neighbour_finder": [], "sampler": []}

        # Check if one of the supported modalities is present in the config
        self._modalities = fetch_modalities(opt.down_conv, MODALITY_NAMES)

        # Check if the 3D convolutions are specified in the config
        self.no_3d_conv = "down_conv_nn" not in opt.down_conv

        # Detect which options format has been used to define the model
        if is_list(opt.down_conv) or self.no_3d_conv and not self.is_multimodal:
            raise NotImplementedError
        else:
            self._init_from_compact_format(
                opt, model_type, dataset, modules_lib)

    def _init_from_compact_format(self, opt, model_type, dataset, modules_lib):
        """Create a backbonebasedmodel from the compact options format
        - where the same convolution is given for each layer, and
        arguments are given in lists.
        """
        print("opt:  ", opt)

        # Initialize the down module list
        self.down_modules = nn.ModuleList()

        ### Feng: Skip creating 2D CNN encoder
        self._module_factories = {}

        # Factories for creating modules for additional modalities
        if self.is_multimodal:
            for m in self.modalities:
                mod_opt = opt.down_conv[m]
                self._module_factories[m] = ModalityFactory(
                    m,
                    mod_opt.down_conv.module_name,
                    mod_opt.atomic_pooling.module_name,
                    mod_opt.view_pooling.module_name,
                    mod_opt.fusion.module_name)

        # Down modules - 3D conv only
        down_modules = []
        if not self.no_3d_conv:
            for i in range(len(opt.down_conv.down_conv_nn)):
                down_conv_3d = self._build_module(opt.down_conv, i, flow="DOWN")
                self._save_sampling_and_search(down_conv_3d)
                down_modules.append(down_conv_3d)

        # Number of early modules with no 3D conv and no skip-connections
        self._n_early_conv = getattr(
            opt.down_conv, 'n_early_conv', int(self.is_multimodal))

        # Down modules - modality-specific branches
        if self.is_multimodal:

            # Whether the multimodal blocks should use 3D convolutions
            # before the fusion, after the fusion or both. Inject
            # Identity accordingly in the down_modules
            conv3d_before_fusion = getattr(
                opt.down_conv, 'conv3d_before_fusion', True)
            conv3d_after_fusion = getattr(
                opt.down_conv, 'conv3d_after_fusion', True)
            assert conv3d_before_fusion or conv3d_after_fusion, \
                f'Multimodal blocks need a 3D convolution either before or ' \
                f'after the fusion.'
            if conv3d_before_fusion and not conv3d_after_fusion:
                down_modules = [y for x in down_modules for y in (x, Identity())]
            if not conv3d_before_fusion and conv3d_after_fusion:
                down_modules = [y for x in down_modules for y in (Identity(), x)]

            # Insert Identity 3D convolutions modules to allow branching
            # directly into the raw 3D features for early fusion
            early_modules = [Identity() for _ in range(self.n_early_conv * 2)]
            down_modules = early_modules + down_modules

            # Compute the number of multimodal blocks
            assert len(down_modules) % 2 == 0 and len(down_modules) > 0, \
                f"Expected an even number of 3D conv modules but got " \
                f"{len(down_modules)} modules instead."
            n_mm_blocks = len(down_modules) // 2

            branches = [
                {m: IdentityBranch() for m in self.modalities}
                for _ in range(n_mm_blocks)]

            for m in self.modalities:

                # Get the branching indices
                b_idx = opt.down_conv[m].branching_index
                b_idx = [b_idx] if not is_list(b_idx) else b_idx
                
                # Check whether the modality module is a UNet
                is_unet = getattr(opt.down_conv[m], 'up_conv', None) is not None
                assert not is_unet or len(b_idx) == 1, \
                    f"Cannot build a {m}-specific UNet with multiple " \
                    f"branching indices. Consider removing the 'up_conv' " \
                    f"from the {m} modality or providing a single branching " \
                    f"index."
                
                # Ensure the modality has no modules pointing to the
                # same branching index
                assert len(set(b_idx)) == len(b_idx), \
                    f"Cannot build multimodal model: some '{m}' blocks have " \
                    f"the same branching index."
                
                # Build the branches
                for i, idx in enumerate(b_idx):

                    ########### NOTE: ViewSelectionExp has a single branch (for M2F features)

                    # Ensure the branching index matches the down_conv
                    # length
                    assert idx < n_mm_blocks, \
                        f"Cannot build multimodal model: branching index " \
                        f"'{idx}' of modality '{m}' is too large for the " \
                        f"'{n_mm_blocks}' multimodal blocks."

                    atomic_pool = self._build_module(
                        opt.down_conv[m].atomic_pooling, i, modality=m,
                        flow='ATOMIC')


                    opt_branch = fetch_arguments_from_list(
                        opt.down_conv[m], i, SPECIAL_NAMES)
                    drop_3d = opt_branch.get('drop_3d', 0)
                    drop_mod = opt_branch.get('drop_mod', 0)
                    keep_last_view = opt_branch.get('keep_last_view', False)
                    checkpointing = opt_branch.get('checkpointing', '')
                    out_channels = opt_branch.get('out_channels', None)
                    interpolate = opt_branch.get('interpolate', False)

                    # Group modules into a UnimodalBranch and update the
                    # branches at the proper branching point
                    branches[idx][m] = UnimodalBranchOnlyAtomicPool(
                        atomic_pool, drop_3d=drop_3d,
                        drop_mod=drop_mod, keep_last_view=keep_last_view,
                        checkpointing=checkpointing, out_channels=out_channels,
                        interpolate=interpolate)

            # Update the down_modules list
            down_modules = [
                MultimodalBlockDown(conv_1, conv_2, **modal_conv)
                for conv_1, conv_2, modal_conv
                in zip(down_modules[::2], down_modules[1::2], branches)]

        # Down modules - combined
        self.down_modules = nn.ModuleList(down_modules)

        self.metric_loss_module, self.miner_module \
            = BaseModel.get_metric_loss_and_miner(
            getattr(opt, "metric_loss", None), getattr(opt, "miner", None))

    def _save_sampling_and_search(self, down_conv):
        sampler = getattr(down_conv, "sampler", None)
        if is_list(sampler):
            self._spatial_ops_dict["sampler"] \
                = sampler + self._spatial_ops_dict["sampler"]
        else:
            self._spatial_ops_dict["sampler"] \
                = [sampler] + self._spatial_ops_dict["sampler"]

        neighbour_finder = getattr(down_conv, "neighbour_finder", None)
        if is_list(neighbour_finder):
            self._spatial_ops_dict["neighbour_finder"] \
                = neighbour_finder + self._spatial_ops_dict["neighbour_finder"]
        else:
            self._spatial_ops_dict["neighbour_finder"] \
                = [neighbour_finder] + self._spatial_ops_dict["neighbour_finder"]


    def _build_module(self, conv_opt, index, flow='DOWN', modality='main'):
        """Builds a convolution (up or down) or a merge block in the
        case of multimodal models.

        Arguments:
            conv_opt - model config subset describing the convolutional
                block
            index - layer index in sequential order (as they come in
                the config)
            flow - "UP", "DOWN", "ATOMIC, "VIEW" or "FUSION"
            modality - string among supported modalities
        """
        args = fetch_arguments_from_list(conv_opt, index, SPECIAL_NAMES)
        args["index"] = index
        module = self._module_factories[modality].get_module(flow, index=index)
        return module(**args)

    @property
    def modalities(self):
        return self._modalities

    @property
    def n_early_conv(self):
        return self._n_early_conv

    
class DeepSetFeat_AttentionWeighting(nn.Module, ABC):
    """Produce element-wise set features based on shared learned
    features.

    Inspired from:
        DeepSets: https://arxiv.org/abs/1703.06114
        PointNet: https://arxiv.org/abs/1612.00593
    """

    _POOLING_MODES = ['max', 'mean', 'min', 'sum']
    _FUSION_MODES = ['residual', 'concatenation', 'both']

    def __init__(
            self, d_in, d_out, pool='max', fusion='concatenation',
            use_num=False, num_classes=None, **kwargs):
        super(DeepSetFeat_AttentionWeighting, self).__init__()

        # Initialize the set-pooling mechanism to aggregate features of
        # elements-level features to set-level features
        pool = pool.split('_')
        assert all([p in self._POOLING_MODES for p in pool]), \
            f"Unsupported pool='{pool}'. Expected elements of: " \
            f"{self._POOLING_MODES}"
        self.f_pool = lambda a, b: torch.cat([
            segment_csr(a, b, reduce=p) for p in pool], dim=-1)
        self.pool = pool

        # Initialize the fusion mechanism to merge set-level and
        # element-level features
        if fusion == 'residual':
            self.f_fusion = lambda a, b: a + b
        elif fusion == 'concatenation':
            self.f_fusion = lambda a, b: torch.cat((a, b), dim=-1)
        elif fusion == 'both':
            self.f_fusion = lambda a, b: torch.cat((a, a + b), dim=-1)
        else:
            raise NotImplementedError(
                f"Unknown fusion='{fusion}'. Please choose among "
                f"supported modes: {self._FUSION_MODES}.")
        self.fusion = fusion

        # Initialize the MLPs
        self.d_in = d_in
        self.d_out = d_out
        self.use_num = use_num
        self.mlp_elt_1 = MLP(
            [d_in, d_out, d_out], bias=False)
        in_set_mlp = d_out * len(self.pool) + self.use_num
        self.mlp_set = MLP(
            [in_set_mlp, d_out, d_out], bias=False)
        in_last_mlp = d_out if fusion == 'residual' else d_out * 2
        self.mlp_elt_2 = MLP(
            [in_last_mlp, d_out, d_out], bias=False)
        
        # E_score computes the compatibility score for each feature
        # group, these are to be further normalized to produce
        # final attention scores
        self.E_score = nn.Linear(d_out, num_classes, bias=True)
        
        self.num_classes = num_classes

    def forward(self, x, csr_idx, x_mod):
        x = self.mlp_elt_1(x)
        x_set = self.f_pool(x, csr_idx)
        if self.use_num:
            # Heuristic to normalize in [0,1]
            set_num = torch.sqrt(1 / (csr_idx[1:] - csr_idx[:-1] + 1e-3))
            x_set = torch.cat((x_set, set_num.view(-1, 1)), dim=1)
        x_set = self.mlp_set(x_set)
        x_set = gather_csr(x_set, csr_idx)
        x_out = self.f_fusion(x, x_set)
        x_out = self.mlp_elt_2(x_out)
        
        
        
        # Attention weighting 
        
        # Compute compatibilities (unscaled scores) : V x num_groups
        compatibilities = self.E_score(x_out)
        

        # Compute attention scores : V x num_classes
        attentions = segment_softmax_csr(
            compatibilities, csr_idx, scaling=False)
        # Apply attention scores : P x F_mod
        x_pool = segment_csr(
            x_mod * expand_group_feat(attentions, self.num_classes, self.num_classes),
            csr_idx, reduce='sum')

        
        return x_pool

    def extra_repr(self) -> str:
        repr_attr = ['pool', 'fusion', 'use_num']
        return "\n".join([f'{a}={getattr(self, a)}' for a in repr_attr])

    
class DeepSetFeat_ViewExperiment(nn.Module, ABC):
    """Produce element-wise set features based on shared learned
    features.

    Inspired from:
        DeepSets: https://arxiv.org/abs/1703.06114
        PointNet: https://arxiv.org/abs/1612.00593
    """

    _POOLING_MODES = ['max', 'mean', 'min', 'sum']
    _FUSION_MODES = ['residual', 'concatenation', 'both']

    def __init__(
            self, d_in, d_out, pool='max', fusion='concatenation',
            use_num=False, num_classes=None, **kwargs):
        super(DeepSetFeat_ViewExperiment, self).__init__()

        # Initialize the set-pooling mechanism to aggregate features of
        # elements-level features to set-level features
        pool = pool.split('_')
        assert all([p in self._POOLING_MODES for p in pool]), \
            f"Unsupported pool='{pool}'. Expected elements of: " \
            f"{self._POOLING_MODES}"
        self.f_pool = lambda a, b: torch.cat([
            segment_csr(a, b, reduce=p) for p in pool], dim=-1)
        self.pool = pool

        # Initialize the fusion mechanism to merge set-level and
        # element-level features
        if fusion == 'residual':
            self.f_fusion = lambda a, b: a + b
        elif fusion == 'concatenation':
            self.f_fusion = lambda a, b: torch.cat((a, b), dim=-1)
        elif fusion == 'both':
            self.f_fusion = lambda a, b: torch.cat((a, a + b), dim=-1)
        else:
            raise NotImplementedError(
                f"Unknown fusion='{fusion}'. Please choose among "
                f"supported modes: {self._FUSION_MODES}.")
        self.fusion = fusion

        # Initialize the MLPs
        self.d_in = d_in
        self.d_out = d_out
        self.use_num = use_num
        self.mlp_elt_1 = MLP(
            [d_in, d_out, d_out], bias=False)
        in_set_mlp = d_out * len(self.pool) + self.use_num
        self.mlp_set = MLP(
            [in_set_mlp, d_out, d_out], bias=False)
        in_last_mlp = d_out if fusion == 'residual' else d_out * 2
        self.mlp_elt_2 = MLP(
            [in_last_mlp, d_out, d_out], bias=False)
        
        # E_score computes the compatibility score for each feature
        # group, these are to be further normalized to produce
        # final attention scores
        self.E_score = nn.Linear(d_out, num_classes, bias=True)
        
        self.num_classes = num_classes

    def forward(self, x, csr_idx, x_mod):
        x = self.mlp_elt_1(x)
        x_set = self.f_pool(x, csr_idx)
        if self.use_num:
            # Heuristic to normalize in [0,1]
            set_num = torch.sqrt(1 / (csr_idx[1:] - csr_idx[:-1] + 1e-3))
            x_set = torch.cat((x_set, set_num.view(-1, 1)), dim=1)
        x_set = self.mlp_set(x_set)
        x_set = gather_csr(x_set, csr_idx)
        x_out = self.f_fusion(x, x_set)
        x_out = self.mlp_elt_2(x_out)

        return x_out

    def extra_repr(self) -> str:
        repr_attr = ['pool', 'fusion', 'use_num']
        return "\n".join([f'{a}={getattr(self, a)}' for a in repr_attr])
    

class ViewSelectionExpEncoder(ViewSelectionExpBackboneBasedModel, ABC):
    """Encoder structure for multimodal models without 3D data.

    Inspired from torchpoints_3d.applications.sparseconv3d.
    
    
    Feng modification: this structure creates multi-view fused features for each 3D point,
        based on its Mask2Former view-wise predictions and the viewing conditions.
    """

    def __init__(
            self, model_config, model_type, dataset, modules, *args, **kwargs):
        # UnwrappedUnetBasedModel init
        super().__init__(
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

        self.use_transformer = False
        self.use_deepset = False
            
        # modules
        if model_config.backbone.use_transformer:
            self.use_transformer = True
            self.fusion = DVA_attention_weighted_M2F_preds(model_config['backbone']['transformer'])
        elif model_config.backbone.use_deepset:
            self.use_deepset = True
            d_in = 8 + dataset.num_classes   # Viewing Conditions + Input pred label (one hot)
            d_hidden = 32
            pool = 'max'
            fusion = 'concatenation'
            use_num = True
                    
            self.fusion = DeepSetFeat_ViewExperiment(d_in=d_in, d_out=d_hidden, pool=pool, fusion=fusion,
            use_num=use_num, num_classes=dataset.num_classes)
        
        self.n_views = model_config.backbone.transformer.n_views
        self.n_classes = model_config.backbone.transformer.n_classes
        self.MAX_SEEN_POINTS = model_config.backbone.transformer.max_n_points
        
        print("WARNING: input points for Multi-View Fusion module are clipped at MAX_SEEN_POINTS for fair model comparison in evaluation", flush=True)
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
        if self.use_transformer:
    
            # Features from only seen point-image matches are included in 'x'
            viewing_feats = data.data.mvfusion_input[:, :, :-1]
            m2f_feats = data.data.mvfusion_input[:, :, -1]

            # Mask2Former predictions per view as feature
            m2f_feats = torch.nn.functional.one_hot(m2f_feats.squeeze().long(), self.n_classes)

            ### Multi-view fusion of M2F and viewing conditions using Transformer
            # TODO: remove assumption that pixel validity is the 1st feature
            invalid_pixel_mask = viewing_feats[:, :, 0] == 0
        
            fusion_input = {
                'invalid_pixels_mask': invalid_pixel_mask.to(self.device),
                'viewing_features': viewing_feats.to(self.device),
                'one_hot_mask_labels': m2f_feats.to(self.device)
            }

            # get logits
            if self._checkpointing:
                out_scores = checkpoint(self.fusion, fusion_input)
            else:
                out_scores = self.fusion(fusion_input)
        elif self.use_deepset:
            
            viewing_conditions = data.modalities['image'][0].mappings.values[2]

            input_preds = data.modalities['image'][0].get_mapped_m2f_features()
            input_preds_one_hot = torch.nn.functional.one_hot(input_preds.long().squeeze(), self.n_classes)
            attention_input = torch.concat((viewing_conditions, input_preds_one_hot), dim=1)
            
            csr_idx = data.modalities['image'][0].view_csr_indexing

            out_scores = self.fusion(attention_input, csr_idx, input_preds_one_hot)
            
        
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



#################################################### Functions hardcoded #####################################################
def nearest_power_of_2(x, min_power=16):
    """Local helper to find the nearest power of 2 of a given number.
    The `min_power` parameter puts a minimum threshold for the returned
    power.
    """
    x = int(x)

    if x < min_power:
        return min_power

    previous_power = 2 ** ((x - 1).bit_length() - 1)
    next_power = 2 ** (x - 1).bit_length()

    if x - previous_power < next_power - x:
        return previous_power
    else:
        return next_power


def group_sizes(num_elements, num_groups):
    """Local helper to compute the group sizes, when distributing
    num_elements across num_groups while keeping group sizes as close
    as possible."""
    sizes = torch.full(
        (num_groups,), math.floor(num_elements / num_groups),
        dtype=torch.long)
    sizes += torch.arange(num_groups) < num_elements - sizes.sum()
    return sizes


def expand_group_feat(A, num_groups, num_channels):
    if num_groups == 1:
        A = A.view(-1, 1)
    elif num_groups < num_channels:
        # Expand compatibilities to features of the same group
        sizes = group_sizes(num_channels, num_groups).to(A.device)
        A = A.repeat_interleave(sizes, dim=1)
    return A


@torch.jit.script
def segment_softmax_csr(src: torch.Tensor, csr_idx: torch.Tensor,
                        eps: float = 1e-12, scaling: bool = False) -> torch.Tensor:
    """Equivalent of scatter_softmax but for CSR indices.
    Based on: torch_scatter/composite/softmax.py

    The `scaling` option allows for scaled softmax computation, where
    `scaling='True'` scales by the number of items in each index group.
    """
    if not torch.is_floating_point(src):
        raise ValueError(
            '`segment_csr_softmax` can only be computed over tensors with '
            'floating point data types.')
    if csr_idx.dim() != 1:
        raise ValueError(
            '`segment_csr_softmax` can only be computed over 1D CSR indices.')
    if src.dim() > 2:
        raise NotImplementedError(
            '`segment_csr_softmax` can only be computed over 1D or 2D source '
            'tensors.')

    # Compute dense indices from CSR indices
    n_groups = csr_idx.shape[0] - 1
    dense_idx = torch.arange(n_groups).to(src.device).repeat_interleave(
        csr_idx[1:] - csr_idx[:-1])
    if src.dim() > 1:
        dense_idx = dense_idx.view(-1, 1).repeat(1, src.shape[1])

    # Center scores maxima near 1 for computation precision
    max_value_per_index = segment_csr(src, csr_idx, reduce='max')
    max_per_src_element = max_value_per_index.gather(0, dense_idx)
    centered_scores = src - max_per_src_element

    # Optionally scale scores by the sqrt of index group sizes
    if scaling:
        num_per_index = (csr_idx[1:] - csr_idx[:-1])
        sqrt_num_per_index = num_per_index.float().sqrt()
        num_per_src_element = torch.repeat_interleave(
            sqrt_num_per_index, num_per_index)
        if src.dim() > 1:
            num_per_src_element = num_per_src_element.view(-1, 1).repeat(
                1, src.shape[1])

        centered_scores /= num_per_src_element

    # Compute the numerators
    centered_scores_exp = centered_scores.exp()

    # Compute the denominators
    sum_per_index = segment_csr(centered_scores_exp, csr_idx, reduce='sum')
    normalizing_constants = sum_per_index.add_(eps).gather(0, dense_idx)

    return centered_scores_exp.div(normalizing_constants)


@torch.jit.script
def gather_csr(src: torch.Tensor, csr_idx: torch.Tensor) -> torch.Tensor:
    """Gather index-level src values into element-level values based on
    CSR indices.

    When applied to the output or segment_csr, this redistributes the
    reduced values to the appropriate segment_csr input elements.
    """
    if not torch.is_floating_point(src):
        raise ValueError(
            '`gather_csr` can only be computed over tensors with '
            'floating point data types.')
    if csr_idx.dim() != 1:
        raise ValueError(
            '`gather_csr` can only be computed over 1D CSR indices.')
    if src.dim() > 2:
        raise NotImplementedError(
            '`gather_csr` can only be computed over 1D or 2D source '
            'tensors.')

    # Compute dense indices from CSR indices
    n_groups = csr_idx.shape[0] - 1
    dense_idx = torch.arange(n_groups).to(src.device).repeat_interleave(
        csr_idx[1:] - csr_idx[:-1])
    if src.dim() > 1:
        dense_idx = dense_idx.view(-1, 1).repeat(1, src.shape[1])

    # Center scores maxima near 1 for computation precision
    return src.gather(0, dense_idx)


@torch.jit.script
def segment_gather_csr(src: torch.Tensor, csr_idx: torch.Tensor,
                       reduce: str = 'sum') -> torch.Tensor:
    """Compute the reduced value between same-index elements, for CSR
    indices, and redistribute them to input elements.
    """
    # Reduce with segment_csr
    reduced_per_index = segment_csr(src, csr_idx, reduce=reduce)

    # Expand with gather_csr
    reduced_per_src_element = gather_csr(reduced_per_index, csr_idx)

    return reduced_per_src_element
