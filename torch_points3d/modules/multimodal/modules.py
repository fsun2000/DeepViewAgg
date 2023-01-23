from abc import ABC

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from torch_points3d.core.multimodal.data import MODALITY_NAMES
from torch_points3d.core.common_modules.base_modules import Identity, \
    BaseModule
from torch_points3d.modules.multimodal.dropout import ModalityDropout
from torchsparse.nn.functional import sphash, sphashquery
import torch_scatter

try:
    import MinkowskiEngine as me
except:
    me = None
try:
    import torchsparse as ts
except:
    ts = None

### Feng
import sys
sys.path.append("/home/fsun/thesis/modeling")
from compare_methods import DVA_cls_5_fusion_7, DVA_attention_weighted_M2F_preds

class MultimodalBlockDown(nn.Module, ABC):
    """Multimodal block with downsampling that looks like:

                 -- 3D Conv ---- Merge i -- 3D Conv --
    MMData IN          ...        |                       MMData OUT
                 -- Mod i Conv --|--------------------
                       ...
    """

    def __init__(self, block_1, block_2, **kwargs):
        """Build the Multimodal module from already-instantiated
        modules. Modality-specific modules are expected to be passed in
        dictionaries holding fully-fledged UnimodalBranch modules.
        """
        # BaseModule initialization
        super(MultimodalBlockDown, self).__init__()

        # Blocks for the implicitly main modality: 3D
        self.block_1 = block_1 if block_1 is not None else Identity()
        self.block_2 = block_2 if block_2 is not None else Identity()

        # Initialize the dict holding the conv and merge blocks for all
        # modalities
        self._modalities = []
        self._init_from_kwargs(**kwargs)

        # Expose the 3D convs .sampler attribute (for
        # UnwrappedUnetBasedModel)
        # TODO this is for KPConv, is it doing the intended, is it
        #  needed at all ?
        self.sampler = [
            getattr(self.block_1, "sampler", None),
            getattr(self.block_2, "sampler", None)]

    def _init_from_kwargs(self, **kwargs):
        """Kwargs are expected to carry fully-fledged modality-specific
        UnimodalBranch modules.
        """
        for m in kwargs.keys():
            assert (m in MODALITY_NAMES), \
                f"Invalid kwarg modality '{m}', expected one of " \
                f"{MODALITY_NAMES}."
            assert isinstance(kwargs[m], (UnimodalBranch, IdentityBranch)) or \
                   isinstance(kwargs[m], (UnimodalBranchOnlyAtomicPool, IdentityBranch)) or \
                   isinstance(kwargs[m], (MVFusionUnimodalBranch, IdentityBranch)) or \
                   isinstance(kwargs[m], (MVAttentionUnimodalBranch, IdentityBranch)), \
                f"Expected a UnimodalBranch or UnimodalBranchOnlyAtomicPool or MVFusionUnimodalBranch module for '{m}' modality " \
                f"but got {type(kwargs[m])} instead."
            setattr(self, m, kwargs[m])
            self._modalities.append(m)

    @property
    def modalities(self):
        return self._modalities

    def forward(self, mm_data_dict):
        """
        Forward pass of the MultiModalBlockDown.

        Expects a tuple of 3D data (Data, SparseTensor, etc.) destined
        for the 3D convolutional modules, and a dictionary of
        modality-specific data equipped with corresponding mappings.
        """
        # Conv on the main 3D modality - assumed to reduce 3D resolution
        mm_data_dict = self.forward_3d_block_down(
            mm_data_dict, self.block_1)

        for m in self.modalities:
            # TODO: does the modality-driven sequence of updates on x_3d
            #  and x_seen affect the modality behavior ? Should the shared
            #  3D information only be updated once all modality branches
            #  have been run on the same input ?
            mod_branch = getattr(self, m)
            
            mm_data_dict = mod_branch(mm_data_dict, m)

        # Conv on the main 3D modality
        mm_data_dict = self.forward_3d_block_down(
            mm_data_dict, self.block_2)

        return mm_data_dict

    @staticmethod
    def forward_3d_block_down(mm_data_dict, block):
        """
        Wrapper method to apply the forward pass on a 3D down conv
        block while preserving modality-specific mappings.

        This both runs the forward method of the input block but also
        catches the reindexing scheme, in case a sampling or sparse
        strided convolution is applied in the 3D conv block.

        For MinkowskiEngine or TorchSparse sparse tensors, the
        reindexing is recovered from the input/output coordinates. If
        no strided convolution was applied, the indexing stays the same
        and a None index is returned. Otherwise, the returned index
        maps indices as follows: i -> idx[i].

        For non-sparse convolutions, the reindexing is carried by the
        sampler's 'last_index' attribute. If no sampling was applied,
        the indexing stays the same and a None index is returned.
        Otherwise, the returned index carries the indices of the
        selected points with respect to their input order.
        """
        # Leave the input untouched if the 3D conv block is Identity
        if isinstance(block, Identity):
            return mm_data_dict

        # Unpack the multimodal data dictionary
        x_3d = mm_data_dict['x_3d']
        x_seen = mm_data_dict['x_seen']

        # Initialize index and indexation mode
        idx = None
        mode = 'pick'

        # Non-sparse forward and reindexing
        if isinstance(x_3d, torch.Tensor):
            # Forward pass on the block while keeping track of the
            # sampler indices
            block.sampler.last_idx = None
            idx_ref = torch.arange(x_3d.shape[0])
            x_3d = block(x_3d)
            idx_sample = block.sampler.last_idx
            if (idx_sample == idx_ref).all():
                idx = None
            else:
                idx = idx_sample
            mode = 'pick'

        # MinkowskiEngine forward and reindexing
        elif me is not None and isinstance(x_3d, me.SparseTensor):
            mode = 'merge'

            # Forward pass on the block while keeping track of the
            # stride levels
            stride_in = x_3d.tensor_stride[0]
            x_3d = block(x_3d)
            stride_out = x_3d.tensor_stride[0]

            if stride_in == stride_out:
                idx = None
            else:
                src, target = x_3d.coords_man.get_coords_map(
                    stride_in, stride_out)
                idx = target[src.argsort()]

        # TorchSparse forward and reindexing
        elif ts is not None and isinstance(x_3d, ts.SparseTensor):
            # Forward pass on the block while keeping track of the
            # stride levels
            stride_in = x_3d.s
            x_3d = block(x_3d)
            stride_out = x_3d.s

            if stride_in == stride_out:
                idx = None
            else:
                # To compute the reindexing of the sparse voxels with
                # torchsparse, we need to make use of the torchsparse
                # sphashquery function to compare sets of coordinates at
                # the same resolution. However, when changing resolution
                # we must be careful to voxelize spatial points but
                # leave the batch indices untouched. For torchsparse,
                # the batch indices are stored in the last column of
                # the coords tensor (unlike MinkowskiEngine which
                # stores batch indices in the first column). Hence we
                # assume here that coordinates to have shape (N x 4) and
                # batch indices to lie in the last column.
                assert x_3d.C.shape[1] == 4, \
                    f"Sparse coordinates are expected to have shape " \
                    f"(N x 4), with batch indices in the first column and " \
                    f"3D spatial coordinates in the following ones. Yet, " \
                    f"received coordinates tensor with shape {x_3d.C.shape} " \
                    f"instead."
                in_coords = x_3d.coord_maps[stride_in]
                in_coords[:, :3] = ((in_coords[:, :3].float() / stride_out
                                     ).floor() * stride_out).int()
                out_coords = x_3d.coord_maps[stride_out]
                idx = sphashquery(sphash(in_coords), sphash(out_coords))
                                
                # idx is -1 when an in_coords could not be matched to an
                # out_coords. Normally, this should never happen in our 
                # use case. But sometimes the GPU computation of idx 
                # produces -1 indices for reasons I still ignore, 
                # while the CPU computation works fine. This is a very 
                # rare occurrence which can have significant downstream 
                # repercussions. To this end, if we detect a negative 
                # index, we re-run the computation on CPU, even if it 
                # means breaking CPU-GPU asynchronicity
                if not idx.ge(0).all():
                    idx = sphashquery(sphash(in_coords.cpu()), sphash(out_coords.cpu()))
                    idx = idx.to(in_coords.device)
            mode = 'merge'

        else:
            raise NotImplementedError(
                f"Unsupported format for x_3d: {type(x_3d)}. If you are trying "
                f"to use MinkowskiEngine or TorchSparse, make sure those are "
                f"properly installed.")

        # Update seen 3D points indices
        if x_seen is not None and idx is not None:
            if mode == 'pick':
                x_seen = x_seen[idx]
            else:
                x_seen = torch_scatter.scatter(x_seen, idx, reduce='sum')

        # Update the multimodal data dictionary
        mm_data_dict['x_3d'] = x_3d
        mm_data_dict['x_seen'] = x_seen

        # Update modality data and mappings wrt new point indexing
        for m in mm_data_dict['modalities'].keys():
            mm_data_dict['modalities'][m] = \
                mm_data_dict['modalities'][m].select_points(idx, mode=mode)

        return mm_data_dict


class MultimodalBlockUp(nn.Module, ABC):
    """Multimodal block with downsampling that looks like:

                 -- 3D Conv ---- Merge i -- 3D Conv --
    MMData IN          ...        |                       MMData OUT
                 -- Mod i Conv --|--------------------
                       ...
    """


class UnimodalBranchOnlyAtomicPool(nn.Module, ABC):
    """Unimodal block with downsampling that looks like:

    IN 3D   ------------------------  OUT 3D
                                     /
                       Atomic Pool --
                     /
    IN Mod  -- Conv -----------------------------------------  OUT Mod

    The convolution may be a down-convolution or preserve input shape.
    However, up-convolutions are not supported, because reliable the
    mappings cannot be inferred when increasing resolution.
    """    
    def __init__(
            self, atomic_pool, drop_3d=0, drop_mod=0,
            hard_drop=False, keep_last_view=False, checkpointing='',
            out_channels=None, interpolate=False):
        super(UnimodalBranchOnlyAtomicPool, self).__init__()
#         self.atomic_pool = atomic_pool
        drop_cls = ModalityDropout if hard_drop else nn.Dropout
        self.drop_3d = drop_cls(p=drop_3d, inplace=False) \
            if drop_3d is not None and drop_3d > 0 \
            else None
        self.drop_mod = drop_cls(p=drop_mod, inplace=True) \
            if drop_mod is not None and drop_mod > 0 \
            else None
        self.keep_last_view = keep_last_view
        self._out_channels = out_channels
        self.interpolate = interpolate

        # Optional checkpointing to alleviate memory at train time.
        # Character rules:
        #     c: convolution
        #     a: atomic pooling
        #     v: view pooling
        #     f: fusion
        assert not checkpointing or isinstance(checkpointing, str),\
            f'Expected checkpointing to be of type str but received ' \
            f'{type(checkpointing)} instead.'
        self.checkpointing = ''.join(set('cavf').intersection(set(checkpointing)))    
    
    @property
    def out_channels(self):
        if self._out_channels is None:
            raise ValueError(
                f'{self.__class__.__name__}.out_channels has not been '
                f'set. Please set it to allow inference even when the '
                f'modality has no data.')
        return self._out_channels

    def forward(self, mm_data_dict, modality):
        # Unpack the multimodal data dictionary. Specific treatment for
        # MinkowskiEngine and TorchSparse SparseTensors
        is_sparse_3d = not isinstance(
            mm_data_dict['x_3d'], (torch.Tensor, type(None)))
        x_3d = mm_data_dict['x_3d'].F if is_sparse_3d else mm_data_dict['x_3d']
        mod_data = mm_data_dict['modalities'][modality]
        
        # Check whether the modality carries multi-setting data
        is_multi_shape = isinstance(mod_data.x, list)

        # If the modality has no data mapped to the current 3D points,
        # ignore the branch forward. `self.out_channels` will guide us
        # on how to replace expected modality features
        if is_multi_shape and all([e.x.shape[0] == 0 for e in mod_data]) \
                or is_multi_shape and len(mod_data) == 0 \
                or not is_multi_shape and mod_data.x.shape[0] == 0:

            # Prepare the channel sizes
            nc_out = self.out_channels
            nc_3d = x_3d.shape[1]
            nc_2d = nc_out - nc_3d if nc_out > nc_3d else nc_3d

            # Make sure we have a valid `self.out_channels` so we can
            # simulate the forward without any modality data
            if nc_out < nc_3d:
                raise ValueError(
                    f'{self.__class__.__name__}.out_channels is smaller than '
                    f'number of features in x_3d: {nc_out} < {nc_3d}')

            # No points are seen
            # x_seen = torch.zeros(nc_3d, dtype=torch.bool)

            # Modify the feature dimension of mod_data to simulate
            # convolutions too
            if not is_multi_shape:
                mod_data.x = mod_data.x[:, [0]].repeat_interleave(nc_2d, dim=1)
            elif len(mod_data) > 0:
                mod_data.x = [
                    x[:, [0]].repeat_interleave(nc_2d, dim=1)
                    for x in mod_data.x]

            # For concatenation fusion, create zero features to
            # 'simulate' concatenation of modality features to x_3d
            if nc_out > nc_3d:
                zeros = torch.zeros_like(x_3d[:, [0]])
                zeros = zeros.repeat_interleave(nc_2d, dim=1)
                x_3d = torch.cat((x_3d, zeros), dim=1)

            # Return the modified multimodal data dictionary despite the
            # absence of modality features
            if is_sparse_3d:
                mm_data_dict['x_3d'].F = x_3d
            else:
                mm_data_dict['x_3d'] = x_3d
            mm_data_dict['modalities'][modality] = mod_data
            # if mm_data_dict['x_seen'] is None:
            #     mm_data_dict['x_seen'] = x_seen
            # else:
            #     mm_data_dict['x_seen'] = torch.logical_or(
            #         x_seen, mm_data_dict['x_seen'])

            return mm_data_dict

        # If the modality has a data list format and that one of the
        # items is an empty feature map, run a recursive forward on the
        # mm_data_dict with these problematic items discarded. This is
        # necessary whenever an element of the batch has no mappings to
        # the modality
        if is_multi_shape and any([e.x.shape[0] == 0 for e in mod_data]):

            # Remove problematic elements from mod_data
            num = len(mod_data)
            removed = {
                i: e for i, e in enumerate(mod_data) if e.x.shape[0] == 0}
            indices = [i for i in range(num) if i not in removed.keys()]
            mm_data_dict['modalities'][modality] = mod_data[indices]

            # Run forward recursively
            mm_data_dict = self.forward(mm_data_dict, modality)

            # Restore problematic elements. This is necessary if we need
            # to restore the initial batch elements with methods such as
            # `MMBatch.to_mm_data_list`
            mod_data = mm_data_dict['modalities'][modality]
            kept = {k: e for k, e in zip(indices, mod_data)}
            joined = {**kept, **removed}
            mod_data = mod_data.__class__([joined[i] for i in range(num)])
            mm_data_dict['modalities'][modality] = mod_data

            return mm_data_dict

#         # Forward pass with `self.conv`
#         mod_data = self.forward_conv(mod_data)

        # Extract mapped features from the feature maps of each input
        # modality setting
#         ########## NOTE: only Mask2Former feature map!
#         x_mod_m2f = mod_data.get_mapped_m2f_features(interpolate=self.interpolate)
        
        
        
        
        

#         # Atomic pooling of the modality features on each separate
#         # setting
#         x_mod_m2f = self.forward_atomic_pool(x_3d, x_mod_m2f, mod_data.atomic_csr_indexing)

#         # View pooling of the modality features
#         x_mod, mod_data, csr_idx = self.forward_view_pool(x_3d, x_mod, mod_data)

        if is_multi_shape:
            print("modality data is multi setting")
            csr_idx = mod_data.view_cat_csr_indexing
        else:
            csr_idx = mod_data.view_csr_indexing

        # Compute the boolean mask of seen points
        x_seen = csr_idx[1:] > csr_idx[:-1]

        
#         if isinstance(x_mod_m2f, list):
#             if len(x_mod_m2f) > 1:
#                 print("x_mod_m2f should only contain one tensor, but currently has more than one!")
#             x_mod_m2f = x_mod_m2f[0]
              
        ### Feng: disable M2F feature dropout because it has no practical use
#         # Dropout 3D or modality features
#         x_3d, x_mod_m2f, mod_data = self.forward_dropout(x_3d, x_mod_m2f, mod_data)

        
        
        # Feng: Skip fusion of modality into 3D point features
#         # Fuse the modality features into the 3D points features
#         x_3d = self.forward_fusion(x_3d, x_mod)

        # In case it has not been provided at initialization, save the
        # output channel size. This is useful for when a batch has no
        # modality data
        if self._out_channels is None:
            self._out_channels = x_3d.shape[1]

        # Update the multimodal data dictionary
        # TODO: does the modality-driven sequence of updates on x_3d
        #  and x_seen affect the modality behavior ? Should the shared
        #  3D information only be updated once all modality branches
        #  have been run on the same input ?
        if is_sparse_3d:
            mm_data_dict['x_3d'].F = x_3d
        else:
            mm_data_dict['x_3d'] = x_3d
        mm_data_dict['modalities'][modality] = mod_data
        if mm_data_dict['x_seen'] is None:
            mm_data_dict['x_seen'] = x_seen
        else:
            mm_data_dict['x_seen'] = torch.logical_or(
                x_seen, mm_data_dict['x_seen'])

        return mm_data_dict

#     def forward_atomic_pool(self, x_3d, x_mod, csr_idx):
#         """Atomic pooling of the modality features on each separate
#         setting.

#         :param x_3d:
#         :param x_mod:
#         :param csr_idx:
#         :return:
#         """
#         # If the modality carries multi-setting data, recursive scheme
#         if isinstance(x_mod, list):
#             x_mod = [
#                 self.forward_atomic_pool(x_3d, x, i)
#                 for x, i in zip(x_mod, csr_idx)]
#             return x_mod

#         if 'a' in self.checkpointing:
#             x_mod = checkpoint(self.atomic_pool, x_3d, x_mod, None, csr_idx)
#         else:
#             x_mod = self.atomic_pool(x_3d, x_mod, None, csr_idx)
#         return x_mod

#     def forward_dropout(self, x_3d, x_mod, mod_data):
#         if self.drop_3d:
#             x_3d = self.drop_3d(x_3d)
#         if self.drop_mod:
#             x_mod = self.drop_mod(x_mod)
#             if self.keep_last_view:
#                 mod_data.last_view_x_mod = self.drop_mod(mod_data.last_view_x_mod)
#         return x_3d, x_mod, mod_data

    def extra_repr(self) -> str:
        repr_attr = ['drop_3d', 'drop_mod', 'keep_last_view', 'checkpointing']
        return "\n".join([f'{a}={getattr(self, a)}' for a in repr_attr])

    
    
class UnimodalBranch(nn.Module, ABC):
    """Unimodal block with downsampling that looks like:

    IN 3D   ------------------------------------           --  OUT 3D
                                   \            \         /
                       Atomic Pool -- View Pool -- Fusion
                     /
    IN Mod  -- Conv -----------------------------------------  OUT Mod

    The convolution may be a down-convolution or preserve input shape.
    However, up-convolutions are not supported, because reliable the
    mappings cannot be inferred when increasing resolution.
    """

    def __init__(
            self, conv, atomic_pool, view_pool, fusion, drop_3d=0, drop_mod=0,
            hard_drop=False, keep_last_view=False, checkpointing='',
            out_channels=None, interpolate=False):
        super(UnimodalBranch, self).__init__()

        self.conv = conv
        self.atomic_pool = atomic_pool
        self.view_pool = view_pool
        self.fusion = fusion
        drop_cls = ModalityDropout if hard_drop else nn.Dropout
        self.drop_3d = drop_cls(p=drop_3d, inplace=False) \
            if drop_3d is not None and drop_3d > 0 \
            else None
        self.drop_mod = drop_cls(p=drop_mod, inplace=True) \
            if drop_mod is not None and drop_mod > 0 \
            else None
        self.keep_last_view = keep_last_view
        self._out_channels = out_channels
        self.interpolate = interpolate

        # Optional checkpointing to alleviate memory at train time.
        # Character rules:
        #     c: convolution
        #     a: atomic pooling
        #     v: view pooling
        #     f: fusion
        assert not checkpointing or isinstance(checkpointing, str),\
            f'Expected checkpointing to be of type str but received ' \
            f'{type(checkpointing)} instead.'
        self.checkpointing = ''.join(set('cavf').intersection(set(checkpointing)))

    @property
    def out_channels(self):
        if self._out_channels is None:
            raise ValueError(
                f'{self.__class__.__name__}.out_channels has not been '
                f'set. Please set it to allow inference even when the '
                f'modality has no data.')
        return self._out_channels

    def forward(self, mm_data_dict, modality):
        # Unpack the multimodal data dictionary. Specific treatment for
        # MinkowskiEngine and TorchSparse SparseTensors
        is_sparse_3d = not isinstance(
            mm_data_dict['x_3d'], (torch.Tensor, type(None)))
        x_3d = mm_data_dict['x_3d'].F if is_sparse_3d else mm_data_dict['x_3d']
        mod_data = mm_data_dict['modalities'][modality]
        


        # Check whether the modality carries multi-setting data
        is_multi_shape = isinstance(mod_data.x, list)

        # If the modality has no data mapped to the current 3D points,
        # ignore the branch forward. `self.out_channels` will guide us
        # on how to replace expected modality features
        if is_multi_shape and all([e.x.shape[0] == 0 for e in mod_data]) \
                or is_multi_shape and len(mod_data) == 0 \
                or not is_multi_shape and mod_data.x.shape[0] == 0:

            # Prepare the channel sizes
            nc_out = self.out_channels
            nc_3d = x_3d.shape[1]
            nc_2d = nc_out - nc_3d if nc_out > nc_3d else nc_3d

            # Make sure we have a valid `self.out_channels` so we can
            # simulate the forward without any modality data
            if nc_out < nc_3d:
                raise ValueError(
                    f'{self.__class__.__name__}.out_channels is smaller than '
                    f'number of features in x_3d: {nc_out} < {nc_3d}')

            # No points are seen
            # x_seen = torch.zeros(nc_3d, dtype=torch.bool)

            # Modify the feature dimension of mod_data to simulate
            # convolutions too
            if not is_multi_shape:
                mod_data.x = mod_data.x[:, [0]].repeat_interleave(nc_2d, dim=1)
            elif len(mod_data) > 0:
                mod_data.x = [
                    x[:, [0]].repeat_interleave(nc_2d, dim=1)
                    for x in mod_data.x]

            # For concatenation fusion, create zero features to
            # 'simulate' concatenation of modality features to x_3d
            if nc_out > nc_3d:
                zeros = torch.zeros_like(x_3d[:, [0]])
                zeros = zeros.repeat_interleave(nc_2d, dim=1)
                x_3d = torch.cat((x_3d, zeros), dim=1)

            # Return the modified multimodal data dictionary despite the
            # absence of modality features
            if is_sparse_3d:
                mm_data_dict['x_3d'].F = x_3d
            else:
                mm_data_dict['x_3d'] = x_3d
            mm_data_dict['modalities'][modality] = mod_data
            # if mm_data_dict['x_seen'] is None:
            #     mm_data_dict['x_seen'] = x_seen
            # else:
            #     mm_data_dict['x_seen'] = torch.logical_or(
            #         x_seen, mm_data_dict['x_seen'])

            return mm_data_dict

        # If the modality has a data list format and that one of the
        # items is an empty feature map, run a recursive forward on the
        # mm_data_dict with these problematic items discarded. This is
        # necessary whenever an element of the batch has no mappings to
        # the modality
        if is_multi_shape and any([e.x.shape[0] == 0 for e in mod_data]):

            # Remove problematic elements from mod_data
            num = len(mod_data)
            removed = {
                i: e for i, e in enumerate(mod_data) if e.x.shape[0] == 0}
            indices = [i for i in range(num) if i not in removed.keys()]
            mm_data_dict['modalities'][modality] = mod_data[indices]

            # Run forward recursively
            mm_data_dict = self.forward(mm_data_dict, modality)

            # Restore problematic elements. This is necessary if we need
            # to restore the initial batch elements with methods such as
            # `MMBatch.to_mm_data_list`
            mod_data = mm_data_dict['modalities'][modality]
            kept = {k: e for k, e in zip(indices, mod_data)}
            joined = {**kept, **removed}
            mod_data = mod_data.__class__([joined[i] for i in range(num)])
            mm_data_dict['modalities'][modality] = mod_data

            return mm_data_dict

        # Forward pass with `self.conv`
        mod_data = self.forward_conv(mod_data)

        # Extract mapped features from the feature maps of each input
        # modality setting
        x_mod = mod_data.get_mapped_features(interpolate=self.interpolate)

        # Atomic pooling of the modality features on each separate
        # setting
        x_mod = self.forward_atomic_pool(x_3d, x_mod, mod_data.atomic_csr_indexing)

        # View pooling of the modality features
        x_mod, mod_data, csr_idx = self.forward_view_pool(x_3d, x_mod, mod_data)

        # Compute the boolean mask of seen points
        x_seen = csr_idx[1:] > csr_idx[:-1]

        # Dropout 3D or modality features
        x_3d, x_mod, mod_data = self.forward_dropout(x_3d, x_mod, mod_data)

        # Fuse the modality features into the 3D points features
        x_3d = self.forward_fusion(x_3d, x_mod)

        # In case it has not been provided at initialization, save the
        # output channel size. This is useful for when a batch has no
        # modality data
        if self._out_channels is None:
            self._out_channels = x_3d.shape[1]

        # Update the multimodal data dictionary
        # TODO: does the modality-driven sequence of updates on x_3d
        #  and x_seen affect the modality behavior ? Should the shared
        #  3D information only be updated once all modality branches
        #  have been run on the same input ?
        if is_sparse_3d:
            mm_data_dict['x_3d'].F = x_3d
        else:
            mm_data_dict['x_3d'] = x_3d
        mm_data_dict['modalities'][modality] = mod_data
        if mm_data_dict['x_seen'] is None:
            mm_data_dict['x_seen'] = x_seen
        else:
            mm_data_dict['x_seen'] = torch.logical_or(
                x_seen, mm_data_dict['x_seen'])

        return mm_data_dict

    def forward_conv(self, mod_data, reset=True):
        """
        Conv on the modality data. The modality data holder
        carries a feature tensor per modality settings. Hence the
        modality features are provided as a list of tensors.
        Update modality features and mappings wrt modality scale. If
        `self.interpolate`, do not modify the mappings' scale, so that
        the features can be interpolated to the input resolution.

        Note that convolved features are preserved in the modality
        data holder, to be later used in potential downstream
        modules.

        :param mod_data:
        :param reset:
        :return:
        """
        if not self.conv:
            return mod_data

        # If the modality carries multi-setting data, recursive scheme
        if isinstance(mod_data.x, list):
            for i in range(len(mod_data)):
                mod_data[i].x = self.forward_conv(mod_data[i], i == 0).x
            return mod_data

        # If checkpointing the conv, need to set requires_grad for input
        # tensor because checkpointing the first layer breaks the
        # gradients
        if 'c' in self.checkpointing:
            mod_x = checkpoint(
                self.conv, mod_data.x.requires_grad_(),
                torch.BoolTensor([reset]))
        else:
            mod_x = self.conv(mod_data.x, True)
        mod_data.x = mod_x

        return mod_data

    def forward_atomic_pool(self, x_3d, x_mod, csr_idx):
        """Atomic pooling of the modality features on each separate
        setting.

        :param x_3d:
        :param x_mod:
        :param csr_idx:
        :return:
        """
        # If the modality carries multi-setting data, recursive scheme
        if isinstance(x_mod, list):
            x_mod = [
                self.forward_atomic_pool(x_3d, x, i)
                for x, i in zip(x_mod, csr_idx)]
            return x_mod

        if 'a' in self.checkpointing:
            x_mod = checkpoint(self.atomic_pool, x_3d, x_mod, None, csr_idx)
        else:
            x_mod = self.atomic_pool(x_3d, x_mod, None, csr_idx)
        return x_mod

    def forward_view_pool(self, x_3d, x_mod, mod_data):
        """View pooling of the modality features.

        :param x_3d:
        :param x_mod:
        :param mod_data:
        :return:
        """
        is_multi_shape = isinstance(x_mod, list)

        # For multi-setting data, concatenate view-level features from
        # each input modality setting and sort them to a CSR-friendly
        # order wrt 3D points features
        if is_multi_shape:
            idx_sorting = mod_data.view_cat_sorting
            x_mod = torch.cat(x_mod, dim=0)[idx_sorting]
            x_map = torch.cat(mod_data.mapping_features, dim=0)[idx_sorting]

        # View pooling of the atomic-pooled modality features
        if is_multi_shape:
            csr_idx = mod_data.view_cat_csr_indexing
        else:
            csr_idx = mod_data.view_csr_indexing

        # Here we keep track of the latest x_mod, x_map and csr_idx
        # in the modality data so as to recover it at the end of a
        # multimodal encoder or UNet. This is necessary when
        # training on a view-level loss.
        if self.keep_last_view:
            mod_data.last_view_x_mod = x_mod
            mod_data.last_view_x_map = x_map
            mod_data.last_view_csr_idx = csr_idx

        if 'v' in self.checkpointing:
            x_mod = checkpoint(self.view_pool, x_3d, x_mod, x_map, csr_idx)
        else:
            x_mod = self.view_pool(x_3d, x_mod, x_map, csr_idx)
        return x_mod, mod_data, csr_idx

    def forward_fusion(self, x_3d, x_mod):
        """Fuse the modality features into the 3D points features.

        :param x_3d:
        :param x_mod:
        :return:
        """
        if 'f' in self.checkpointing:
            x_3d = checkpoint(self.fusion, x_3d, x_mod)
        else:
            x_3d = self.fusion(x_3d, x_mod)
        return x_3d

    def forward_dropout(self, x_3d, x_mod, mod_data):
        if self.drop_3d:
            x_3d = self.drop_3d(x_3d)
        if self.drop_mod:
            x_mod = self.drop_mod(x_mod)
            if self.keep_last_view:
                mod_data.last_view_x_mod = self.drop_mod(mod_data.last_view_x_mod)
        return x_3d, x_mod, mod_data

    def extra_repr(self) -> str:
        repr_attr = ['drop_3d', 'drop_mod', 'keep_last_view', 'checkpointing']
        return "\n".join([f'{a}={getattr(self, a)}' for a in repr_attr])


class MVFusionUnimodalBranch(nn.Module, ABC):
    """Unimodal block with downsampling that looks like:

    IN 3D   ------------------------------------           --  OUT 3D
                                   \            \         /
                       Atomic Pool -- View Pool -- Fusion
                     /
    IN Mod  -- Conv -----------------------------------------  OUT Mod

    The convolution may be a down-convolution or preserve input shape.
    However, up-convolutions are not supported, because reliable the
    mappings cannot be inferred when increasing resolution.
    """

    def __init__(
            self, conv, atomic_pool, view_pool, fusion, drop_3d=0, drop_mod=0,
            hard_drop=False, keep_last_view=False, checkpointing='',
            out_channels=None, interpolate=False, transformer_config=None):
        super(MVFusionUnimodalBranch, self).__init__()
        
        self.transformerfusion = DVA_cls_5_fusion_7(transformer_config)
        self.n_classes = transformer_config['n_classes']
#         self.conv = conv
#         self.atomic_pool = atomic_pool
#         self.view_pool = view_pool
        self.gating = view_pool if transformer_config['gating'] is True else None
        self.fusion = fusion
    
        drop_cls = ModalityDropout if hard_drop else nn.Dropout
        self.drop_3d = drop_cls(p=drop_3d, inplace=False) \
            if drop_3d is not None and drop_3d > 0 \
            else None
        self.drop_mod = drop_cls(p=drop_mod, inplace=True) \
            if drop_mod is not None and drop_mod > 0 \
            else None
        self.keep_last_view = keep_last_view
        self._out_channels = out_channels
        self.interpolate = interpolate

        # Optional checkpointing to alleviate memory at train time.
        # Character rules:
        #     c: convolution
        #     a: atomic pooling
        #     v: view pooling
        #     f: fusion
        assert not checkpointing or isinstance(checkpointing, str),\
            f'Expected checkpointing to be of type str but received ' \
            f'{type(checkpointing)} instead.'
        self.checkpointing = ''.join(set('cavf').intersection(set(checkpointing)))

    @property
    def out_channels(self):
        if self._out_channels is None:
            raise ValueError(
                f'{self.__class__.__name__}.out_channels has not been '
                f'set. Please set it to allow inference even when the '
                f'modality has no data.')
        return self._out_channels

    def forward(self, mm_data_dict, modality):
        # Unpack the multimodal data dictionary. Specific treatment for
        # MinkowskiEngine and TorchSparse SparseTensors
        is_sparse_3d = not isinstance(
            mm_data_dict['x_3d'], (torch.Tensor, type(None)))
        x_3d = mm_data_dict['x_3d'].F if is_sparse_3d else mm_data_dict['x_3d']
        mod_data = mm_data_dict['modalities'][modality]
        


        # Check whether the modality carries multi-setting data
        is_multi_shape = isinstance(mod_data.x, list)

        # If the modality has no data mapped to the current 3D points,
        # ignore the branch forward. `self.out_channels` will guide us
        # on how to replace expected modality features
        if is_multi_shape and all([e.x.shape[0] == 0 for e in mod_data]) \
                or is_multi_shape and len(mod_data) == 0 \
                or not is_multi_shape and mod_data.x.shape[0] == 0:

            # Prepare the channel sizes
            nc_out = self.out_channels
            nc_3d = x_3d.shape[1]
            nc_2d = nc_out - nc_3d if nc_out > nc_3d else nc_3d

            # Make sure we have a valid `self.out_channels` so we can
            # simulate the forward without any modality data
            if nc_out < nc_3d:
                raise ValueError(
                    f'{self.__class__.__name__}.out_channels is smaller than '
                    f'number of features in x_3d: {nc_out} < {nc_3d}')

            # No points are seen
            # x_seen = torch.zeros(nc_3d, dtype=torch.bool)

            # Modify the feature dimension of mod_data to simulate
            # convolutions too
            if not is_multi_shape:
                mod_data.x = mod_data.x[:, [0]].repeat_interleave(nc_2d, dim=1)
            elif len(mod_data) > 0:
                mod_data.x = [
                    x[:, [0]].repeat_interleave(nc_2d, dim=1)
                    for x in mod_data.x]

            # For concatenation fusion, create zero features to
            # 'simulate' concatenation of modality features to x_3d
            if nc_out > nc_3d:
                zeros = torch.zeros_like(x_3d[:, [0]])
                zeros = zeros.repeat_interleave(nc_2d, dim=1)
                x_3d = torch.cat((x_3d, zeros), dim=1)

            # Return the modified multimodal data dictionary despite the
            # absence of modality features
            if is_sparse_3d:
                mm_data_dict['x_3d'].F = x_3d
            else:
                mm_data_dict['x_3d'] = x_3d
            mm_data_dict['modalities'][modality] = mod_data
            # if mm_data_dict['x_seen'] is None:
            #     mm_data_dict['x_seen'] = x_seen
            # else:
            #     mm_data_dict['x_seen'] = torch.logical_or(
            #         x_seen, mm_data_dict['x_seen'])

            return mm_data_dict

        # If the modality has a data list format and that one of the
        # items is an empty feature map, run a recursive forward on the
        # mm_data_dict with these problematic items discarded. This is
        # necessary whenever an element of the batch has no mappings to
        # the modality
        if is_multi_shape and any([e.x.shape[0] == 0 for e in mod_data]):

            # Remove problematic elements from mod_data
            num = len(mod_data)
            removed = {
                i: e for i, e in enumerate(mod_data) if e.x.shape[0] == 0}
            indices = [i for i in range(num) if i not in removed.keys()]
            mm_data_dict['modalities'][modality] = mod_data[indices]

            # Run forward recursively
            mm_data_dict = self.forward(mm_data_dict, modality)

            # Restore problematic elements. This is necessary if we need
            # to restore the initial batch elements with methods such as
            # `MMBatch.to_mm_data_list`
            mod_data = mm_data_dict['modalities'][modality]
            kept = {k: e for k, e in zip(indices, mod_data)}
            joined = {**kept, **removed}
            mod_data = mod_data.__class__([joined[i] for i in range(num)])
            mm_data_dict['modalities'][modality] = mod_data

            return mm_data_dict
 
        x_mod = self.forward_transformerfusion(mm_data_dict)
            
#         # Forward pass with `self.conv`
#         print("mod_data: ", mod_data)   # ImageBatch(num_settings=1, num_views=3, num_points=11819, device=cuda:0)
#         mod_data = self.forward_conv(mod_data)
#         print("mod_data = self.forward_conv(mod_data)")
#         print("mod_data: ", mod_data)

#         # Extract mapped features from the feature maps of each input
#         # modality setting
#         x_mod = mod_data.get_mapped_features(interpolate=self.interpolate)
#         print("x_mod = mod_data.get_mapped_features(interpolate=self.interpolate)")
#         print("x_mod[0]: ", x_mod[0], x_mod[0].shape)

        ### x_mod before and after atomic pool is the same if there is only 1 setting
        """
        # Atomic pooling of the modality features on each separate
        # setting
        x_mod = self.forward_atomic_pool(x_3d, x_mod, mod_data.atomic_csr_indexing)
        print("x_mod = self.forward_atomic_pool(x_3d, x_mod, mod_data.atomic_csr_indexing)")
        print("x_mod: ", x_mod)
        """
        
#         # View pooling of the modality features
#         x_mod, mod_data, csr_idx = self.forward_view_pool(x_3d, x_mod, mod_data)
#         print("x_mod, mod_data, csr_idx = self.forward_view_pool(x_3d, x_mod, mod_data)")
#         print("x_mod: ", x_mod, x_mod.shape)
#         print("mod_data: ", mod_data)
#         print("csr_idx: ", csr_idx)

        # Dropout 3D or modality features
        x_3d, x_mod, mod_data = self.forward_dropout(x_3d, x_mod, mod_data)

        # Fuse the modality features into the 3D points features
        x_3d = self.forward_fusion(x_3d, x_mod)

        # In case it has not been provided at initialization, save the
        # output channel size. This is useful for when a batch has no
        # modality data
        if self._out_channels is None:
            self._out_channels = x_3d.shape[1]

        # Boolean mask of seen points (including thoes that were not used as
        # input to the Transformer)
        csr_idx = mod_data[0].view_csr_indexing
        x_seen = csr_idx[1:] > csr_idx[:-1]       
            
        # Update the multimodal data dictionary
        # TODO: does the modality-driven sequence of updates on x_3d
        #  and x_seen affect the modality behavior ? Should the shared
        #  3D information only be updated once all modality branches
        #  have been run on the same input ?
        if is_sparse_3d:
            mm_data_dict['x_3d'].F = x_3d
        else:
            mm_data_dict['x_3d'] = x_3d
        mm_data_dict['modalities'][modality] = mod_data
        if mm_data_dict['x_seen'] is None:
            mm_data_dict['x_seen'] = x_seen
        else:
            mm_data_dict['x_seen'] = torch.logical_or(
                x_seen, mm_data_dict['x_seen'])

        return mm_data_dict

    def forward_transformerfusion(self, mm_data_dict, reset=True):
        ### multi-view mapping & M2F feature fusion using Transformer 
        
        # Features from only seen point-image matches are included in 'x'
        viewing_feats = mm_data_dict['transformer_input'][:, :, :-1]
        m2f_feats = mm_data_dict['transformer_input'][:, :, -1]
        
        # One hot features of M2F preds
        m2f_feats = torch.nn.functional.one_hot(m2f_feats.squeeze().long(), self.n_classes)
        
        # Uncomment to run without M2F labels (all labels are set to 0)
#         m2f_feats = torch.zeros((m2f_feats.shape[0], m2f_feats.shape[1], self.n_classes), device=m2f_feats.device)
    
        ### Multi-view fusion of M2F and viewing conditions using Transformer
        # TODO: remove assumption that pixel validity is the 1st feature
        invalid_pixel_mask = (viewing_feats[:, :, 0] == 0)   
        
                        
        if 'c' in self.checkpointing:
            fusion_input = {
                'invalid_pixels_mask': invalid_pixel_mask,
                'viewing_features': viewing_feats.requires_grad_(),
                'one_hot_mask_labels': m2f_feats
            }
            seen_x_mod = checkpoint(self.transformerfusion, fusion_input)
        else:
            fusion_input = {
                'invalid_pixels_mask': invalid_pixel_mask,
                'viewing_features': viewing_feats,
                'one_hot_mask_labels': m2f_feats
            }
            seen_x_mod = self.transformerfusion(fusion_input)
            
            
        ### Gating mechanism
        # Remove multi-view predicted masks and/or
        # geometric information if these provided no beneficial information 
        # for the 3D point.
        if self.gating is not None:
            seen_x_mod = self.gating(seen_x_mod)


        # Assign fused features back to points that were seen 
        x_mod = torch.zeros((mm_data_dict['modalities']['image'].num_points, 
                             seen_x_mod.shape[-1]), device=seen_x_mod.device)
        x_mod[mm_data_dict['transformer_x_seen']] = seen_x_mod
        return x_mod
            
    
#     def forward_conv(self, mod_data, reset=True):
#         """
#         Conv on the modality data. The modality data holder
#         carries a feature tensor per modality settings. Hence the
#         modality features are provided as a list of tensors.
#         Update modality features and mappings wrt modality scale. If
#         `self.interpolate`, do not modify the mappings' scale, so that
#         the features can be interpolated to the input resolution.

#         Note that convolved features are preserved in the modality
#         data holder, to be later used in potential downstream
#         modules.

#         :param mod_data:
#         :param reset:
#         :return:
#         """
#         if not self.conv:
#             return mod_data

#         # If the modality carries multi-setting data, recursive scheme
#         if isinstance(mod_data.x, list):
#             for i in range(len(mod_data)):
#                 mod_data[i].x = self.forward_conv(mod_data[i], i == 0).x
#             return mod_data

#         # If checkpointing the conv, need to set requires_grad for input
#         # tensor because checkpointing the first layer breaks the
#         # gradients
#         if 'c' in self.checkpointing:
#             mod_x = checkpoint(
#                 self.conv, mod_data.x.requires_grad_(),
#                 torch.BoolTensor([reset]))
#         else:
#             mod_x = self.conv(mod_data.x, True)
#         mod_data.x = mod_x

#         return mod_data

#     def forward_atomic_pool(self, x_3d, x_mod, csr_idx):
#         """Atomic pooling of the modality features on each separate
#         setting.

#         :param x_3d:
#         :param x_mod:
#         :param csr_idx:
#         :return:
#         """
#         # If the modality carries multi-setting data, recursive scheme
#         if isinstance(x_mod, list):
#             x_mod = [
#                 self.forward_atomic_pool(x_3d, x, i)
#                 for x, i in zip(x_mod, csr_idx)]
#             return x_mod

#         if 'a' in self.checkpointing:
#             x_mod = checkpoint(self.atomic_pool, x_3d, x_mod, None, csr_idx)
#         else:
#             x_mod = self.atomic_pool(x_3d, x_mod, None, csr_idx)
#         return x_mod

#     def forward_view_pool(self, x_3d, x_mod, mod_data):
#         """View pooling of the modality features.

#         :param x_3d:
#         :param x_mod:
#         :param mod_data:
#         :return:
#         """
#         is_multi_shape = isinstance(x_mod, list)

#         # For multi-setting data, concatenate view-level features from
#         # each input modality setting and sort them to a CSR-friendly
#         # order wrt 3D points features
#         if is_multi_shape:
#             idx_sorting = mod_data.view_cat_sorting
#             x_mod = torch.cat(x_mod, dim=0)[idx_sorting]
#             x_map = torch.cat(mod_data.mapping_features, dim=0)[idx_sorting]

#         # View pooling of the atomic-pooled modality features
#         if is_multi_shape:
#             csr_idx = mod_data.view_cat_csr_indexing
#         else:
#             csr_idx = mod_data.view_csr_indexing

#         # Here we keep track of the latest x_mod, x_map and csr_idx
#         # in the modality data so as to recover it at the end of a
#         # multimodal encoder or UNet. This is necessary when
#         # training on a view-level loss.
#         if self.keep_last_view:
#             mod_data.last_view_x_mod = x_mod
#             mod_data.last_view_x_map = x_map
#             mod_data.last_view_csr_idx = csr_idx

#         if 'v' in self.checkpointing:
#             x_mod = checkpoint(self.view_pool, x_3d, x_mod, x_map, csr_idx)
#         else:
#             x_mod = self.view_pool(x_3d, x_mod, x_map, csr_idx)
#         return x_mod, mod_data, csr_idx

    def forward_fusion(self, x_3d, x_mod):
        """Fuse the modality features into the 3D points features.

        :param x_3d:
        :param x_mod:
        :return:
        """
        if 'f' in self.checkpointing:
            x_3d = checkpoint(self.fusion, x_3d, x_mod)
        else:
            x_3d = self.fusion(x_3d, x_mod)
        return x_3d

    def forward_dropout(self, x_3d, x_mod, mod_data):
        if self.drop_3d:
            x_3d = self.drop_3d(x_3d)
        if self.drop_mod:
            x_mod = self.drop_mod(x_mod)
            if self.keep_last_view:
                mod_data.last_view_x_mod = self.drop_mod(mod_data.last_view_x_mod)
        return x_3d, x_mod, mod_data

    def extra_repr(self) -> str:
        repr_attr = ['drop_3d', 'drop_mod', 'keep_last_view', 'checkpointing']
        return "\n".join([f'{a}={getattr(self, a)}' for a in repr_attr])

    
    
class IdentityBranch(BaseModule):
    def __init__(self):
        super(IdentityBranch, self).__init__()

    def forward(self, mm_data_dict, modality):
        return mm_data_dict
    
class MVAttentionUnimodalBranch(nn.Module, ABC):
    def __init__(
            self, conv, atomic_pool, view_pool, fusion, drop_3d=0, drop_mod=0,
            hard_drop=False, keep_last_view=False, checkpointing='',
            out_channels=None, interpolate=False, transformer_config=None):
        super(MVAttentionUnimodalBranch, self).__init__()
        
        self.use_3D = transformer_config['use_3D']
        
        self.use_transformer = False
        self.use_deepset = False
        self.use_random = False
        self.use_average = False
        
        self.gating = None
        self.n_classes = transformer_config['n_classes']
        if transformer_config.use_transformer:
            self.use_transformer = True
            self.attn_fusion = DVA_attention_weighted_M2F_preds(transformer_config)
        elif transformer_config.use_deepset:
            self.use_deepset = True
            d_in = 8 + self.n_classes   # Viewing Conditions + Input pred label (one hot)
            d_hidden = 32
            pool = 'max'
            use_num = True
             
            if self.use_3D:
                self.attn_fusion = DeepSetFeat_ViewFusion(d_in=d_in, d_out=d_hidden, pool=pool, fusion='concatenation',
                                                          use_num=use_num, num_classes=self.n_classes)
            else:
                self.attn_fusion = DeepSetFeat_AttentionWeighting(d_in=d_in, d_out=d_hidden, pool=pool, fusion='concatenation',
                use_num=use_num, num_classes=self.n_classes)
        elif transformer_config.use_random:
            self.use_random = True
#         elif transformer_config.use_heuristic:
#             self.use_heuristic = True
        elif transformer_config.use_average:
            self.use_average = True
            
        self.fusion = fusion
    
        drop_cls = ModalityDropout if hard_drop else nn.Dropout
        self.drop_3d = drop_cls(p=drop_3d, inplace=False) \
            if drop_3d is not None and drop_3d > 0 \
            else None
        self.drop_mod = drop_cls(p=drop_mod, inplace=True) \
            if drop_mod is not None and drop_mod > 0 \
            else None
        self.keep_last_view = keep_last_view
        self._out_channels = out_channels
        self.interpolate = interpolate

        # Optional checkpointing to alleviate memory at train time.
        # Character rules:
        #     c: convolution
        #     a: atomic pooling
        #     v: view pooling
        #     f: fusion
        assert not checkpointing or isinstance(checkpointing, str),\
            f'Expected checkpointing to be of type str but received ' \
            f'{type(checkpointing)} instead.'
        self.checkpointing = ''.join(set('cavf').intersection(set(checkpointing)))

    @property
    def out_channels(self):
        if self._out_channels is None:
            raise ValueError(
                f'{self.__class__.__name__}.out_channels has not been '
                f'set. Please set it to allow inference even when the '
                f'modality has no data.')
        return self._out_channels

    def forward(self, mm_data_dict, modality):
        # Unpack the multimodal data dictionary. Specific treatment for
        # MinkowskiEngine and TorchSparse SparseTensors
        is_sparse_3d = not isinstance(
            mm_data_dict['x_3d'], (torch.Tensor, type(None)))
        x_3d = mm_data_dict['x_3d'].F if is_sparse_3d else mm_data_dict['x_3d']
        mod_data = mm_data_dict['modalities'][modality]
        


        # Check whether the modality carries multi-setting data
        is_multi_shape = isinstance(mod_data.x, list)

        # If the modality has no data mapped to the current 3D points,
        # ignore the branch forward. `self.out_channels` will guide us
        # on how to replace expected modality features
        if is_multi_shape and all([e.x.shape[0] == 0 for e in mod_data]) \
                or is_multi_shape and len(mod_data) == 0 \
                or not is_multi_shape and mod_data.x.shape[0] == 0:

            # Prepare the channel sizes
            nc_out = self.out_channels
            nc_3d = x_3d.shape[1]
            nc_2d = nc_out - nc_3d if nc_out > nc_3d else nc_3d

            # Make sure we have a valid `self.out_channels` so we can
            # simulate the forward without any modality data
            if nc_out < nc_3d:
                raise ValueError(
                    f'{self.__class__.__name__}.out_channels is smaller than '
                    f'number of features in x_3d: {nc_out} < {nc_3d}')

            # No points are seen
            # x_seen = torch.zeros(nc_3d, dtype=torch.bool)

            # Modify the feature dimension of mod_data to simulate
            # convolutions too
            if not is_multi_shape:
                mod_data.x = mod_data.x[:, [0]].repeat_interleave(nc_2d, dim=1)
            elif len(mod_data) > 0:
                mod_data.x = [
                    x[:, [0]].repeat_interleave(nc_2d, dim=1)
                    for x in mod_data.x]

            # For concatenation fusion, create zero features to
            # 'simulate' concatenation of modality features to x_3d
            if nc_out > nc_3d:
                zeros = torch.zeros_like(x_3d[:, [0]])
                zeros = zeros.repeat_interleave(nc_2d, dim=1)
                x_3d = torch.cat((x_3d, zeros), dim=1)

            # Return the modified multimodal data dictionary despite the
            # absence of modality features
            if is_sparse_3d:
                mm_data_dict['x_3d'].F = x_3d
            else:
                mm_data_dict['x_3d'] = x_3d
            mm_data_dict['modalities'][modality] = mod_data
            # if mm_data_dict['x_seen'] is None:
            #     mm_data_dict['x_seen'] = x_seen
            # else:
            #     mm_data_dict['x_seen'] = torch.logical_or(
            #         x_seen, mm_data_dict['x_seen'])

            return mm_data_dict

        # If the modality has a data list format and that one of the
        # items is an empty feature map, run a recursive forward on the
        # mm_data_dict with these problematic items discarded. This is
        # necessary whenever an element of the batch has no mappings to
        # the modality
        if is_multi_shape and any([e.x.shape[0] == 0 for e in mod_data]):

            # Remove problematic elements from mod_data
            num = len(mod_data)
            removed = {
                i: e for i, e in enumerate(mod_data) if e.x.shape[0] == 0}
            indices = [i for i in range(num) if i not in removed.keys()]
            mm_data_dict['modalities'][modality] = mod_data[indices]

            # Run forward recursively
            mm_data_dict = self.forward(mm_data_dict, modality)

            # Restore problematic elements. This is necessary if we need
            # to restore the initial batch elements with methods such as
            # `MMBatch.to_mm_data_list`
            mod_data = mm_data_dict['modalities'][modality]
            kept = {k: e for k, e in zip(indices, mod_data)}
            joined = {**kept, **removed}
            mod_data = mod_data.__class__([joined[i] for i in range(num)])
            mm_data_dict['modalities'][modality] = mod_data

            return mm_data_dict
 
        if self.use_transformer:
            x_mod = self.forward_transformerfusion(mm_data_dict)
        elif self.use_deepset:
            x_mod = self.forward_deepset(mm_data_dict)
        elif self.use_random:
            x_mod = self.forward_random(mm_data_dict)
        elif self.use_average:
            x_mod = self.forward_average(mm_data_dict)
       

        # Dropout 3D or modality features
        x_3d, x_mod, mod_data = self.forward_dropout(x_3d, x_mod, mod_data)

        # Fuse the modality features into the 3D points features
        x_3d = self.forward_fusion(x_3d, x_mod)

        # In case it has not been provided at initialization, save the
        # output channel size. This is useful for when a batch has no
        # modality data
        if self._out_channels is None:
            self._out_channels = x_3d.shape[1]

        # Boolean mask of seen points (including thoes that were not used as
        # input to the Transformer)
        csr_idx = mod_data[0].view_csr_indexing
        x_seen = csr_idx[1:] > csr_idx[:-1]       
            
        # Update the multimodal data dictionary
        # TODO: does the modality-driven sequence of updates on x_3d
        #  and x_seen affect the modality behavior ? Should the shared
        #  3D information only be updated once all modality branches
        #  have been run on the same input ?
        if is_sparse_3d:
            mm_data_dict['x_3d'].F = x_3d
        else:
            mm_data_dict['x_3d'] = x_3d
        mm_data_dict['modalities'][modality] = mod_data
        if mm_data_dict['x_seen'] is None:
            mm_data_dict['x_seen'] = x_seen
        else:
            mm_data_dict['x_seen'] = torch.logical_or(
                x_seen, mm_data_dict['x_seen'])

        return mm_data_dict

    def forward_transformerfusion(self, mm_data_dict, reset=True):
        ### multi-view mapping & M2F feature fusion using Transformer 
        
        # Features from only seen point-image matches are included in 'x'
        viewing_feats = mm_data_dict['transformer_input'][:, :, :-1]
        m2f_feats = mm_data_dict['transformer_input'][:, :, -1]
        
        # One hot features of M2F preds
        m2f_feats = torch.nn.functional.one_hot(m2f_feats.squeeze().long(), self.n_classes)
    
        ### Multi-view fusion of M2F and viewing conditions using Transformer
        # TODO: remove assumption that pixel validity is the 1st feature
        invalid_pixel_mask = (viewing_feats[:, :, 0] == 0)   
        
                        
        if 'c' in self.checkpointing:
            fusion_input = {
                'invalid_pixels_mask': invalid_pixel_mask,
                'viewing_features': viewing_feats.requires_grad_(),
                'one_hot_mask_labels': m2f_feats
            }
            seen_x_mod = checkpoint(self.attn_fusion, fusion_input)
        else:
            fusion_input = {
                'invalid_pixels_mask': invalid_pixel_mask,
                'viewing_features': viewing_feats,
                'one_hot_mask_labels': m2f_feats
            }
            seen_x_mod = self.attn_fusion(fusion_input)
            
            
        ### Gating mechanism
        # Remove multi-view predicted masks and/or
        # geometric information if these provided no beneficial information 
        # for the 3D point.
        if self.gating is not None:
            seen_x_mod = self.gating(seen_x_mod)


        # Assign fused features back to points that were seen 
        x_mod = torch.zeros((mm_data_dict['modalities']['image'].num_points, 
                             seen_x_mod.shape[-1]), device=seen_x_mod.device)
        x_mod[mm_data_dict['transformer_x_seen']] = seen_x_mod
        return x_mod

    def forward_deepset(self, mm_data_dict, reset=True):
#         print(mm_data_dict['modalities'])
                
        viewing_conditions = mm_data_dict['modalities']['image'][0].mappings.values[2]

        input_preds = mm_data_dict['modalities']['image'][0].get_mapped_m2f_features()
        input_preds_one_hot = torch.nn.functional.one_hot(input_preds.long().squeeze(), self.n_classes)
        attention_input = torch.concat((viewing_conditions, input_preds_one_hot), dim=1)

        csr_idx = mm_data_dict['modalities']['image'][0].view_csr_indexing
        
        n_seen = (csr_idx[1:] - csr_idx[:-1])
        
#         print("n seen points: ", (n_seen > 0).sum())
        
#         print('total n seen', n_seen.sum())
        
#         print(attention_input.shape, csr_idx.shape, input_preds_one_hot.shape)
        seen_x_mod = self.attn_fusion(attention_input, csr_idx, input_preds_one_hot)        
       

       
        if self.use_3D:
            # Assign fused features back to points that were seen 
#             x_mod = torch.zeros((mm_data_dict['modalities']['image'].num_points, 
#                                  seen_x_mod.shape[-1]), device=seen_x_mod.device)
            
# #             print(mm_data_dict['orig_data'])
            
#             csr_idx = mm_data_dict['modalities']['image'][0].view_csr_indexing # mm_data_dict['orig_data']['modalities']['image'][0].view_csr_indexing
            
            
            
#             x_seen = csr_idx[1:] > csr_idx[:-1]       
            
#             print(x_mod.shape, x_seen.shape, seen_x_mod.shape)
            x_mod = seen_x_mod
        else:
            x_mod = seen_x_mod
            
        return x_mod
    
    def forward_random(self, mm_data_dict, reset=True):
        input_preds = mm_data_dict['transformer_input'][:, 0, -1].long()
        
        
        selected_view_preds_one_hot = torch.nn.functional.one_hot(input_preds.squeeze(), self.n_classes)
    
        # Assign fused features back to points that were seen 
        x_mod = torch.zeros((mm_data_dict['modalities']['image'].num_points, 
                             selected_view_preds_one_hot.shape[-1]), device=selected_view_preds_one_hot.device)
        x_mod[mm_data_dict['transformer_x_seen']] = selected_view_preds_one_hot.float()
        return x_mod
    
    
    def forward_average(self, mm_data_dict, reset=True):
        input_preds = mm_data_dict['transformer_input'][:, 0, -1].long()
        

        
        mode_preds_one_hot = torch.nn.functional.one_hot(input_preds.squeeze(), self.n_classes)
        
    
        # Assign fused features back to points that were seen 
        x_mod = torch.zeros((mm_data_dict['modalities']['image'].num_points, 
                             mode_preds_one_hot.shape[-1]), device=mode_preds_one_hot.device)
        x_mod[mm_data_dict['transformer_x_seen']] = mode_preds_one_hot.float()
        return x_mod
    
    
    def forward_fusion(self, x_3d, x_mod):
        """Fuse the modality features into the 3D points features.

        :param x_3d:
        :param x_mod:
        :return:
        """
        if 'f' in self.checkpointing:
            x_3d = checkpoint(self.fusion, x_3d, x_mod)
        else:
            x_3d = self.fusion(x_3d, x_mod)
        return x_3d

    def forward_dropout(self, x_3d, x_mod, mod_data):
        if self.drop_3d:
            x_3d = self.drop_3d(x_3d)
        if self.drop_mod:
            x_mod = self.drop_mod(x_mod)
            if self.keep_last_view:
                mod_data.last_view_x_mod = self.drop_mod(mod_data.last_view_x_mod)
        return x_3d, x_mod, mod_data

    def extra_repr(self) -> str:
        repr_attr = ['drop_3d', 'drop_mod', 'keep_last_view', 'checkpointing']
        return "\n".join([f'{a}={getattr(self, a)}' for a in repr_attr])
    
    
    
    
    
    
    
    
    
################################## Adjusted DeepSetFeat for view selection experiment   ##################################
import sys
import torch
import torch.nn.functional as F
from torch_scatter import segment_csr, scatter_min, scatter_max
from torch_points3d.core.common_modules import MLP
import math

# New improved
class DeepSetFeat_ViewFusion(nn.Module, ABC):
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
        super(DeepSetFeat_ViewFusion, self).__init__()

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
        self.E_score = nn.Linear(d_out, d_out, bias=True)
        
        self.d_out = d_out
        
        self.num_classes = num_classes

    def forward(self, x, csr_idx, x_mod):
        
#         print('x', x.shape)
        
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
        
        # Compute compatibilities (unscaled scores) : V x d_out
        compatibilities = self.E_score(x_out)
        

        # Compute attention scores : V x d_out
        attentions = segment_softmax_csr(
            compatibilities, csr_idx, scaling=False)
                
        # Apply attention scores : P x d_out
        x_pool = segment_csr(
            x_out * expand_group_feat(attentions, self.d_out, self.d_out),
            csr_idx, reduce='sum')
        
#         print(x_out.shape, x_pool.shape, attentions.shape, csr_idx.shape)
        
        return x_pool

    def extra_repr(self) -> str:
        repr_attr = ['pool', 'fusion', 'use_num']
        return "\n".join([f'{a}={getattr(self, a)}' for a in repr_attr])


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
