from abc import ABC

from ..scannet import *
from .utils import read_image_pose_pairs, read_image_pose_pairs_without_frameskip
from torch_points3d.core.multimodal.image import SameSettingImageData
from torch_points3d.datasets.base_dataset_multimodal import BaseDatasetMM
from torch_points3d.core.multimodal.data import MMData
from tqdm.auto import tqdm as tq
from itertools import repeat
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


import os
import os.path as osp
import time
import random
from torch_points3d.core.data_transform.multimodal.image import RandomHorizontalFlip



log = logging.getLogger(__name__)


########################################################################################
#                                                                                      #
#                            ScanNet image processing utils                            #
#                                                                                      #
########################################################################################

def load_pose(filename):
    """Read ScanNet pose file.
    Credit: https://github.com/angeladai/3DMV/blob/f889b531f8813d409253fe065fb9b18c5ca2b495/3dmv/data_util.py
    """
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
    out = torch.from_numpy(np.asarray(lines).astype(np.float32))
    return out



########################################################################################
#                                                                                      #
#                                 ScanNet 2D-3D dataset                                #
#                                                                                      #
########################################################################################

class ScannetMM(Scannet):
    """Scannet dataset for multimodal learning on 3D points and 2D images, you
    will have to agree to terms and conditions by hitting enter so that it
    downloads the dataset.

    http://www.scan-net.org/

    Inherits from `torch_points3d.datasets.segmentation.scannet.Scannet`.
    However, because 2D data is too heavy to be entirely loaded in memory, it
    does not follow the exact philosophy of `InMemoryDataset`. More
    specifically, 3D data is preprocessed and loaded in memory in the same way,
    while 2D data is preprocessed into per-scan files which are loaded at
    __getitem__ time.


    Parameters
    ----------
    root : str
        Path to the data
    split : str, optional
        Split used (train, val or test)
    transform (callable, optional):
        A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
    pre_transform (callable, optional):
        A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before being saved to disk.
    pre_filter (callable, optional):
        A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the final dataset.
    version : str, optional
        version of scannet, by default "v2"
    use_instance_labels : bool, optional
        Wether we use instance labels or not, by default False
    use_instance_bboxes : bool, optional
        Wether we use bounding box labels or not, by default False
    donotcare_class_ids : list, optional
        Class ids to be discarded
    max_num_point : [type], optional
        Max number of points to keep during the pre processing step
    process_workers : int, optional
        Number of process workers
    normalize_rgb : bool, optional
        Normalise rgb values, by default True
    types : list, optional
        File types to download from the ScanNet repository
    frame_depth : bool, optional
        Whether depth images should be exported from the `.sens`file
    frame_rgb : bool, optional
        Whether RGB images should be exported from the `.sens`file
    frame_pose : bool, optional
        Whether poses should be exported from the `.sens`file
    frame_intrinsics : bool, optional
        Whether intrinsic parameters should be exported from the `.sens`file
    frame_skip : int, optional
        Period of frames to skip when parsing the `.sens` frame streams. e.g. setting `frame_skip=50` will export 2% of the stream frames.
    is_test : bool, optional
        Switch to help debugging the dataset
    img_ref_size : tuple, optional
        The size at which rgb images will be processed
    pre_transform_image : (callable, optional):
        Image pre_transforms
    transform_image : (callable, optional):
        Image transforms
    """

    DEPTH_IMG_SIZE = (640, 480)

    def __init__(
            self, *args, img_ref_size=(320, 240), pre_transform_image=None,
            transform_image=None, neucon_metas_dir='', neucon_frame_skip=1, m2f_preds_dirname='', load_m2f_masks=False,
            undo_axis_align=False,
            center_xy=False,
            center_z=False, 
            n_views=None,
            n_views_ablation=None,
            store_mode_pred=False,
            store_random_pred=False,
        **kwargs):
        
        self.pre_transform_image = pre_transform_image
        self.transform_image = transform_image
        self.img_ref_size = img_ref_size
        self.neucon_metas_dir = neucon_metas_dir
        self.neucon_frame_skip = neucon_frame_skip
        self.m2f_preds_dirname = m2f_preds_dirname
        self.load_m2f_masks = load_m2f_masks
        self.undo_axis_align = undo_axis_align
        self.center_xy = center_xy
        self.center_z = center_z
        self.n_views = n_views
        self.n_views_ablation = n_views_ablation
        
        # only for experiments
        self.store_mode_pred = store_mode_pred
        self.store_random_pred = store_random_pred
        
        if store_mode_pred:
            print("Training using the mode prediction from all views for each point!")
        if store_random_pred:
            print("Selecting one random input prediction for each point from all of its views!")
        
        if self.n_views_ablation is not None:
            print("Currently running ablation studies on the number of points!")
            print("self.n_views_ablation ", self.n_views_ablation)

        super(ScannetMM, self).__init__(*args, **kwargs)
    
        
#         if self.split == 'train':
#             # Instead of saving all 3D pre-transformed scans in CPU memory, save each
#             # individually processed scan in processed_3d_'split' directory and load 
#             # individual scans in the __getitem__ method. This saves tremendous amounts
#             # of memory when many dataloader workers are created, but is slightly slower.
#             scan_id_to_name = getattr(
#                 self, f"MAPPING_IDX_TO_SCAN_{self.split.upper()}_NAMES")
#             data_dict = self.uncollate(self.data, self.slices, scan_id_to_name)
#             del self.data   # Keep self.slices for dataset functionality

#             # Export individual 3D Data to .pt files in processed_3d_'split' directory
#             out_dir = osp.join(self.processed_dir, f'processed_3d_{self.split}')
#             if not osp.exists(out_dir):
#                 os.makedirs(out_dir)
#             for scan_name, data in data_dict.items():
#                 processed_3d_scan_path = osp.join(out_dir, f'{scan_name}.pt')
#                 if not osp.exists(processed_3d_scan_path):
#                     torch.save(data.clone(), processed_3d_scan_path)
#             del data_dict
            
        
    def process(self):        
        if self.is_test:
            return

        # --------------------------------------------------------------
        # Preprocess 3D data
        # Download, pre_transform and pre_filter raw 3D data
        # Output will be saved to <SPLIT>.pt
        # --------------------------------------------------------------
        print("pre-processing 3D data")
        super(ScannetMM, self).process()

        # ----------------------------------------------------------
        # Recover image data and run image pre-transforms
        # This is where images are loaded and mappings are computed.
        # Output is saved to processed_2d_<SPLIT>/<SCAN_NAME>.pt
        # ----------------------------------------------------------
        for i, (scan_names, split) in enumerate(zip(self.scan_names, self.SPLITS)):
            
#             if split in ['train', 'val']:
                
#                 print(f"Skipping 2D preprocessing for {split} split!")
#                 continue

            print(f'\nStarting preprocessing 2d for {split} split')

            # Prepare the processed_2d_<SPLIT> directory where per-scan
            # images and mappings will be saved
            if not osp.exists(self.processed_2d_paths[i]):
                os.makedirs(self.processed_2d_paths[i])

            if set(os.listdir(self.processed_2d_paths[i])) == set([s + '.pt' for s in scan_names]):
                print(f'skipping split {split}')
                continue

            # Recover the collated Data (one for each scan) produced by
            # super().process(). Uncollate into a dictionary of Data
            print('Loading preprocessed 3D data...')
            scan_id_to_name = getattr(
                self, f"MAPPING_IDX_TO_SCAN_{split.upper()}_NAMES")
            data_collated, slices = torch.load(self.processed_paths[i])
            data_dict = self.uncollate(data_collated, slices, scan_id_to_name)
            del data_collated

            print('Preprocessing 2D data...')
            scannet_dir = osp.join(self.raw_dir, "scans" if split in ["train", "val"] else "scans_test")

            for scan_name in tq(scan_names):
                print("current scan name: ", scan_name, flush=True)

                scan_sens_dir = osp.join(scannet_dir, scan_name, 'sens')
                meta_file = osp.join(scannet_dir, scan_name, scan_name + ".txt")
                processed_2d_scan_path = osp.join(self.processed_2d_paths[i], scan_name + '.pt')

                if osp.exists(processed_2d_scan_path):
                    continue

                    
                # Recover the image-pose pairs
                image_info_list = [
                    {'path': i_file, 'extrinsic': load_pose(p_file)}
                    for i_file, p_file in read_image_pose_pairs(
                        osp.join(scan_sens_dir, 'color'),
                        osp.join(scan_sens_dir, 'pose'),
                        image_suffix='.jpg', pose_suffix='.txt', skip_freq=self.frame_skip,
                        neucon_metas_dir=self.neucon_metas_dir)]
                
#                 # TEMPORARY ADJUSTMENT TO PROCESS NEUCON META IDS AND TEST BENCHMARK IDS SIMULTANEOUSLY
#                 image_info_list = [
#                     {'path': i_file, 'extrinsic': load_pose(p_file)}
#                     for i_file, p_file in read_image_pose_pairs_without_frameskip(
#                         osp.join(scannet_dir, scan_name, 'color_resized'),
#                         osp.join(scan_sens_dir, 'pose'),
#                         image_suffix='.png', pose_suffix='.txt')]

                # Aggregate all RGB image paths
                path = np.array([info['path'] for info in image_info_list])

                # Aggregate all extrinsic 4x4 matrices
                # Train and val scans have undergone axis alignment
                # transformations. Need to recover and apply those
                # to camera poses too. Test scans have no axis
                # alignment
                axis_align_matrix = read_axis_align_matrix(meta_file)
                if axis_align_matrix is not None:
                    extrinsic = torch.cat([
                        axis_align_matrix.mm(info['extrinsic']).unsqueeze(0)
                        for info in image_info_list], dim=0)
                else:
                    extrinsic = torch.cat([
                        info['extrinsic'].unsqueeze(0)
                        for info in image_info_list], dim=0)

                # For easier image handling, extract the images
                # position from the extrinsic matrices
                xyz = extrinsic[:, :3, 3]

                # Read intrinsic parameters for the depth camera
                # because this is the one related to the extrinsic.
                # Strangely, using the color camera intrinsic along
                # with the pose does not produce the expected
                # projection
                intrinsic = load_pose(osp.join(
                    scan_sens_dir, 'intrinsic/intrinsic_depth.txt'))
                fx = intrinsic[0][0].repeat(len(image_info_list))
                fy = intrinsic[1][1].repeat(len(image_info_list))
                mx = intrinsic[0][2].repeat(len(image_info_list))
                my = intrinsic[1][2].repeat(len(image_info_list))

                # Save scan images as SameSettingImageData
                # NB: the image is first initialized to
                # DEPTH_IMG_SIZE because the intrinsic parameters
                # are defined accordingly. Setting ref_size
                # afterwards calls the @adjust_intrinsic method to
                # rectify the intrinsics consequently
                image_data = SameSettingImageData(
                    ref_size=self.DEPTH_IMG_SIZE, proj_upscale=1,
                    path=path, pos=xyz, fx=fx, fy=fy, mx=mx, my=my,
                    extrinsic=extrinsic)
                image_data.ref_size = self.img_ref_size

                # Run image pre-transform
                if self.pre_transform_image is not None:
                    _, image_data = self.pre_transform_image(
                        data_dict[scan_name], image_data)

                # Save scan 2D data to a file that will be loaded when
                # __get_item__ is called
                print("Saving image data in : ", processed_2d_scan_path, flush=True)
                torch.save(image_data, processed_2d_scan_path)

    @property
    def processed_file_names(self):
        return [f"{s}.pt" for s in Scannet.SPLITS] + self.processed_2d_paths

    @property
    def processed_2d_paths(self):
        return [osp.join(self.processed_dir, f"processed_2d_{s}") for s in Scannet.SPLITS]

    def __getitem__(self, idx):
        """
        Indexing mechanism for the Dataset. Only supports int indexing.

        Overwrites the torch_geometric.InMemoryDataset.__getitem__()
        used for indexing Dataset. Extends its mechanisms to multimodal
        data.

        Get a ScanNet scene 3D points as Data along with 2D image data
        and mapping, along with the list idx.

        Note that 3D data is already loaded in memory, which allows fast
        indexing. 2D data, however, is too heavy to be entirely kept in
        memory, so it is loaded on the fly from preprocessing files.
        """
        assert isinstance(idx, int), \
            f"Indexing with {type(idx)} is not supported, only " \
            f"{int} are accepted."
        
        scan_id_to_name = getattr(
            self, f"MAPPING_IDX_TO_SCAN_{self.split.upper()}_NAMES")
        scan_name = scan_id_to_name[idx]     
        
#         if self.split == 'train':
#             # Load individual scan from preprocessed_3d_'split' directory   
#             processed_3d_scan_path = osp.join(self.processed_dir, 
#                                               f'processed_3d_{self.split}', f'{scan_name}.pt')
#             data = torch.load(processed_3d_scan_path)
#         else:
#             # Get the 3D point sample
#             data = self.get(idx)
        
        # Get the 3D point sample
        data = self.get(idx)
        
#         #### Temporary solution to measure Cross-view consistency
#         def get_instance_labels(dset, scan_name):
#             scannet_dir = osp.join(dset.raw_dir, "scans" if dset.split in ["train", "val"] else "scans_test")
#             args = (
#                     scannet_dir,
#                     scan_name,
#                     dset.label_map_file,
#                     dset.donotcare_class_ids,
#                     dset.max_num_point,
#                     dset.VALID_CLASS_IDS,
#                     dset.normalize_rgb,
#                     dset.frame_depth,
#                     dset.frame_rgb,
#                     dset.frame_pose,
#                     dset.frame_intrinsics,
#                     dset.frame_skip,
#                 )

#             data = dset.read_one_scan(*args)
#             return data['instance_labels']
        
#         print("Adding instance labels to mm_data")
#         data['instance_labels'] = get_instance_labels(self, scan_name)
        
        
        
        

        # Load the corresponding 2D data and mappings
        i_split = self.SPLITS.index(self.split)
        images = torch.load(osp.join(
            self.processed_2d_paths[i_split], scan_name + '.pt'))
        
        # Initialize internal attributes because these were not implemented in the originally preprocessed data
        images.gt_mask = None
        images.gt_mask_path = None
        
        #### FENG: first undo alignment  
        if self.undo_axis_align and self.split != 'test':
            path = osp.join(self.root, 'raw', 'scans')
            axis_align_matrix_path = osp.join(path, scan_name, scan_name + '.txt')
            axis_align_matrix = read_axis_align_matrix(axis_align_matrix_path)
            
            inv = torch.linalg.inv(axis_align_matrix.T)
            data.pos = (torch.concat((data.pos, torch.ones((len(data.pos), 1))), axis=-1) @ inv)[:, :3]
    
            # Transform camera positions for visualization purposes
            images.pos = (torch.concat((images.pos, torch.ones((len(images.pos), 1))), axis=-1) @ inv.double())[:, :3] 
            images.extrinsic = inv.T  @ images.extrinsic    
 
        # apply 3D transforms
        data = data if self.transform is None else self.transform(data)
        
        #### FENG: then center pcd on X and Y after augmenting
        if self.center_xy:
            if self.center_z:
                data_mean = data.pos[:, :3].mean(0)
            else:
                # Z is not centered because originally, Z ranged from 0 to ~3 and centering would break this
                data_mean = torch.concat((data.pos[:, :2].mean(0), torch.zeros((1))), axis=-1)
            data.pos -= data_mean
            data.x[:, :3] = data.pos
            images.pos -= data_mean        
                
        # Run image transforms
        if self.transform_image is not None and self.load_m2f_masks is False:
            data, images = self.transform_image(data, images)
        else:
            for transform in self.transform_image.transforms:
                # Perform transform after loading M2F masks, otherwise 
                # those would not be flipped 
                if isinstance(transform, RandomHorizontalFlip):
                    randomhorizontalflip = transform
                    continue
                else:
                    data, images = transform(data, images)
                                
        
            
        # Load Mask2Former predicted masks if dirname is given in dataset config       
        if self.m2f_preds_dirname is not None and self.load_m2f_masks is True:
            if len(images) > 1:
                print(f"ImageData contains {len(images)} different camera settings!")
                for i in len(images):
                    print(f"{i}th setting:")
                    print(images[i].path)
            
            first_img_path = images[0].path[0]
            scan_dir = first_img_path.split(os.sep)[:-3]
            
            # Change directory name following migration from Lisa to Snellius
            if self.m2f_preds_dirname == 'ViT_masks':
                scan_dir[1] = 'scratch-shared'
                m2f_dir = ['', 'home', 'fsun', 'data', 'scannet', 'scans', scan_dir[-1]]
                m2f_dir = os.sep.join([*m2f_dir, self.m2f_preds_dirname])
            elif self.m2f_preds_dirname == 'm2f_masks':
                scan_dir[1] = 'scratch-shared'
                m2f_dir = ['', 'scratch-shared', 'fsun', 'data', 'scannet', 'scans', scan_dir[-1]]
                m2f_dir = os.sep.join([*m2f_dir, self.m2f_preds_dirname])                
            
#             print("Changing gt_dir to m2f_masks_refined! ")
            gt_dir = os.path.join('/scratch-shared/fsun/data/scannet/scans', scan_dir[-1], 'label-filt-scannet20')#'label-filt-scannet20')
                                                
            m2f_masks, m2f_mask_paths, gt_masks, gt_mask_paths = [], [], [], []
            print(len(images[0].path))
            for rgb_path in images[0].path:
                # Pred masks
                m2f_filename, ext = osp.splitext(rgb_path.split(os.sep)[-1])
                m2f_filename += '.png'
                pred_mask_path = osp.join(m2f_dir, m2f_filename)
                pred_mask = Image.open(pred_mask_path)
                pred_mask = pred_mask.resize(self.img_ref_size, resample=Image.NEAREST) 
                # minus 1 to match DVA label classes ranging [0, 19] instead of [1, 20]
                m2f_masks.append(pil_to_tensor(pred_mask) - 1)
                m2f_mask_paths.append(pred_mask_path)
                
                # GT masks
                gt_filename, ext = osp.splitext(rgb_path.split(os.sep)[-1])
                gt_filename += '.png'
                gt_mask_path = osp.join(gt_dir, gt_filename)
                gt_mask = Image.open(gt_mask_path)
                gt_mask = gt_mask.resize(self.img_ref_size, resample=Image.NEAREST) 
                # minus 1 to match DVA label classes ranging [-1, 19] instead of [0, 20]
                gt_masks.append(pil_to_tensor(gt_mask).long() - 1)
                gt_mask_paths.append(gt_mask_path)
                
                                
            m2f_masks = torch.stack(m2f_masks)
            gt_masks = torch.stack(gt_masks)
                
            # Store M2F pred mask in data
            images[0].m2f_pred_mask = m2f_masks
            images[0].m2f_pred_mask_path = np.array(m2f_mask_paths)
            
            # Store GT mask in data
            images[0].gt_mask = gt_masks
            images[0].gt_mask_path = np.array(gt_mask_paths)
            
            
            
                      
            # Left-over transform
            try:
                data, images = randomhorizontalflip(data, images)
            except: 
                pass
                
            data = MMData(data, image=images)
            del images
            

            csr_idx = data.modalities['image'][0].view_csr_indexing
            n_seen = csr_idx[1:] - csr_idx[:-1]
            seen_mask = ( n_seen > 0 )
            n_seen_points = seen_mask.sum()
            seen_csr_idx = csr_idx[torch.cat((torch.ones(1, dtype=torch.bool), seen_mask), dim=-1)]
            
            # Take subset of seen points because we only need points with mapping feats
            n_seen = n_seen[seen_mask]
                    
            N_VIEWS = self.n_views
            mapping_feats = data.modalities['image'][0].mappings.values[2]
            view_feats = torch.zeros((n_seen_points, N_VIEWS, mapping_feats.shape[-1]))
            
            
            # at most how many viewpoints each point has
            clipped_n_seen = torch.clip(n_seen, max=N_VIEWS)
            
            pixel_validity = torch.range(1, N_VIEWS).repeat(n_seen_points, 1)
            pixel_validity = ( pixel_validity <= clipped_n_seen.unsqueeze(-1) )

            # select mapping feature vectors for each 3d point
            view_feat_idx = []
            for i in range(len(seen_csr_idx) - 1):
                n = clipped_n_seen[i]
                if n < N_VIEWS:
                    view_feat_idx.extend(list(range(seen_csr_idx[i], seen_csr_idx[i+1])))
                else:
                    view_feat_idx.extend(np.random.choice(range(seen_csr_idx[i], seen_csr_idx[i+1]), size=n.numpy(), replace=False))
                    
            view_feat_idx = torch.LongTensor(view_feat_idx)

            # change format of mapping features to be compatible with MVFusion model: [n_points, n_views, map_feat_dim]
            view_feats[pixel_validity] = mapping_feats[view_feat_idx]
            
            # insert pixel validity feature
            view_feats = torch.concat((pixel_validity.unsqueeze(-1), view_feats), dim=-1)
                        
            # same for m2f feats
            mapped_m2f_feats = data.modalities['image'][0].get_mapped_m2f_features(interpolate=True)
            m2f_feats = torch.randint(low=0, high=self.num_classes, size=(n_seen_points, N_VIEWS, 1))
            m2f_feats[pixel_validity] = mapped_m2f_feats[view_feat_idx].long()

            
            # Limit number of views for ablation studies
            if self.n_views_ablation is not None and self.n_views_ablation < 9:
                view_feats[:, self.n_views_ablation:, :] = 0
                m2f_feats[~view_feats[:, :, 0].bool()] = torch.randint(low=0, high=self.num_classes, size=(n_seen_points, N_VIEWS, 1))[~view_feats[:, :, 0].bool()]
                
            
            ##### hack: change all 2d pred labels for a point into the mode prediction/randomly selected view pred.
            #####       for the view selection experiment
            if self.store_mode_pred:                
                valid_m2f_feats = []
                for i in range(len(m2f_feats)):
                    valid_m2f_feats.append(m2f_feats[i][pixel_validity[i]])

                mode_preds = []
                for m2feats_of_seen_point in valid_m2f_feats:
                    mode_preds.append(torch.mode(m2feats_of_seen_point.squeeze(), dim=0)[0])
                mode_preds = torch.stack(mode_preds, dim=0)
                
                # save directly in 'pred' attribute
                data.data.pred = mode_preds
                
#                 # repeat to MVFusion input shape
#                 m2f_feats = mode_preds.unsqueeze(-1).repeat_interleave(self.n_views, dim=-1).unsqueeze(-1)
                
            elif self.store_random_pred:
                valid_m2f_feats = []
                for i in range(len(m2f_feats)):
                    valid_m2f_feats.append(m2f_feats[i][pixel_validity[i]])

                selected_view_preds = []
                for m2feats_of_seen_point in valid_m2f_feats:
                    selected_idx = torch.randint(low=0, high=m2feats_of_seen_point.shape[0], size=(1,))
                    selected_pred = m2feats_of_seen_point[selected_idx].squeeze(0)
                    selected_view_preds.append(selected_pred)
                selected_view_preds = torch.stack(selected_view_preds, dim=0)
                
                # save directly in 'pred' attribute
                data.data.pred = selected_view_preds.squeeze()
                
#                 # repeat to MVFusion input shape
#                 m2f_feats = selected_view_preds.repeat_interleave(self.n_views, dim=-1).unsqueeze(-1)
    
            # Save mapping + m2f features
            data.data.mvfusion_input = torch.cat((view_feats, m2f_feats), dim=-1)
                
        else:
            data = MMData(data, image=images)

#         # Take subset of only seen points
#         # NOTE: each point is contained multiple times if it has multiple correspondences
#         csr_idx = data.modalities['image'][0].view_csr_indexing
#         dense_idx_list = torch.arange(data.modalities['image'].num_points).repeat_interleave(csr_idx[1:] - csr_idx[:-1])
#         # take subset of only seen points without re-indexing the same point
#         data = data[dense_idx_list.unique()]

        return data

    
    @staticmethod
    def uncollate(data_collated, slices_dict, scan_id_to_name, skip_keys=[]):
        r"""Reverses collate. Transforms a collated Data and associated
        slices into a python dictionary of Data objects. The keys are
        the scan names provided by scan_id_to_name.
        """
        data_dict = {}
        for idx in range(data_collated.id_scan.shape[0]):

            data = data_collated.__class__()
            if hasattr(data_collated, '__num_nodes__'):
                data.num_nodes = data_collated.__num_nodes__[idx]

            for key in data_collated.keys:
                if key in skip_keys:
                    continue

                item, slices = data_collated[key], slices_dict[key]
                start, end = slices[idx].item(), slices[idx + 1].item()
                if torch.is_tensor(item):
                    s = list(repeat(slice(None), item.dim()))
                    s[data_collated.__cat_dim__(key, item)] = slice(start, end)
                elif start + 1 == end:
                    s = slices[start]
                else:
                    s = slice(start, end)
                data[key] = item[s]

            data_dict[scan_id_to_name[int(data.id_scan.item())]] = data

        return data_dict



class ScannetDatasetMM(BaseDatasetMM, ABC):
    """ Wrapper around Scannet that creates train and test datasets.
    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain
            - dataroot
            - version
            - max_num_point (optional)
            - use_instance_labels (optional)
            - use_instance_bboxes (optional)
            - donotcare_class_ids (optional)
            - pre_transforms (optional)
            - train_transforms (optional)
            - val_transforms (optional)
            - frame_depth (optional)
            - frame_rgb (optional)
            - frame_pose (optional)
            - frame_intrinsics (optional)
            - frame_skip (optional)
    """

    SPLITS = SPLITS

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        use_instance_labels: bool = dataset_opt.use_instance_labels
        use_instance_bboxes: bool = dataset_opt.use_instance_bboxes
        donotcare_class_ids: [] = list(dataset_opt.get('donotcare_class_ids', []))
        max_num_point: int = dataset_opt.get('max_num_point', None)
        process_workers: int = dataset_opt.get('process_workers', 0)
        is_test: bool = dataset_opt.get('is_test', False)
        types = [".sens", ".txt", "_vh_clean_2.ply", "_vh_clean_2.0.010000.segs.json", ".aggregation.json"]
        frame_depth: int = dataset_opt.get('frame_depth', False)
        frame_rgb: int = dataset_opt.get('frame_rgb', True)
        frame_pose: int = dataset_opt.get('frame_pose', True)
        frame_intrinsics: int = dataset_opt.get('frame_intrinsics', True)
        frame_skip: int = dataset_opt.get('frame_skip', 50)
        neucon_metas_dir: str = dataset_opt.get('neucon_metas_dir', '')
        neucon_frame_skip: int = dataset_opt.get('neucon_frame_skip', 1)
        m2f_preds_dirname: str = dataset_opt.get('m2f_preds_dirname', '')
        load_m2f_masks: bool = dataset_opt.get('load_m2f_masks', False)
        print("Load predicted 2D semantic segmentation labels from directory ", dataset_opt.get('m2f_preds_dirname', None))
        undo_axis_align: bool = dataset_opt.get('undo_axis_align', False)
        center_xy: bool = dataset_opt.get('center_xy', False)
        center_z: bool = dataset_opt.get('center_z', False)
        n_views: int = dataset_opt.get('n_views', 0)
        n_views_ablation = dataset_opt.get('n_views_ablation', None)
        store_mode_pred: bool = dataset_opt.get('store_mode_pred', False)
        store_random_pred: bool = dataset_opt.get('store_random_pred', False)
            
        print("initialize train dataset")
        self.train_dataset = ScannetMM(
            self._data_path,
            split="train",
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            pre_transform_image=self.pre_transform_image,
            transform_image=self.train_transform_image,
            version=dataset_opt.version,
            use_instance_labels=use_instance_labels,
            use_instance_bboxes=use_instance_bboxes,
            donotcare_class_ids=donotcare_class_ids,
            max_num_point=max_num_point,
            process_workers=process_workers,
            is_test=is_test,
            types=types,
            frame_depth=frame_depth,
            frame_rgb=frame_rgb,
            frame_pose=frame_pose,
            frame_intrinsics=frame_intrinsics,
            frame_skip=frame_skip,
            neucon_metas_dir=neucon_metas_dir,
            neucon_frame_skip=neucon_frame_skip,
            m2f_preds_dirname=m2f_preds_dirname,
            load_m2f_masks=load_m2f_masks,
            undo_axis_align=undo_axis_align,
            center_xy=center_xy,
            center_z=center_z, 
            n_views=n_views,
            n_views_ablation=n_views_ablation,
            store_mode_pred=store_mode_pred,
            store_random_pred=store_random_pred
        )

        print("initialize val dataset")
        self.val_dataset = ScannetMM(
            self._data_path,
            split="val",
            transform=self.val_transform,
            pre_transform=self.pre_transform,
            pre_transform_image=self.pre_transform_image,
            transform_image=self.val_transform_image,
            version=dataset_opt.version,
            use_instance_labels=use_instance_labels,
            use_instance_bboxes=use_instance_bboxes,
            donotcare_class_ids=donotcare_class_ids,
            max_num_point=max_num_point,
            process_workers=process_workers,
            is_test=is_test,
            types=types,
            frame_depth=frame_depth,
            frame_rgb=frame_rgb,
            frame_pose=frame_pose,
            frame_intrinsics=frame_intrinsics,
            frame_skip=frame_skip,
            neucon_metas_dir=neucon_metas_dir,
            neucon_frame_skip=neucon_frame_skip,
            m2f_preds_dirname=m2f_preds_dirname,
            load_m2f_masks=load_m2f_masks,
            undo_axis_align=undo_axis_align,
            center_xy=center_xy,
            center_z=center_z, 
            n_views=n_views,
            n_views_ablation=n_views_ablation,
            store_mode_pred=store_mode_pred,
            store_random_pred=store_random_pred
        )

#         print("initialize test dataset")
#         self.test_dataset = ScannetMM(
#             self._data_path,
#             split="test",
#             transform=self.val_transform,
#             pre_transform=self.pre_transform,
#             pre_transform_image=self.pre_transform_image,
#             transform_image=self.test_transform_image,
#             version=dataset_opt.version,
#             use_instance_labels=use_instance_labels,
#             use_instance_bboxes=use_instance_bboxes,
#             donotcare_class_ids=donotcare_class_ids,
#             max_num_point=max_num_point,
#             process_workers=process_workers,
#             is_test=is_test,
#             types=types,
#             frame_depth=frame_depth,
#             frame_rgb=frame_rgb,
#             frame_pose=frame_pose,
#             frame_intrinsics=frame_intrinsics,
#             frame_skip=frame_skip,
#             neucon_frame_skip=neucon_frame_skip,
#             neucon_metas_dir=neucon_metas_dir,
#             m2f_preds_dirname=m2f_preds_dirname,
#             load_m2f_masks=load_m2f_masks,
#             undo_axis_align=undo_axis_align,
#             center_xy=center_xy,
#             center_z=center_z, 
#             n_views=n_views,
#             n_views_ablation=n_views_ablation,
#             store_mode_pred=store_mode_pred,
#             store_random_pred=store_random_pred
#         )

        if dataset_opt.class_weight_method:
            self.add_weights(
                class_weight_method=dataset_opt.class_weight_method)

    @property
    def path_to_submission(self):
        return self.train_dataset.path_to_submission

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker
        Arguments:
            dataset {[type]}
            wandb_log - Log using weight and biases
        Returns:
            [BaseTracker] -- tracker
        """
        from torch_points3d.metrics.scannet_segmentation_tracker import ScannetSegmentationTracker

        return ScannetSegmentationTracker(
            self, wandb_log=wandb_log, use_tensorboard=tensorboard_log, ignore_label=IGNORE_LABEL
        )


########################################################################################
#                                                                                      #
#                          Script to load a few ScanNet scans                          #
#                                                                                      #
########################################################################################


if __name__ == '__main__':

    from omegaconf import OmegaConf

    # Recover dataset options
    DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    dataset_options = OmegaConf.load(os.path.join(DIR, 'conf/data/segmentation/multimodal/Feng/scannet-neucon-smallres-m2f.yaml'))

    # Choose download root directory
    dataset_options.data.dataroot = os.path.join(DIR, "data")
    reply = input(f"Save dataset to {dataset_options.data.dataroot} ? [y/n] ")
    if reply.lower() == 'n':
        dataset_options.data.dataroot = ""
        while not osp.exists(dataset_options.data.dataroot):
            dataset_options.data.dataroot = input(f"Please provide an existing directory to which the dataset should be dowloaded : ")

    # Download the hard-coded release scans 
    dataset = ScannetDatasetMM(dataset_options.data)
