from abc import ABC

from ..scannet import *
from .utils import read_image_pose_pairs
from torch_points3d.core.multimodal.image import SameSettingImageData
from torch_points3d.datasets.base_dataset_multimodal import BaseDatasetMM
from torch_points3d.core.multimodal.data import MMData
from tqdm.auto import tqdm as tq
from itertools import repeat
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


import os
import os.path as osp


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
            transform_image=None, neucon_metas_dir='', neucon_frame_skip=1, m2f_preds_dirname='', load_m2f_masks=False, **kwargs):
        self.pre_transform_image = pre_transform_image
        self.transform_image = transform_image
        self.img_ref_size = img_ref_size
        self.neucon_metas_dir = neucon_metas_dir
        self.neucon_frame_skip = neucon_frame_skip
        self.m2f_preds_dirname = m2f_preds_dirname
        self.load_m2f_masks = load_m2f_masks
        super(ScannetMM, self).__init__(*args, **kwargs)
    
        print("temporarily hard code N-views in get_view_dependent_features()")


    def process(self):
        if self.is_test:
            return

        # --------------------------------------------------------------
        # Preprocess 3D data
        # Download, pre_transform and pre_filter raw 3D data
        # Output will be saved to <SPLIT>.pt
        # --------------------------------------------------------------
#         print("do not Preprocess 3D data")
#         print("torch_points3d/datasets/segmentation/multimodal/scannet.py line 123")
        print("pre-processing 3D data")
        super(ScannetMM, self).process()

        # ----------------------------------------------------------
        # Recover image data and run image pre-transforms
        # This is where images are loaded and mappings are computed.
        # Output is saved to processed_2d_<SPLIT>/<SCAN_NAME>.pt
        # ----------------------------------------------------------
        for i, (scan_names, split) in enumerate(zip(self.scan_names, self.SPLITS)):

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
                print("current scan name: ", scan_name)

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

        # Get the 3D point sample and apply transforms
        data = self.get(idx)
        data = data if self.transform is None else self.transform(data)
        

        # Recover the scan name
        mapping_idx_to_scan = getattr(
            self, f"MAPPING_IDX_TO_SCAN_{self.split.upper()}_NAMES")
        scan_name = mapping_idx_to_scan[int(data.id_scan.item())]

        # Load the corresponding 2D data and mappings
        i_split = self.SPLITS.index(self.split)
        images = torch.load(osp.join(
            self.processed_2d_paths[i_split], scan_name + '.pt'))
        
        # Run image transforms
        if self.transform_image is not None:
            data, images = self.transform_image(data, images)
        
            
        # Load Mask2Former predicted masks if dirname is given in dataset config       
        if self.m2f_preds_dirname is not None and self.load_m2f_masks is True:
            if len(images) > 1:
                print(f"ImageData contains {len(images)} different camera settings!")
                for i in len(images):
                    print(f"{i}th setting:")
                    print(images[i].path)
            
            first_img_path = images[0].path[0]
            m2f_dir = first_img_path.split(os.sep)[:-3]
            m2f_dir = os.sep.join([*m2f_dir, self.m2f_preds_dirname])
            
            m2f_masks = []
            m2f_mask_paths = []
            for rgb_path in images[0].path:

                m2f_filename, ext = osp.splitext(rgb_path.split(os.sep)[-1])
                m2f_filename += '.png'
                
                pred_mask_path = osp.join(m2f_dir, m2f_filename)
                pred_mask = Image.open(pred_mask_path)
                pred_mask = pred_mask.resize(self.img_ref_size, resample=Image.NEAREST)
                
                m2f_masks.append(pil_to_tensor(pred_mask))   # maybe need to be saved as floats
                
                m2f_mask_paths.append(pred_mask_path)
                                
            m2f_masks = torch.stack(m2f_masks)
                
            # Store M2F pred mask in data
            images[0].m2f_pred_mask = m2f_masks
            images[0].m2f_pred_mask_path = np.array(m2f_mask_paths)
            
            
            data = MMData(data, image=images)

            # Take subset of only seen points
            # NOTE: each point is contained multiple times if it has multiple correspondences
            dense_idx_list = [
                        torch.arange(im.num_points, device=images.device).repeat_interleave(
                            im.view_csr_indexing[1:] - im.view_csr_indexing[:-1])
                        for im in images]
#             # take subset of only seen points without re-indexing the same point
#             data = data[dense_idx_list[0].unique()]


            # Save mapping features and M2F features in x
            data.data.x = torch.cat(self.get_view_dependent_features(data[dense_idx_list[0].unique()]), dim=-1)
            # Keep track of seen points
            csr_idx = data.modalities['image'].view_cat_csr_indexing
            data.data.x_seen_mask = csr_idx[1:] > csr_idx[:-1]
                                           
                                    
                        
            return data
        
        return MMData(data, image=images)

    def get_view_dependent_features(self, mm_data):
        n_views = 9
        
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
        
        return combined_tensor.reshape(mm_data.num_points, n_views, -1), combined_m2f_tensor.reshape(mm_data.num_points, n_views, 1)
    
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
            load_m2f_masks=load_m2f_masks
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
            load_m2f_masks=load_m2f_masks
        )

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
#             m2f_preds_dirname=m2f_preds_dirname
#         )

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
    dataset_options = OmegaConf.load(os.path.join(DIR, 'conf/data/multimodal/scannet.yaml'))

    # Choose download root directory
    dataset_options.data.dataroot = os.path.join(DIR, "data")
    reply = input(f"Save dataset to {dataset_options.data.dataroot} ? [y/n] ")
    if reply.lower() == 'n':
        dataset_options.data.dataroot = ""
        while not osp.exists(dataset_options.data.dataroot):
            dataset_options.data.dataroot = input(f"Please provide an existing directory to which the dataset should be dowloaded : ")

    # Download the hard-coded release scans 
    dataset = ScannetDatasetMM(dataset_options.data)
    print(dataset)
