import os
import os.path as osp
import sys
import torch
import numpy as np
from time import time
from omegaconf import OmegaConf
start = time()
import warnings
warnings.filterwarnings('ignore')

# torch.cuda.set_device(I_GPU)
DIR = os.path.dirname(os.getcwd())
ROOT = os.path.join(DIR, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, DIR)

from torch_points3d.utils.config import hydra_read
from torch_geometric.data import Data
from torch_points3d.core.multimodal.data import MMData, MMBatch
from torch_points3d.visualization.multimodal_data import visualize_mm_data
from torch_points3d.core.multimodal.image import SameSettingImageData, ImageData
from torch_points3d.datasets.segmentation.multimodal.scannet import ScannetDatasetMM
from torch_points3d.datasets.segmentation.scannet import CLASS_COLORS, CLASS_NAMES, CLASS_LABELS
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.metrics.scannet_segmentation_tracker import ScannetSegmentationTracker
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq


from PIL import Image

import matplotlib.pyplot as plt 


CLASS_COLORS[0] = (174.0, 199.0, 232.0)
CLASS_COLORS[-1] = (0, 0, 0)


# Functions for evaluation

def get_seen_points(mm_data):
    ### Select seen points
    csr_idx = mm_data.modalities['image'][0].view_csr_indexing
    dense_idx_list = torch.arange(mm_data.modalities['image'][0].num_points).repeat_interleave(csr_idx[1:] - csr_idx[:-1])
    # take subset of only seen points without re-indexing the same point
    mm_data = mm_data[dense_idx_list.unique()]
    return mm_data

def get_mode_pred(data):
    pixel_validity = data.data.mvfusion_input[:, :, 0].bool()
    mv_preds = data.data.mvfusion_input[:, :, -1].long()
            
    valid_m2f_feats = []
    for i in range(len(mv_preds)):
        valid_m2f_feats.append(mv_preds[i][pixel_validity[i]])

    mode_preds = []
    for m2feats_of_seen_point in valid_m2f_feats:
        mode_preds.append(torch.mode(m2feats_of_seen_point.squeeze(), dim=0)[0])
    mode_preds = torch.stack(mode_preds, dim=0)
        
    return mode_preds

def get_random_view_pred(data):
    pixel_validity = data.data.mvfusion_input[:, :, 0].bool()
    mv_preds = data.data.mvfusion_input[:, :, -1].long()
            
    valid_m2f_feats = []
    for i in range(len(mv_preds)):
        valid_m2f_feats.append(mv_preds[i][pixel_validity[i]])

    selected_view_preds = []
    for m2feats_of_seen_point in valid_m2f_feats:
        selected_idx = torch.randint(low=0, high=m2feats_of_seen_point.shape[0], size=(1,))
        selected_pred = m2feats_of_seen_point[selected_idx].squeeze(0)
        selected_view_preds.append(selected_pred)
    selected_view_preds = torch.stack(selected_view_preds, dim=0)
        
    return selected_view_preds


def get_average_weighted_pred(data):
    pixel_validity = data.data.mvfusion_input[:, :, 0].bool()
    mv_preds = data.data.mvfusion_input[:, :, -1].long()
    
    normalized_depth = data.data.mvfusion_input[:, :, 1]
            
    valid_m2f_feats = []
    depths = []
    for i in range(len(mv_preds)):
        valid_m2f_feats.append(mv_preds[i][pixel_validity[i]])
        depths.append(normalized_depth[i][pixel_validity[i]])

    preds = []
    for m2feats_of_seen_point, depth in zip(valid_m2f_feats, depths):
        
        sum_of_all_dists = depth.sum()
        weights = sum_of_all_dists / depth
        one_hot = torch.nn.functional.one_hot(m2feats_of_seen_point, num_classes=20)
        per_class_score = one_hot * weights.unsqueeze(1)
        aggregated_weights = per_class_score.sum(0)
        
        final_pred = aggregated_weights.argmax(0)
        preds.append(final_pred)
    preds = torch.stack(preds, dim=0)
        
    return preds


def get_normalized_entropy(labels):
    counts = torch.unique(labels, return_counts=True)[1]
    
    pk = counts / counts.sum()
    len_pk = torch.tensor(len(pk))
    if len_pk == 1:
        normalized_entropy = 0.
    else:
        normalized_entropy = -sum(pk * torch.log2(pk)) / torch.log2(len_pk)
    return normalized_entropy
        
def get_semantic_image_from_camera(dataset, scene, mesh_triangles, intrinsic, extrinsic, class_id_faces, im_size=(480, 640)):
    """
    Returns the back-projected semantic label image given camera parameters and (semantic) mesh.
    """
    
    # Initialize rays for given camera
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=intrinsic,
        extrinsic_matrix=extrinsic,
        width_px=im_size[1],
        height_px=im_size[0],
    )

    # Get result
    ans = scene.cast_rays(rays)

    primitive_ids = ans['primitive_ids'].numpy()
    primitive_uvs = ans['primitive_uvs'].numpy()

    # Select the closest vertex for each valid face in the projected mesh
    valid_mask = primitive_ids != scene.INVALID_ID

    # https://stackoverflow.com/questions/45212949/vertex-of-a-3d-triangle-that-is-closest-to-a-point-given-barycentric-parameter-o
    w_coords = (1 - primitive_uvs[:, :, 0][valid_mask] - primitive_uvs[:, :, 1][valid_mask])
    barycentric_coords = np.concatenate((w_coords[:, None], primitive_uvs[valid_mask]), axis=-1)

    selected_vertex_idx = np.argmax(barycentric_coords, axis=-1)

    contained_mesh_triangles = mesh_triangles[primitive_ids[valid_mask]]
    closest_mesh_vertices = contained_mesh_triangles[range(len(barycentric_coords)), selected_vertex_idx]
    
    # Map mesh vertices to semantic label
    labels = class_id_faces[closest_mesh_vertices]
    # Remap to [0 ; num_labels - 1]
    labels = dataset.val_dataset._remap_labels(torch.tensor(labels))

    # Visualize back-projection
    image = torch.ones(im_size, dtype=torch.long) * -1
    image[valid_mask] = labels


    # NN interpolation at invalid pixels          
    nearest_neighbor = scipy.ndimage.morphology.distance_transform_edt(
        image==-1, return_distances=False, return_indices=True)    

    image = image[nearest_neighbor].numpy()
    return image

def read_axis_align_matrix(filename):
    lines = open(filename).readlines()
    axis_align_matrix = None
    for line in lines:
        if "axisAlignment" in line:
            axis_align_matrix = torch.Tensor([float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")]).reshape((4, 4))
            break
    return axis_align_matrix

def save_semantic_prediction_as_txt(tracker, model_name, mask_name):
    orginal_class_ids = np.asarray(tracker._dataset.train_dataset.valid_class_idx)
    path_to_submission = tracker._dataset.path_to_submission
    
    path_to_submission = osp.join(path_to_submission, model_name, mask_name)
    if not osp.exists(path_to_submission):
        os.makedirs(path_to_submission)
    
    for scan_id in tracker._full_preds:
        full_pred = tracker._full_preds[scan_id].cpu().numpy().astype(np.int8)
        full_pred = orginal_class_ids[full_pred]  # remap labels to original labels between 0 and 40
        scan_name = tracker._raw_datas[scan_id].scan_name
        path_file = osp.join(path_to_submission, "{}.txt".format(scan_name))
        
        np.savetxt(path_file, full_pred, delimiter="/n", fmt="%d")
        
    return path_to_submission
        
        
import os
import os.path as osp
import copy
import torch
import hydra
import time
import logging
import scipy.ndimage
import numpy as np
from PIL import Image
import open3d as o3d

# Import building function for model and dataset
from torch_points3d.datasets.dataset_factory import instantiate_dataset
from torch_points3d.models.model_factory import instantiate_model

# Import BaseModel / BaseDataset for type checking
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

# Import from metrics
from torch_points3d.metrics.base_tracker import BaseTracker
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

# Utils import
from torch_points3d.utils.colors import COLORS
from torch_points3d.utils.wandb_utils import Wandb
from torch_points3d.utils.config import getattr_recursive
from torch_points3d.visualization import Visualizer
from torch_points3d.core.data_transform.transforms import PointcloudMerge
from torch_points3d.datasets.segmentation.scannet import CLASS_COLORS, CLASS_NAMES, CLASS_LABELS





log = logging.getLogger(__name__)


class Evaluator():
    
    def __init__(self, cfg, checkpoint_dir, early_break=False):
        self.scans_dir = "/scratch-shared/fsun/data/scannet/scans"

#         self._dataset = dataset
        self._cfg = cfg
    
        if self._cfg.model_name in ['MVFusion_orig', 'DeepSetAttention']:
            self.no_3d = True
        else:
            self.no_3d = False
            
        print("self.no_3d == ", self.no_3d)
            
        self.wandb_log = False
        self.tensorboard_log = False
        
        self.early_break = early_break


#         # Create the model
#         print(f"Creating model: {self._cfg.model_name}")
#         model = instantiate_model(self._cfg, self._dataset)

#         # Load the checkpoint and recover the 'best_miou' model weights
#         checkpoint = torch.load(f'{checkpoint_dir}/{self._cfg.model_name}.pt', map_location='cpu')
#         model.load_state_dict_with_same_shape(checkpoint['models']['best_miou'], strict=False)

#         # Prepare the model for training
#         self._model = model.cuda()
#         self._device = self._model.device

#         print(f"Device is {self._device}")
        
        
        # Enable CUDNN BACKEND
        torch.backends.cudnn.enabled = self.enable_cudnn

        if not self.has_training:
            self._cfg.training = self._cfg
            resume = bool(self._cfg.checkpoint_dir)
        else:
            resume = bool(self._cfg.training.checkpoint_dir)

        # Get device
        if self._cfg.training.cuda > -1 and torch.cuda.is_available():
            device = "cuda"
            torch.cuda.set_device(self._cfg.training.cuda)
        else:
            device = "cpu"
        self._device = torch.device(device)
        log.info("DEVICE : {}".format(self._device))

        
        # Start Wandb if public
        if self.wandb_log:
            Wandb.launch(
                self._cfg, self._cfg.training.wandb.public and self.wandb_log)

        # Checkpoint
        self._checkpoint: ModelCheckpoint = ModelCheckpoint(
            checkpoint_dir, self._cfg.model_name,
            "", run_config=self._cfg, resume=resume)
            
        # Recover the merged config from Checkpoint
        self._cfg = self._checkpoint.run_config

        # Update dataset config with number of views for the MVFusion Transformer
        model_config = getattr(self._cfg.models, self._cfg.model_name, None)
        self._cfg.data.n_views = model_config.backbone.transformer.n_views
        
        print("####################### Temporary Solution for baseline + 2D-3D projection: SETTING MODEL_NAME TO : select_random")
        print("Number of views for baseline input: ", self._cfg.data.n_views)        
        self._cfg.model_name = 'select_random'
        
        if self._cfg.model_name == 'select_random':
            self.aggregation_func = get_random_view_pred
        elif self._cfg.model_name == 'majority_vote':
            self.aggregation_func = get_mode_pred
        elif self._cfg.model_name == 'weighted_averaging':
            self.aggregation_func = get_average_weighted_pred
        
        # Create model and datasets
        if not self._checkpoint.is_empty:
            self._dataset: BaseDataset = instantiate_dataset(self._cfg.data)
            self._model: BaseModel = self._checkpoint.create_model(
                self._dataset, weight_name="best_miou")
        else:
            log.warning("Checkpoint is empty or model cannot be instantiated from given checkpoint.")

#             self._dataset: BaseDataset = instantiate_dataset(self._cfg.data)
#             self._model: BaseModel = instantiate_model(
#                 copy.deepcopy(self._cfg), self._dataset)
#             self._model.instantiate_optimizers(self._cfg, "cuda" in device)
#             self._model.set_pretrained_weights()
#             if not self._checkpoint.validate(self._dataset.used_properties):
#                 log.warning(
#                     "The model will not be able to be used from pretrained "
#                     "weights without the corresponding dataset. Current "
#                     "properties are {}".format(self._dataset.used_properties))
                
        self._checkpoint.dataset_properties = self._dataset.used_properties

        log.info(self._model)

        self._model.log_optimizers()
        log.info("Model size = %i", sum(param.numel() for param in self._model.parameters() if param.requires_grad))

        # Set dataloaders
        self._dataset.create_dataloaders(
            self._model,
            self._cfg.training.batch_size,
            self._cfg.training.shuffle,
            self._cfg.training.num_workers,
            self.precompute_multi_scale,
            train_only=False,
            val_only=True,
            test_batch_size=1
        )
        
        log.info(self._dataset)

        # Verify attributes in dataset
        self._model.verify_data(self._dataset.train_dataset[0])

        # Choose selection stage
        selection_stage = getattr(self._cfg, "selection_stage", "")
        self._checkpoint.selection_stage = self._dataset.resolve_saving_stage(
            selection_stage)
        
        self.mapping_idx_to_scan_names = getattr(self._dataset.val_dataset, "MAPPING_IDX_TO_SCAN_{}_NAMES".format(self._dataset.val_dataset.split.upper()))

        
    def eval_all_metrics(self, stage_name=""):
        self._is_training = False
        
        self._tracker_refined: BaseTracker = self._dataset.get_tracker(
            self.wandb_log, self.tensorboard_log)
        self._tracker_refined_seen_points: BaseTracker = self._dataset.get_tracker(
            self.wandb_log, self.tensorboard_log)            
                        
        if self._dataset.has_val_loader:
            if not stage_name or stage_name == "val":
                print("Evaluating on validation set")
                self._test_refined(epoch=1, stage_name="val")   
                
        # Upscale predictions containing all points to 0.01 voxel size and save point cloud predictions   
        self._tracker_refined.finalise(full_res=True)
        path_to_submission = save_semantic_prediction_as_txt(
            self._tracker_refined, self._cfg.model_name, self._dataset.val_dataset.m2f_preds_dirname)
        
        print("Upscaled predictions in 3D of all points: ", self._tracker_refined.get_metrics())
        print("Exiting now, we do not evaluate other metrics")
        
        
#         # Back-project semantic mesh (from pcd) to 2D images given the maximum number of views per scene, and save.
#         # Skips this step if refined images already exist for given model and mask
#         self.mesh_to_image(self._cfg, self._dataset, path_to_submission, self.scans_dir, save_output='if_not_exists') 
        
#         # Evaluate 2D semantic segmentation
#         self._tracker_refined_2d_iou: BaseTracker = self._dataset.get_tracker(
#             self.wandb_log, self.tensorboard_log)
        
#         self._evaluate_2d_iou()
        
#         # Evaluate 2D cross-view consistency
#         # Temporary solution to evaluate both 2D input masks and 2D refined masks for Cross-view Consistency:
#         # load refined masks into gt_mask attribute
#         self._dataset.val_dataset.gt_dir_name = osp.join(f"{self._dataset.val_dataset.m2f_preds_dirname}_refined", 
#                                                          self._cfg.model_name)
#         self._evaluate_2d_CC()        
        
#         # Evaluate 2D temporal consistency
#         self.tracker_TC_baseline: BaseTracker = self._dataset.get_tracker(
#             self.wandb_log, self.tensorboard_log)
#         self.tracker_TC_refined: BaseTracker = self._dataset.get_tracker(
#             self.wandb_log, self.tensorboard_log)
#         self._evaluate_2d_TC()   
        
    def _evaluate_2d_TC(self):
        from mmflow.apis import init_model, inference_model
        from mmflow.datasets import visualize_flow, write_flow
        import mmcv
        from mmflow.ops import Warp

        # Specify the path to model config and checkpoint file
        config_file = '/home/fsun/DeepViewAgg/flow/configs/flownet2_8x1_sfine_flyingthings3d_subset_384x768.py'
        checkpoint_file = '/home/fsun/DeepViewAgg/flow/pretrained/flownet2_8x1_sfine_flyingthings3d_subset_384x768.pth'

        # build the model from a config file and a checkpoint file
        flow_model = init_model(config_file, checkpoint_file, device='cuda:0')

        warp = Warp(mode='nearest',
                    padding_mode='zeros',
                    align_corners=False,
                    use_mask=True).cuda()
        
    
        mask_foldername = self._dataset.val_dataset.m2f_preds_dirname
        refined_mask_foldername = osp.join(f"{mask_foldername}_refined", self._cfg.model_name)
        mask_scans_dir = '/scratch-shared/fsun/data/scannet/scans'
        

        self.tracker_TC_baseline.reset(stage='val')
        self.tracker_TC_refined.reset(stage='val')

        scan_ids = list(self.mapping_idx_to_scan_names.values())

        scans_dir = "/scratch-shared/fsun/data/scannet/scans"

        for scan_id in Ctq(scan_ids):
            scan_dir = osp.join(scans_dir, scan_id)

            mask_scan_dir = osp.join(mask_scans_dir, scan_id)

            mask_dir = osp.join(mask_scan_dir, mask_foldername)
            refined_mask_dir = osp.join(scan_dir, refined_mask_foldername)

            refined_mask_names = os.listdir(refined_mask_dir)
            refined_mask_names = sorted(refined_mask_names, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

            color_im_dir = osp.join(scan_dir, 'color_resized')

            # Loop over all pairs of consecutive images
            for i in range(len(refined_mask_names) - 1):
                cur_frame_name, next_frame_name = refined_mask_names[i], refined_mask_names[i+1]
                cur_frame_id, next_frame_id = int(refined_mask_names[i].split(".")[0]), int(refined_mask_names[i+1].split(".")[0])

                # Skip if jump in frames is too big
                if next_frame_id - cur_frame_id > 50:
                    continue

                cur_color_p, next_color_p = osp.join(color_im_dir, cur_frame_name), osp.join(color_im_dir, next_frame_name)
                cur_color_im, next_color_im = Image.open(cur_color_p), Image.open(next_color_p)

                result = inference_model(flow_model, next_color_p, cur_color_p)
                flow_map = torch.tensor(result).permute(2, 0, 1).unsqueeze(0)

                ################# 2D input masks
                seg_im_p1 = osp.join(mask_dir, cur_frame_name)
                seg_im_p2 = osp.join(mask_dir, next_frame_name)
                seg_im1 = np.asarray(Image.open(seg_im_p1)) 
                seg_im2 = np.asarray(Image.open(seg_im_p2)).astype(np.int) - 1   # Adjust labels

                # warping im1 to im2
                seg_im1_semantic = torch.tensor(seg_im1).unsqueeze(0).unsqueeze(0).float()
                seg_im_warped = warp(seg_im1_semantic, flow_map).permute(0, 2, 3, 1)[0].squeeze() - 1    # Adjust labels

                # take im2 as 'pseudo gt' for temporal consistency. Thus, invalidly warped pixels should be ignored
                # by setting the corresponding gt label to the IGNORE_LABEL
                seg_im2[seg_im_warped == -1] = IGNORE_LABEL
                self.tracker_TC_baseline.track(pred_labels=seg_im_warped.long(), gt_labels=seg_im2, model=None)

                ################ 2D refined masks
                seg_im_p1 = osp.join(refined_mask_dir, cur_frame_name)
                seg_im_p2 = osp.join(refined_mask_dir, next_frame_name)
                seg_im1 = np.asarray(Image.open(seg_im_p1))  + 1 # Adjust labels for flow warping (invalid pixels will be set to 0)

                # Resize in case
                seg_im2 = Image.open(seg_im_p2).resize((640, 480), resample=0)
                seg_im2 = np.asarray(seg_im2).astype(np.int) 

                # warping im1 to im2
                seg_im1_semantic = torch.tensor(seg_im1).unsqueeze(0).unsqueeze(0).float()
                seg_im_warped = warp(seg_im1_semantic, flow_map).permute(0, 2, 3, 1)[0].squeeze() - 1 # Adjust labels back to normal

                # take im2 as 'pseudo gt' for temporal consistency. Thus, invalidly warped pixels should be ignored
                # by setting the corresponding gt label to the IGNORE_LABEL
                seg_im2[seg_im_warped == -1] = IGNORE_LABEL
                self.tracker_TC_refined.track(pred_labels=seg_im_warped.long(), gt_labels=seg_im2, model=None)

            if self.early_break:
                break
            
        print("Temporal Consistency Scores measured over all validation scenes, using all image views")
        print("Input 2D", self.tracker_TC_baseline.get_metrics())
        print(self.tracker_TC_baseline._miou_per_class)   
        print("Refined 2D ", self.tracker_TC_refined.get_metrics())
        print(self.tracker_TC_refined._miou_per_class)   

    def _evaluate_2d_CC(self):
        
        temp_tracker: BaseTracker = self._dataset.get_tracker(
            self.wandb_log, self.tensorboard_log)
                
        # Load instance labels for cross-view consistency evaluation
        self._dataset.val_dataset.load_instance_labels = True
        
#         # Load refined masks in attribute of gt mask for CC evaluation
#         self._dataset.val_dataset.gt_dir_name = osp.join(self._cfg.data.m2f_preds_dirname + '_refined', self._cfg.model_name) 

        
        del self._dataset._val_loader
        
        # Re-create dataloader
        self._dataset.create_dataloaders(
            self._model,
            self._cfg.training.batch_size,
            self._cfg.training.shuffle,
            self._cfg.training.num_workers,
            self.precompute_multi_scale,
            train_only=False,
            val_only=True,
            test_batch_size=1
        )
                
        self.instance_count = torch.zeros(2, 20)
        self.cum_n_entropy = torch.zeros(2, 20)

        self._model.eval()
            
        with Ctq(self._dataset.val_dataloader) as tq_loader:
            for batch in tq_loader:
                
                with torch.no_grad():
#                     self._model.set_input(batch, self._device)
                    
#                     with torch.cuda.amp.autocast(enabled=self._model.is_mixed_precision()):
#                         self._model.forward(epoch=1)

#                     batch.data.pred = self._model.output.detach().cpu().argmax(1)

                    batch = get_seen_points(batch)
    
                    


                    temp_tracker.track(model=None, pred_labels=self.aggregation_func(batch), gt_labels=batch.data.y)

                    # Accumulate entropy of seen points
                    self.add_entropy_to_accumulator(batch)
                    
                if self.early_break:
                    break
            
        print("3D seen points metrics")
        print(temp_tracker.get_metrics())
        
        self.cum_n_entropy = self.cum_n_entropy / (self.instance_count + 1e-8)
        self.crossview_consistency = (1 - self.cum_n_entropy) * 100       
        print("2D cross-view consistency (CC) for input masks")
        print(self.crossview_consistency[0])
        print("Mean: ", self.crossview_consistency[0].mean())

        print("2D cross-view consistency (CC) for refined masks")
        print(self.crossview_consistency[1])
        print("Mean: ", self.crossview_consistency[1].mean())

        
                
    def add_entropy_to_accumulator(self, mm_data):
        for instance_id in mm_data.data['instance_labels'].unique():

            if instance_id == 0:
                continue

            instance_mask = mm_data.data['instance_labels'] == instance_id
            instance = mm_data[instance_mask]
            instance_class = instance.y[0]

            # Skip invalid semantic class
            if instance_class == IGNORE_LABEL:
                continue

            # Track all per-point predictions of active views for current instance. Some points have more 
            # predictions than others.
            input_preds = instance.modalities['image'][0].get_mapped_m2f_features().squeeze()
            output_preds = instance.modalities['image'][0].get_mapped_gt_labels().squeeze() + 1   # +1 label offset

            input_n_entropy = get_normalized_entropy(input_preds)
            output_n_entropy = get_normalized_entropy(output_preds)

            self.cum_n_entropy[0, instance_class] += input_n_entropy
            self.cum_n_entropy[1, instance_class] += output_n_entropy
            self.instance_count[:, instance_class] += 1

    def mesh_to_image(self, cfg, dataset, path_to_submission, scans_dir, save_output='if_not_exists'):
        # User input
        output_image_size = (480, 640)
        preprocessed_2d_data_dir = "/scratch-shared/fsun/dvata/scannet-neucon-smallres-m2f/processed/processed_2d_val"

        ########################################################################################################################
        input_mask_name = cfg.data.m2f_preds_dirname
        scan_names = list(dataset.val_dataset.MAPPING_IDX_TO_SCAN_VAL_NAMES.values())

#         path_to_submission = osp.join(path_to_submission, cfg.model_name, input_mask_name)

        for scan_name in Ctq(scan_names):
            # Output folder location
            refined_mask_dir = osp.join(scans_dir, scan_name, f"{input_mask_name}_refined", f"{self._cfg.model_name}")

            if not osp.exists(refined_mask_dir):
                os.makedirs(refined_mask_dir)

#             # Skip this step if output folder already contains masks
#             if len(os.listdir(refined_mask_dir)) < 10:

            # Load data
            mesh = o3d.io.read_triangle_mesh(f"{scans_dir}/{scan_name}/{scan_name}_vh_clean_2.ply")
            mesh_triangles = np.asarray(mesh.triangles)
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

            # Load predicted class label per vertex
            class_id_faces = np.loadtxt(f"{path_to_submission}/{scan_name}.txt").astype(int)

            # Camera parameters
            intrinsic = np.loadtxt(f"{scans_dir}/{scan_name}/sens/intrinsic/intrinsic_depth.txt")[:3, :3]
            images = torch.load(f"{preprocessed_2d_data_dir}/{scan_name}.pt")

            # Undo axis alignment for extrinsics  
            axis_align_matrix_path = osp.join(scans_dir, scan_name, scan_name + '.txt')
            axis_align_matrix = read_axis_align_matrix(axis_align_matrix_path)
            inv = torch.linalg.inv(axis_align_matrix.T)
            images.extrinsic = inv.T  @ images.extrinsic        

            # Make world-to-camera
            extrinsics = torch.linalg.inv(images.extrinsic).numpy()
            image_names = [osp.splitext(osp.basename(x))[0] for x in images.path]

            # Raycasting
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(mesh)

            for i in range(len(image_names)):
                image = get_semantic_image_from_camera(dataset=dataset, scene=scene, mesh_triangles=mesh_triangles, intrinsic=intrinsic,
                                                       extrinsic=extrinsics[i], 
                                                       class_id_faces=class_id_faces, im_size=output_image_size)


                # Save refined prediction (backprojected from mesh + interpolated missing pixels)
                image = Image.fromarray(image.astype(np.uint8), 'L')
                im_save_path = osp.join(refined_mask_dir, image_names[i] + '.png')
                image.save(im_save_path)   

#             else:
#                 print("Output directory already exists and contains >10 masks!")

            if self.early_break:
                break
        
    def _evaluate_2d_iou(self):
        input_mask_name = self._cfg.data.m2f_preds_dirname

        scan_names = list(self._dataset.val_dataset.MAPPING_IDX_TO_SCAN_VAL_NAMES.values())
        
        self._tracker_refined_2d_iou.reset(stage="val")

        with Ctq(self._dataset.val_dataloader) as tq_loader:
            for batch in tq_loader:
                scan_name = self.mapping_idx_to_scan_names[batch.id_scan.item()]

                gt_dir = osp.join(self.scans_dir, scan_name, 'label-filt-scannet20')
                mask_dir = osp.join(self.scans_dir, scan_name, f"{input_mask_name}_refined", f"{self._cfg.model_name}")

                im_names = [osp.basename(x) for x in batch.modalities['image'][0].m2f_pred_mask_path]

                for im in im_names:
                    gt = Image.open(osp.join(gt_dir, im))
                    mask = Image.open(osp.join(mask_dir, im))
                    gt = gt.resize((640, 480), 0)

                    gt = np.asarray(gt).astype(float) - 1
                    mask = np.asarray(mask)

                    self._tracker_refined_2d_iou.track(pred_labels=mask, gt_labels=gt, model=None)
                    
                if self.early_break:
                    break

        print("2D evaluation of refined masks")
        print(self._tracker_refined_2d_iou.get_metrics())
        print(self._tracker_refined_2d_iou._miou_per_class)       
            
    def _test_refined(self, epoch, stage_name: str):

        loaders = [self._dataset.val_dataloader]

        self._model.eval()
            
        for loader in loaders:
            print("Input mask name: ", loader.dataset.m2f_preds_dirname)
            
            stage_name = loader.dataset.name
            self._tracker_refined.reset(stage_name)
            self._tracker_refined_seen_points.reset(stage_name)
            
            with Ctq(loader) as tq_loader:
                for data in tq_loader:
                    with torch.no_grad():
                        if self.no_3d:
                            data = get_seen_points(data)
                            self._model.set_input(data, self._device)
                        else:
                            self._model.set_input(data, self._device)
                            
#                         with torch.cuda.amp.autocast(enabled=self._model.is_mixed_precision()):
#                             self._model.forward(epoch=epoch)


                        self._model.output = torch.nn.functional.one_hot(self.aggregation_func(data), num_classes=20)


                        data.data.pred = self._model.output.detach().cpu().argmax(1)

                        # 3D mIoU, all points
                        self._tracker_refined.track(self._model, full_res=True, data=data)

                        if not self.no_3d:
                            # 3D mIoU, seen points
                            data = get_seen_points(data)
                        self._tracker_refined_seen_points.track(pred_labels=data.data.pred, gt_labels=data.data.y, model=None)
        

                    tq_loader.set_postfix(**self._tracker_refined.get_metrics())
                    
                    if self.early_break:
                        break

            print("Evaluated scores for 3D semantic segmentation on all points: ")
            
            print("--- Refined 3D ---")
            print(self._tracker_refined.get_metrics())
            print(self._tracker_refined._miou_per_class)
            
            print("Evaluated scores for 3D semantic segmentation on seen points: ")
    
            print("--- Refined 3D ---")
            print(self._tracker_refined_seen_points.get_metrics())
            print(self._tracker_refined_seen_points._miou_per_class)
            
                            
        
    def eval_baseline(self, stage_name=""):
        self._is_training = False
        
        self._tracker_baseline: BaseTracker = self._dataset.get_tracker(
            self.wandb_log, self.tensorboard_log)
            
        self._tracker_baseline_2d: BaseTracker = self._dataset.get_tracker(
            self.wandb_log, self.tensorboard_log)
                        
        epoch = 1

        if self._dataset.has_val_loader:
            if not stage_name or stage_name == "val":
                self._test_baseline(epoch, "val")   

    @property
    def has_training(self):
        return getattr(self._cfg, "training", None)
    
    @property
    def enable_cudnn(self):
        if self.has_training:
            return getattr(self._cfg.training, "enable_cudnn", True)
        else:
            return getattr(self._cfg, "enable_cudnn", True)

    @property
    def precompute_multi_scale(self):
        if not self.has_training:
            return self._model.conv_type == "PARTIAL_DENSE" and getattr(self._cfg.training, "precompute_multi_scale", False)
        else:
            return self._model.conv_type == "PARTIAL_DENSE" and getattr(self._cfg, "precompute_multi_scale", False)
        
        
        
if __name__ == "__main__":
    import argparse
    
    def parse_args():
        parser = argparse.ArgumentParser(description='Model name, config file and checkpoint path')
        parser.add_argument('--input-mask', 
                            choices=['m2f_masks', 'ViT_masks'],
                            help='input mask')
        parser.add_argument('--model-name', help='model name')
        
        args = parser.parse_args()
        return args
    
    args = parse_args()
    
    MASK_IDX = 0 if args.input_mask == 'm2f_masks' else 1
    model_name = args.model_name
        
    dataset_config = 'segmentation/multimodal/Feng/scannet-neucon-smallres-m2f-allviews.yaml'   

    
    if model_name == 'MVFusion_3D_small_6views':
        checkpoint_dir = ["/home/fsun/DeepViewAgg/outputs/MVFusion_3D_6_views_m2f_masks",
                          "/home/fsun/DeepViewAgg/outputs/ViT_masks_3rd_run"]
        models_config = 'segmentation/multimodal/Feng/mvfusion' 
    elif model_name == 'Deepset_3D':
        checkpoint_dir = ["/home/fsun/DeepViewAgg/outputs/2023-02-05/23-15-04",
                          "/home/fsun/DeepViewAgg/outputs/2023-01-23/12-57-16"]        
        models_config = 'segmentation/multimodal/Feng/view_selection_experiment' 
    elif model_name == 'MVFusion_orig':
        checkpoint_dir = ['/home/fsun/DeepViewAgg_31-10-22/DeepViewAgg/outputs/2023-02-11/22-17-12',
                          '/home/fsun/DeepViewAgg/outputs/MVFusion_orig']
        models_config = 'segmentation/multimodal/Feng/mvfusion_orig' 
    elif model_name == 'DeepSetAttention':
        checkpoint_dir = ['/home/fsun/DeepViewAgg/outputs/2023-02-11/10-54-19',
                          '/home/fsun/DeepViewAgg/outputs/2023-02-11/10-52-09']   # old: 'DeepSet_feats_labels_superconvergence'
        models_config = 'segmentation/multimodal/Feng/view_selection_experiment' 


    checkpoint_dir = checkpoint_dir[MASK_IDX]
    
            
            
    overrides = [
        'task=segmentation',
        f'data={dataset_config}',
        f'models={models_config}',
        f'model_name={model_name}',
    ]

    cfg = hydra_read(overrides)
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    cfg.data.load_m2f_masks = True   # load input masks
    cfg.data.m2f_preds_dirname = args.input_mask
    cfg.data.n_views = cfg.models[model_name].backbone.transformer.n_views
    print("Maximum number of views per point: ", cfg.data.n_views)


    evaluator = Evaluator(cfg, checkpoint_dir=checkpoint_dir, early_break=False)
    evaluator.eval_all_metrics(stage_name='val')

    