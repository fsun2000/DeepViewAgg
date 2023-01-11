import os
import copy
import torch
import hydra
import time
import logging

# Feng: extra imports for 2D evaluation
from torch_points3d.utils.multimodal import lexargsort
from torch_points3d.core.multimodal.csr import CSRData
import scipy.ndimage
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from tqdm.auto import tqdm
import wandb
from omegaconf import OmegaConf

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

def save_confusion_matrix(cm, path2save, ordered_names):
    sns.set(font_scale=5)

    template_path = os.path.join(path2save, "{}.png")
    # PRECISION
    cmn = cm.astype("float") / cm.sum(axis=-1)[:, np.newaxis]
    cmn[np.isnan(cmn) | np.isinf(cmn)] = 0
    fig, ax = plt.subplots(figsize=(31, 31))
    sns.heatmap(
        cmn, annot=True, fmt=".2f", xticklabels=ordered_names, yticklabels=ordered_names, annot_kws={"size": 20}
    )
    # g.set_xticklabels(g.get_xticklabels(), rotation = 35, fontsize = 20)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    path_precision = template_path.format("precision")
    plt.tight_layout()
    plt.savefig(path_precision, format="png")

    # RECALL
    cmn = cm.astype("float") / cm.sum(axis=0)[np.newaxis, :]
    cmn[np.isnan(cmn) | np.isinf(cmn)] = 0
    fig, ax = plt.subplots(figsize=(31, 31))
    sns.heatmap(
        cmn, annot=True, fmt=".2f", xticklabels=ordered_names, yticklabels=ordered_names, annot_kws={"size": 20}
    )
    # g.set_xticklabels(g.get_xticklabels(), rotation = 35, fontsize = 20)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    path_recall = template_path.format("recall")
    plt.tight_layout()
    plt.savefig(path_recall, format="png")



class Trainer:
    """
    TorchPoints3d Trainer handles the logic between
        - BaseModel,
        - Dataset and its Tracker
        - A custom ModelCheckpoint
        - A custom Visualizer
    It supports MC dropout - multiple voting_runs for val / test datasets
    """

    def __init__(self, cfg, eval_m2f_preds=False):
        self._cfg = cfg
        self._initialize_trainer()
        
        self.eval_m2f_preds = eval_m2f_preds

    def _initialize_trainer(self):
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

        # Profiling
        if self.profiling:
            # Set the num_workers as torch.utils.bottleneck doesn't work
            # well with it
            self._cfg.training.num_workers = 0

        # Start Wandb if public
        if self.wandb_log:
            Wandb.launch(
                self._cfg, self._cfg.training.wandb.public and self.wandb_log)

        # Checkpoint
        self._checkpoint: ModelCheckpoint = ModelCheckpoint(
            self._cfg.training.checkpoint_dir, self._cfg.model_name,
            "", run_config=self._cfg, resume=resume)
            
        # Recover the merged config from Checkpoint
        self._cfg = self._checkpoint.run_config

        # Update dataset config with number of views for the MVFusion Transformer
        model_config = getattr(self._cfg.models, self._cfg.model_name, None)
        self._cfg.data.n_views = model_config.backbone.transformer.n_views
                
        # Create model and datasets
        if not self._checkpoint.is_empty:
            self._dataset: BaseDataset = instantiate_dataset(self._cfg.data)
            self._model: BaseModel = self._checkpoint.create_model(
                self._dataset, weight_name=self._cfg.training.weight_name)
        else:
            self._dataset: BaseDataset = instantiate_dataset(self._cfg.data)
            self._model: BaseModel = instantiate_model(
                copy.deepcopy(self._cfg), self._dataset)
            self._model.instantiate_optimizers(self._cfg, "cuda" in device)
            self._model.set_pretrained_weights()
            if not self._checkpoint.validate(self._dataset.used_properties):
                log.warning(
                    "The model will not be able to be used from pretrained "
                    "weights without the corresponding dataset. Current "
                    "properties are {}".format(self._dataset.used_properties))
                
        self._checkpoint.dataset_properties = self._dataset.used_properties

        log.info(self._model)

        self._model.log_optimizers()
        log.info("Model size = %i", sum(param.numel() for param in self._model.parameters() if param.requires_grad))

        # Set both dataloaders for first epoch. For some reason the validation loaders do not consume memory untill called.
        self._dataset.create_dataloaders(
            self._model,
            self._cfg.training.batch_size,
            self._cfg.training.shuffle,
            self._cfg.training.num_workers,
            self.precompute_multi_scale,
            train_only=False,
            val_only=False,
            test_batch_size=self._cfg.training.test_batch_size
        )
        
        log.info(self._dataset)

        # Verify attributes in dataset
        self._model.verify_data(self._dataset.train_dataset[0])

        # Choose selection stage
        selection_stage = getattr(self._cfg, "selection_stage", "")
        self._checkpoint.selection_stage = self._dataset.resolve_saving_stage(
            selection_stage)
        self._tracker: BaseTracker = self._dataset.get_tracker(
            self.wandb_log, self.tensorboard_log)
        # 2D evaluation
        self._tracker_2d_mvfusion_pred_masks = None
        self._tracker_2d_model_pred_masks = None

        if self.wandb_log:
            Wandb.launch(self._cfg, not self._cfg.training.wandb.public and self.wandb_log)

        # Run training / evaluation
        self._model = self._model.to(self._device)
        if self.has_visualization:
            self._visualizer = Visualizer(
                self._cfg.visualization, self._dataset.num_batches,
                self._dataset.batch_size, os.getcwd())

    def train(self):
        self._is_training = True
        
        for epoch in range(self._checkpoint.start_epoch, self._cfg.training.epochs):
            log.info("EPOCH %i / %i", epoch, self._cfg.training.epochs)

            if not self.eval_m2f_preds:
                if self.lr_range_test:
                    print("Executing lr range test", flush=True)
                    self._do_lr_test()
                else:
                    self._train_epoch(epoch)
                
                # clear GPU cache as it might fragment a lot
                torch.cuda.empty_cache()
            else:
                self.evaluate_m2f_train_set(epoch)

            if self.profiling:
                return 0

            if epoch % self.eval_frequency != 0:
                continue

            # Create validation loader and delete train loader from system memory
            self._dataset.create_dataloaders(
                self._model,
                self._cfg.training.batch_size,
                self._cfg.training.shuffle,
                self._cfg.training.num_workers,
                self.precompute_multi_scale,
                train_only=False,
                val_only=True,
                test_batch_size=self._cfg.training.test_batch_size
            )
        
            if self._dataset.has_val_loader:
                if not self.eval_m2f_preds:
                    self._test_epoch(epoch, "val")
                else:
                    self.evaluate_m2f_val_set(epoch)

#             if self._dataset.has_test_loaders:
#                 self._test_epoch(epoch, "test")

            # Create train loader and delete validation loader from system memory
            self._dataset.create_dataloaders(
                self._model,
                self._cfg.training.batch_size,
                self._cfg.training.shuffle,
                self._cfg.training.num_workers,
                self.precompute_multi_scale,
                train_only=True,
                val_only=False
            )


        # Single test evaluation in resume case
        if self._checkpoint.start_epoch > self._cfg.training.epochs:
            if self._dataset.has_test_loaders:
                self._test_epoch(epoch, "test")

    def eval(self, stage_name=""):
        self._is_training = False
        
        print("trainer.py: Tracking 2D mask and 2D refined mask scores!")
        self._tracker_2d_model_pred_masks: BaseTracker = self._dataset.get_tracker(
            self.wandb_log, self.tensorboard_log)
        self._tracker_2d_mvfusion_pred_masks: BaseTracker = self._dataset.get_tracker(
            self.wandb_log, self.tensorboard_log)
            
#         print("trainer.py: Tracking view entropy scores for Input Masks, Refined Prediction and GT Masks!")
#         self._tracker_m2f_entropy: BaseTracker = self._dataset.get_tracker(
#             self.wandb_log, self.tensorboard_log)
#         self._tracker_mvfusion_entropy: BaseTracker = self._dataset.get_tracker(
#             self.wandb_log, self.tensorboard_log)
#         self._tracker_gt_entropy: BaseTracker = self._dataset.get_tracker(
#             self.wandb_log, self.tensorboard_log)
            
            
            
        epoch = self._checkpoint.start_epoch
        
#         print()
#         print(" Evaluation of validation set is disabled in trainer.py at line 217 ")
#         print()
        if self._dataset.has_val_loader:
            if not stage_name or stage_name == "val":
                self._test_epoch(epoch, "val")

        # Free memory
        del self._dataset._val_loader

        if self._dataset.has_test_loaders:
            if not stage_name or stage_name == "test":
                self._test_epoch(epoch, "test")

    def _finalize_epoch(self, epoch):
        self._tracker.finalise(**self.tracker_options)
        if self._is_training:
            metrics = self._tracker.publish(epoch)
            self._checkpoint.save_best_models_under_current_metrics(
                self._model, metrics, self._tracker.metric_func)
            if self.wandb_log and self._cfg.training.wandb.public:
                Wandb.add_file(self._checkpoint.checkpoint_path)
            if self._tracker._stage == "train":
                log.info("Learning rate = %f" % self._model.learning_rate)

    def _train_epoch(self, epoch: int):

        self._model.train()
        self._tracker.reset("train")
        self._visualizer.reset(epoch, "train")
        train_loader = self._dataset.train_dataloader

        iter_data_time = time.time()
        with Ctq(train_loader) as tq_train_loader:
            for i, data in enumerate(tq_train_loader):
                
                t_data = time.time() - iter_data_time

                
                if self._dataset.dataset_opt.train_with_mix3d and self._dataset.batch_size > 1:
                    mix3d_time = time.time()
                    data = PointcloudMerge(data, n_merge=2)
                    mix3d_time = time.time() - mix3d_time
                
                iter_start_time = time.time()
                self._model.set_input(data, self._device)
                self._model.optimize_parameters(epoch, self._dataset.batch_size)
                if i % 10 == 0:
                    with torch.no_grad():
                        self._tracker.track(
                            self._model, data=data, **self.tracker_options)

                tq_train_loader.set_postfix(
                    **self._tracker.get_metrics(),
                    data_loading=float(t_data),
                    mix_3d=float(mix3d_time),
                    iteration=float(time.time() - iter_start_time),
                    learning_rate=self._model.learning_rate,
                    color=COLORS.TRAIN_COLOR
                )

                if self._visualizer.is_active:
                    self._visualizer.save_visuals(
                        self._model.get_current_visuals())

                iter_data_time = time.time()

                if self.early_break:
                    break

                if self.profiling:
                    if i > self.num_batches:
                        return 0

        self._finalize_epoch(epoch)
        self._tracker.print_summary()

        
    def evaluate_m2f_train_set(self, epoch):
        self._model.eval()
        if self.enable_dropout:
            self._model.enable_dropout_in_eval()
        self._tracker.reset("train")
        if self.has_visualization:
            self._visualizer.reset(epoch, "train")
        train_loader = self._dataset.train_dataloader

        iter_data_time = time.time()
        with Ctq(train_loader) as tq_train_loader:
            for i, data in enumerate(tq_train_loader):
                
                t_data = time.time() - iter_data_time
                iter_start_time = time.time()
                
                with torch.no_grad():
#                     self._model.set_input(data, self._device)
#                     with torch.cuda.amp.autocast(enabled=self._model.is_mixed_precision()):
#                         self._model.forward(epoch=epoch)

                    
                    #### Mode preds
                    pixel_validity = data.data.mvfusion_input[:, :, 0].bool()
                    mv_preds = data.data.mvfusion_input[:, :, -1].long()

                    valid_m2f_feats = []
                    for i in range(len(mv_preds)):
                        valid_m2f_feats.append(mv_preds[i][pixel_validity[i]])

                    mode_preds = []
                    for m2feats_of_seen_point in valid_m2f_feats:
                        mode_preds.append(torch.mode(m2feats_of_seen_point.squeeze(), dim=0)[0])
                    mode_preds = torch.stack(mode_preds, dim=0)

                    out_scores = torch.nn.functional.one_hot(mode_preds.squeeze().long(), self._dataset.num_classes).float()

                    # Save M2F mode one-hot preds
                    self._model.output = out_scores
                    self._model.labels = data.y

                    if i % 10 == 0:
                        self._tracker.track(
                            self._model, data=data, **self.tracker_options)
                tq_train_loader.set_postfix(
                    **self._tracker.get_metrics(),
                    data_loading=float(t_data),
                    iteration=float(time.time() - iter_start_time),
                    color=COLORS.TRAIN_COLOR
                )                        

                if self._visualizer.is_active:
                    self._visualizer.save_visuals(
                        self._model.get_current_visuals())

                iter_data_time = time.time()

                if self.early_break:
                    break

                if self.profiling:
                    if i > self.num_batches:
                        return 0

        self._finalize_epoch(epoch)
        self._tracker.print_summary()
        
    def evaluate_m2f_val_set(self, epoch):
        loader = self._dataset.val_dataloader
        stage_name = loader.dataset.name
        self._tracker.reset(stage_name)

        with Ctq(loader) as tq_loader:
            for data in tq_loader:
                #### Mode preds
                pixel_validity = data.data.mvfusion_input[:, :, 0].bool()
                mv_preds = data.data.mvfusion_input[:, :, -1].long()

                valid_m2f_feats = []
                for i in range(len(mv_preds)):
                    valid_m2f_feats.append(mv_preds[i][pixel_validity[i]])

                mode_preds = []
                for m2feats_of_seen_point in valid_m2f_feats:
                    mode_preds.append(torch.mode(m2feats_of_seen_point.squeeze(), dim=0)[0])
                mode_preds = torch.stack(mode_preds, dim=0)

                out_scores = torch.nn.functional.one_hot(mode_preds.squeeze().long(), self._dataset.num_classes).float()                
                
                self._model.output = out_scores
                self._model.labels = data.y
                
                self._tracker.track(self._model, data=data, **self.tracker_options)
                tq_loader.set_postfix(**self._tracker.get_metrics(), color=COLORS.TEST_COLOR)

                if self.profiling:
                    if i > self.num_batches:
                        return 0

        self._finalize_epoch(epoch)
        self._tracker.print_summary()
        
    def _track_2d_results(self, model, mm_data):
        """ Track 2D scores for input semantic segmentation masks and output Multi-View Fusion refined 2D masks using simple nearest-neighbor interpolation and projected 3D point predictions.
        """
        mm_data.data.pred = model.output.detach().cpu().argmax(1)
        
        mappings = mm_data.modalities['image'][0].mappings
        point_ids = torch.arange(
                        mappings.num_groups, device=mappings.device).repeat_interleave(
                        mappings.pointers[1:] - mappings.pointers[:-1])
        image_ids = mappings.images.repeat_interleave(
                        mappings.values[1].pointers[1:] - mappings.values[1].pointers[:-1])    
        pixels_full = mappings.pixels

        # Sort point and image ids based on image_id
        idx_sort = lexargsort(image_ids, point_ids)
        image_ids = image_ids[idx_sort]
        point_ids = point_ids[idx_sort]
        pixels_full = pixels_full[idx_sort].long()

        # Get pointers for easy indexing
        pointers = CSRData._sorted_indices_to_pointers(image_ids)

        # Loop over all N views
        for i, x in enumerate(mm_data.modalities['image'][0]):

            # Grab the 3D points corresponding to ith view
            start, end = pointers[i], pointers[i+1]    
            points = point_ids[start:end]
            pixels = pixels_full[start:end]
            # Image (x, y) pixel index
            w, h = pixels[:, 0], pixels[:, 1]

            # Grab set of points visible in current view
            mm_data_of_view = mm_data[points]
            
            im_ref_w, im_ref_h = x.ref_size

            # Get nearest neighbor interpolated projection image filled with 3D labels
            pred_mask_2d = -1 * torch.ones((im_ref_h, im_ref_w), dtype=torch.long, device=mm_data_of_view.device)    
            pred_mask_2d[h, w] = mm_data_of_view.data.pred.squeeze()
            
            nearest_neighbor = scipy.ndimage.morphology.distance_transform_edt(
                pred_mask_2d==-1, return_distances=False, return_indices=True)    
            pred_mask_2d = pred_mask_2d[nearest_neighbor].numpy().astype(np.uint8)
            pred_mask_2d = Image.fromarray(pred_mask_2d, 'L')            
            pred_mask_2d = np.asarray(pred_mask_2d.resize((640, 480), resample=0))
            
            # 2D mIoU calculation for M2F labels per view
            # Get gt 2d image
            gt_img_path = x.m2f_pred_mask_path[0].split("/")
            # Adjust filepath after Snellius migration
            gt_img_path[1] = 'scratch-shared'
            gt_img_path[-2] = 'label-filt-scannet20'
            gt_img_path = "/".join(gt_img_path)
            gt_img = Image.open(gt_img_path)
            
            
            gt_img = np.asarray(gt_img.resize((640, 480), resample=0)).astype(int) - 1   # -1 label offset

            # Input mask and refined mask for current view
            refined_2d_pred = pred_mask_2d
            
            # Get gt 2d image
            orig_2d_pred = np.asarray(Image.open(x.m2f_pred_mask_path[0])).astype(int) - 1 # x.m2f_pred_mask[0][0]
            
            # 2D segmentation network mIoU
            self._tracker_2d_model_pred_masks.track(
                pred_labels=orig_2d_pred, gt_labels=gt_img, model=None)
                            
            # 2D MVFusion mIoU
            self._tracker_2d_mvfusion_pred_masks.track(
                pred_labels=refined_2d_pred, gt_labels=gt_img, model=None)
            
            # View Entropy Scores (using 2D view gt labels and 3d projected predictions)
            # NOTE: M2F case uses 2D interpolated predictions and 2D gt labels
            # M2F predicted mask for current view
            m2f_labels_2d = x.get_mapped_m2f_features().squeeze()
            
            upscaled_indexing = tuple([2*coor for coor in x.mappings.feature_map_indexing[2:]])
            gt_labels_2d = torch.LongTensor(gt_img[upscaled_indexing])
            
            self._tracker_m2f_entropy.track(pred_labels=m2f_labels_2d, gt_labels=gt_labels_2d, model=None)
            self._tracker_mvfusion_entropy.track(pred_labels=mm_data_of_view.data.pred, gt_labels=gt_labels_2d, model=None)

            mm_data_of_view.data.y[mm_data_of_view.data.y == -1] = 0        # Replace ignored labels in 3D gt
            self._tracker_gt_entropy.track(pred_labels=mm_data_of_view.data.y, gt_labels=gt_labels_2d, model=None)


        return
        
    def get_multiview_entropy_scores(self, tracker):
        confusion_mat = tracker._confusion_matrix.get_confusion_matrix()

        per_class_normalized_entropy = []
        for i in range(len(confusion_mat)):
            nonzero_entries = confusion_mat[:, i][confusion_mat[:, i] > 0]

            # normalized entropy using log2 base
            pk = nonzero_entries / nonzero_entries.sum()

            if len(pk) <= 1:
                per_class_normalized_entropy.append(0.)
            else:
                normalized_entropy = -sum(pk * np.log2(pk)) / np.log2(len(pk))
                per_class_normalized_entropy.append(normalized_entropy)

        return np.mean(np.round(np.array(per_class_normalized_entropy), 4)), np.round(per_class_normalized_entropy, 4)

    def _test_epoch(self, epoch, stage_name: str):
        
        voting_runs = self._cfg.get("voting_runs", 1)
        if stage_name == "test":
            loaders = self._dataset.test_dataloaders
        else:
            loaders = [self._dataset.val_dataloader]

        self._model.eval()
        if self.enable_dropout:
            self._model.enable_dropout_in_eval()

        for loader in loaders:
            stage_name = loader.dataset.name
            self._tracker.reset(stage_name)
            
            if self._tracker_2d_mvfusion_pred_masks is not None:
                self._tracker_2d_mvfusion_pred_masks.reset(stage_name)
                self._tracker_2d_model_pred_masks.reset(stage_name)
                
                self._tracker_m2f_entropy.reset(stage_name)
                self._tracker_mvfusion_entropy.reset(stage_name)
                self._tracker_gt_entropy.reset(stage_name)
            
            if self.has_visualization:
                self._visualizer.reset(epoch, stage_name)
            if not self._dataset.has_labels(stage_name) and not self.tracker_options.get(
                "make_submission", False
            ):  # No label, no submission -> do nothing
                log.warning("No forward will be run on dataset %s." % stage_name)
                continue

            for i in range(voting_runs):
                with Ctq(loader) as tq_loader:
                    for data in tq_loader:
                        with torch.no_grad():

                            self._model.set_input(data, self._device)
                            with torch.cuda.amp.autocast(enabled=self._model.is_mixed_precision()):
                                self._model.forward(epoch=epoch)
                            # 3D mIoU
                            self._tracker.track(self._model, data=data, **self.tracker_options)
                            
                            # 2D mIoU of both original and refined semantic label masks
                            # also tracks view entropy metric
                            if self._tracker_2d_mvfusion_pred_masks is not None:
                                self._track_2d_results(self._model, data)
                            
                        tq_loader.set_postfix(**self._tracker.get_metrics(), color=COLORS.TEST_COLOR)

                        if self.has_visualization and self._visualizer.is_active:
                            self._visualizer.save_visuals(self._model.get_current_visuals())

                        if self.early_break:
                            break

                        if self.profiling:
                            if i > self.num_batches:
                                return 0
            log.info("Evaluated scores for 3D semantic segmentation: ")
            self._finalize_epoch(epoch)
            self._tracker.print_summary()
            cm = self._tracker._confusion_matrix.confusion_matrix
            confusion_m_dir = "/home/fsun/DeepViewAgg/notebooks/confusion_matrix/mvfusion_3d"
            os.makedirs(confusion_m_dir, exist_ok=True)
            save_confusion_matrix(cm, path2save=confusion_m_dir, ordered_names=CLASS_LABELS)
            
            # Finalise 2D evaluation
            if self._tracker_2d_mvfusion_pred_masks is not None:
                log.info("Evaluated scores for 2D refined masks: ")
                self._tracker_2d_mvfusion_pred_masks.finalise(**self.tracker_options)
                self._tracker_2d_mvfusion_pred_masks.print_summary()
                cm = self._tracker_2d_mvfusion_pred_masks._confusion_matrix.confusion_matrix
                confusion_m_dir = "/home/fsun/DeepViewAgg/notebooks/confusion_matrix/mvfusion_2d"
                os.makedirs(confusion_m_dir, exist_ok=True)
                save_confusion_matrix(cm, path2save=confusion_m_dir, ordered_names=CLASS_LABELS)
                
                log.info("Evaluated scores for 2D input masks: ")
                self._tracker_2d_model_pred_masks.finalise(**self.tracker_options)
                self._tracker_2d_model_pred_masks.print_summary()
                cm = self._tracker_2d_model_pred_masks._confusion_matrix.confusion_matrix
                confusion_m_dir = "/home/fsun/DeepViewAgg/notebooks/confusion_matrix/baseline_ViT_2d"
                os.makedirs(confusion_m_dir, exist_ok=True)
                save_confusion_matrix(cm, path2save=confusion_m_dir, ordered_names=CLASS_LABELS)
                
                
                # Incorrect view entropy metric
#                 log.info("Evaluated view entropy scores for input masks (using 2D-3D correspondences and 3D GT): ")
#                 self._tracker_m2f_entropy.finalise(**self.tracker_options)
#                 self._tracker_m2f_entropy.print_summary()
#                 mean_view_entropy, per_class_view_entropy = self.get_multiview_entropy_scores(self._tracker_m2f_entropy)
#                 log.info(f"mean_view_entropy: {mean_view_entropy}")
#                 log.info(f"per_class_view_entropy: {per_class_view_entropy}")

                    
#                 log.info("Evaluated view entropy scores for refined predictions (using 2D-3D correspondences and 3D GT): ")
#                 self._tracker_mvfusion_entropy.finalise(**self.tracker_options)
#                 self._tracker_mvfusion_entropy.print_summary()
#                 mean_view_entropy, per_class_view_entropy = self.get_multiview_entropy_scores(self._tracker_mvfusion_entropy)
#                 log.info(f"mean_view_entropy: {mean_view_entropy}")
#                 log.info(f"per_class_view_entropy: {per_class_view_entropy}")

                
#                 log.info("Evaluated view entropy scores for gt masks (using 2D-3D correspondences and 3D GT): ")
#                 self._tracker_gt_entropy.finalise(**self.tracker_options)
#                 self._tracker_gt_entropy.print_summary()
#                 mean_view_entropy, per_class_view_entropy = self.get_multiview_entropy_scores(self._tracker_gt_entropy)
#                 log.info(f"mean_view_entropy: {mean_view_entropy}")
#                 log.info(f"per_class_view_entropy: {per_class_view_entropy}")
                
    @property
    def early_break(self):
        if not hasattr(self._cfg, "debugging"):
            return False
        return getattr(self._cfg.debugging, "early_break", False) and self._is_training

    @property
    def profiling(self):
        if not hasattr(self._cfg, "debugging"):
            return False
        return getattr(self._cfg.debugging, "profiling", False)

    @property
    def num_batches(self):
        if not hasattr(self._cfg, "debugging"):
            return None
        return getattr(self._cfg.debugging, "num_batches", 50)

    @property
    def enable_cudnn(self):
        if self.has_training:
            return getattr(self._cfg.training, "enable_cudnn", True)
        else:
            return getattr(self._cfg, "enable_cudnn", True)

    @property
    def enable_dropout(self):
        return getattr(self._cfg, "enable_dropout", True)

    @property
    def has_visualization(self):
        return getattr(self._cfg, "visualization", False)

    @property
    def has_tensorboard(self):
        return getattr(self._cfg, "tensorboard", False)

    @property
    def has_training(self):
        return getattr(self._cfg, "training", None)

    @property
    def precompute_multi_scale(self):
        if not self.has_training:
            return self._model.conv_type == "PARTIAL_DENSE" and getattr(self._cfg.training, "precompute_multi_scale", False)
        else:
            return self._model.conv_type == "PARTIAL_DENSE" and getattr(self._cfg, "precompute_multi_scale", False)

    @property
    def wandb_log(self):
        if not self.has_training:
            return False
        return getattr_recursive(self._cfg, 'training.wandb.log', False)
        
    @property
    def tensorboard_log(self):
        if not self.has_training:
            return False
        return getattr_recursive(self._cfg, 'training.tensorboard.log', False)
        
    @property
    def tracker_options(self):
        return self._cfg.get("tracker_options", {})

    @property
    def eval_frequency(self):
        return self._cfg.get("eval_frequency", 1)
    @property
    def lr_range_test(self):
        return self._cfg.get("lr_range_test", False)

    def _do_lr_test(self, epoch=1):
        self._model.train()
        self._tracker.reset("train")
        self._visualizer.reset(epoch, "train")
        train_loader = self._dataset.train_dataloader

        iter_data_time = time.time()
        
        running_loss = 0.
        avg_beta = 0.98
        clr = CLR(self._model._optimizer, len(train_loader),
                 base_lr=0.0001, max_lr=100)
        update_lr(self._model._optimizer, lr=0.0001)  


        with Ctq(train_loader) as tq_train_loader:
            for i, data in enumerate(tq_train_loader):
                
                if self._dataset.dataset_opt.train_with_mix3d:
                    data = PointcloudMerge(data, n_merge=2)
                
                t_data = time.time() - iter_data_time
                iter_start_time = time.time()
                self._model.set_input(data, self._device)
                cur_lr = self._model.learning_rate
                self._model.optimize_parameters(epoch, self._dataset.batch_size)
                
                loss = self._model.loss_seg
                
                running_loss = avg_beta * running_loss + (1-avg_beta) * loss.item()
                smoothed_loss = running_loss / (1 - avg_beta**(i+1))
                
                with torch.no_grad():
                    self._tracker.track(
                        self._model, data=data, **self.tracker_options)
                
                iter_data_time = time.time()
                tq_train_loader.set_postfix(
                    **self._tracker.get_metrics(),
                    data_loading=float(t_data),
                    iteration=float(time.time() - iter_start_time),
                    train_loss_smooth=smoothed_loss,
                    learning_rate=cur_lr,
                    color=COLORS.TRAIN_COLOR
                )
                
                lr = clr.calc_lr(smoothed_loss)
                if lr == -1 :
                    break
                update_lr(self._model._optimizer, lr)  

#         self._finalize_epoch(epoch)
        clr.plot()
        self._tracker.finalise(**self.tracker_options)
        self._tracker.print_summary()
        
    
def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

def update_mom(optimizer, mom):
    for g in optimizer.param_groups:
        g['momentum'] = mom
        
import math
import matplotlib.pyplot as plt

class CLR(object):
    """
    The method is described in paper : https://arxiv.org/abs/1506.01186 to find out optimum 
    learning rate. The learning rate is increased from lower value to higher per iteration 
    for some iterations till loss starts exploding.The learning rate one power lower than 
    the one where loss is minimum is chosen as optimum learning rate for training.
    Args:
        optim   Optimizer used in training.
        bn      Total number of iterations used for this test run.
                The learning rate increasing factor is calculated based on this 
                iteration number.
        base_lr The lower boundary for learning rate which will be used as
                initial learning rate during test run. It is adviced to start from
                small learning rate value like 1e-4.
                Default value is 1e-5
        max_lr  The upper boundary for learning rate. This value defines amplitude
                for learning rate increase(max_lr-base_lr). max_lr value may not be 
                reached in test run as loss may explode before reaching max_lr.
                It is adviced to use higher value like 10, 100.
                Default value is 100.
    """
    def __init__(self, optim, bn, base_lr=1e-5, max_lr=100):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.optim = optim
        self.bn = bn - 1
        ratio = self.max_lr/self.base_lr
        self.mult = ratio ** (1/self.bn)
        self.best_loss = 1e9
        self.iteration = 0
        self.lrs = []
        self.losses = []
        
    def calc_lr(self, loss):
        self.iteration +=1
        if math.isnan(loss) or loss > 4 * self.best_loss:
            return -1
        if loss < self.best_loss and self.iteration > 1:
            self.best_loss = loss
            
        mult = self.mult ** self.iteration
        lr = self.base_lr * mult
        
        self.lrs.append(lr)
        self.losses.append(loss)
        
        return lr
        
    def plot(self, start=10, end=-5):
        plt.xlabel("Learning Rate")
        plt.ylabel("Losses")
        plt.plot(self.lrs[start:end], self.losses[start:end])
        plt.xscale('log')
        plt.savefig("learning_rate_test.png")