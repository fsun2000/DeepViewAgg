import os
import os.path as osp
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
from collections import defaultdict


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


def plot_metrics(filepath, train_data, test_data):
    for k, v in train_data.items():
        plt.plot(v, label=k)

    for k, v in test_data.items():
        plt.plot(v, label=k)

    ax = plt.gca()
    ax.set_ylim([0, 100])
    plt.legend()
    plt.grid()
    plt.savefig(filepath)    

    print(train_data)
    print(test_data)

class Trainer:
    """
    TorchPoints3d Trainer handles the logic between
        - BaseModel,
        - Dataset and its Tracker
        - A custom ModelCheckpoint
        - A custom Visualizer
    It supports MC dropout - multiple voting_runs for val / test datasets
    """

    def __init__(self, cfg):
        self._cfg = cfg
        self._initialize_trainer()
        
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
            
        self.early_stopping = True if self._cfg.early_stopping is not False else False
        self.patience = self._cfg.early_stopping
        self.encoder_lr_factor = self._cfg.encoder_lr_factor
        
        # Adjust lr factor for last parameter group (which is usally the view-fusion/encoder module)
        self._model._optimizer.param_groups[-1]['lr'] *= self.encoder_lr_factor
        
        self._model.log_optimizers()


    def train(self):
        self._is_training = True
        
        best_mIoU = 0.0
        best_mIoU_epoch = 0
        epoch_no_change = 0
        train_metrics = defaultdict(lambda: [])
        test_metrics = defaultdict(lambda: [])
        
        for epoch in range(self._checkpoint.start_epoch, self._cfg.training.epochs + 1):
            log.info("EPOCH %i / %i", epoch, self._cfg.training.epochs)


            self._train_epoch(epoch)
            train_out = self._tracker.get_metrics()
                
            # clear GPU cache as it might fragment a lot
            torch.cuda.empty_cache()

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
                self._test_epoch(epoch, "val")
                test_out = self._tracker.get_metrics()

            # Training/testing metrics
            for k, v in train_out.items():
                if k == 'train_loss_seg':
                    v = v * 100
                train_metrics[k].append(v)
            for k, v in test_out.items():
                if k == 'val_loss_seg':
                    v = v * 100
                test_metrics[k].append(v)

            
            output_plot_dir = f"/home/fsun/DeepViewAgg/job_logs/thesis_results/superconvergence_adamw/plots/{self._cfg.data.m2f_preds_dirname}"
            if not osp.exists(output_plot_dir):
                os.makedirs(output_plot_dir, exist_ok=True)
            fp = f"{output_plot_dir}/{self._cfg.model_name}_{self._cfg.training.epochs}.png"
            plot_metrics(fp, train_metrics, test_metrics)
                
            if self.early_stopping:
                if test_out['val_miou'] < best_mIoU:
                    epoch_no_change += 1
                else:
                    best_mIoU = test_out['val_miou']
                    best_mIoU_epoch = epoch
                    epoch_no_change = 0
                if epoch_no_change >= self.patience:
                    print("Number of epochs without validation improvement: ", epoch_no_change)
                    print(f"Best epoch: {best_mIoU_epoch} = {best_mIoU}")
                    print("Stopping training early to prevent overfitting")
                    break
                   
            # Log optimizer
            self._model.log_optimizers()
                
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
        else:
            # After training finished, run final evaluation on 2D and 3D
            
            # Create val loader and delete train loader
            self._dataset.create_dataloaders(
                self._model,
                self._cfg.training.batch_size,
                self._cfg.training.shuffle,
                self._cfg.training.num_workers,
                self.precompute_multi_scale,
                train_only=False,
                val_only=True
            )
            
            log.info("Evaluating model on 3D seen points")
            self.eval_3d_seen_points(stage_name="val")

    def eval(self, stage_name=""):
        self._is_training = False
        
        print("trainer.py: Tracking 2D mask and 2D refined mask scores!")
        self._tracker_2d_model_pred_masks: BaseTracker = self._dataset.get_tracker(
            self.wandb_log, self.tensorboard_log)
        self._tracker_2d_mvfusion_pred_masks: BaseTracker = self._dataset.get_tracker(
            self.wandb_log, self.tensorboard_log)
            
            
        epoch = self._checkpoint.start_epoch
        
        if self._dataset.has_val_loader:
            if not stage_name or stage_name == "val":
                self._test_epoch(epoch, "val")

        # Free memory
        del self._dataset._val_loader

        if self._dataset.has_test_loaders:
            if not stage_name or stage_name == "test":
                self._test_epoch(epoch, "test")
                
    def eval_3d_seen_points(self, stage_name=""):
        self._is_training = False
        
        self._tracker_baseline: BaseTracker = self._dataset.get_tracker(
            self.wandb_log, self.tensorboard_log)
        self._tracker_mvfusion: BaseTracker = self._dataset.get_tracker(
            self.wandb_log, self.tensorboard_log)
            
        print("trainer.py: Tracking 2D mask and 2D refined mask scores!")
        self._tracker_2d_model_pred_masks: BaseTracker = self._dataset.get_tracker(
            self.wandb_log, self.tensorboard_log)
        self._tracker_2d_mvfusion_pred_masks: BaseTracker = self._dataset.get_tracker(
            self.wandb_log, self.tensorboard_log)
                        
        epoch = self._checkpoint.start_epoch

        if self._dataset.has_val_loader:
            if not stage_name or stage_name == "val":
                self._test_epoch_3d_seen_points(epoch, "val")
                
    def eval_3d_seen_points_view_selection_experiment(self, stage_name=""):
        self._is_training = False
        
        self._tracker_baseline: BaseTracker = self._dataset.get_tracker(
            self.wandb_log, self.tensorboard_log)
        self._tracker_mvfusion: BaseTracker = self._dataset.get_tracker(
            self.wandb_log, self.tensorboard_log)
            
        print("trainer.py: Tracking 2D mask and 2D refined mask scores!")
        self._tracker_2d_model_pred_masks: BaseTracker = self._dataset.get_tracker(
            self.wandb_log, self.tensorboard_log)
        self._tracker_2d_mvfusion_pred_masks: BaseTracker = self._dataset.get_tracker(
            self.wandb_log, self.tensorboard_log)
                        
        epoch = self._checkpoint.start_epoch

        if self._dataset.has_val_loader:
            if not stage_name or stage_name == "val":
                self._test_epoch_3d_seen_points_view_selection_experiment(epoch, "val")
                

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

                mix3d_time = 0.0
                if self._dataset.dataset_opt.train_with_mix3d and self._dataset.batch_size > 1:
                    mix3d_time = time.time()
                    data = PointcloudMerge(data, n_merge=2)
                    mix3d_time = time.time() - mix3d_time
                
                iter_start_time = time.time()
                self._model.set_input(data, self._device)
                self._model.optimize_parameters(epoch, self._dataset.batch_size)
                
                # Adjust lr factor for last parameter group (which is usally the view-fusion/encoder module)
                self._model._optimizer.param_groups[-1]['lr'] *= self.encoder_lr_factor
                
                lr = ' '.join(str(x['lr']) for x in self._model._optimizer.param_groups)
                
                if i % 10 == 0:
                    with torch.no_grad():
                        self._tracker.track(
                            self._model, data=data, **self.tracker_options)

                tq_train_loader.set_postfix(
                    **self._tracker.get_metrics(),
                    data_loading=float(t_data),
                    mix_3d=float(mix3d_time),
                    iteration=float(time.time() - iter_start_time),
                    learning_rate=lr, #self._model.learning_rate,
                    max_mem=torch.cuda.max_memory_allocated()/1048576,
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

        
    def _track_2d_results(self, model, mm_data, contains_pred=False, save_output=False):
        """ Track 2D scores for input semantic segmentation masks and output Multi-View Fusion refined 2D masks using simple nearest-neighbor interpolation and projected 3D point predictions.
        """
        if contains_pred == False:
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

        # Save refined masks
        im_paths = mm_data.modalities['image'][0].gt_mask_path
        scan_dir = os.sep.join(im_paths[0].split(os.sep)[:-2])
        input_mask_name = mm_data.modalities['image'][0].m2f_pred_mask_path[0].split(os.sep)[-2]

        # Dirty workaround for masks in different directory
        if input_mask_name == 'ViT_masks':
            scan_id = scan_dir.split(os.sep)[-1]
            mask_im_dir = osp.join("/home/fsun/data/scannet/scans", scan_id, input_mask_name)
            refined_mask_im_dir = osp.join(scan_dir, input_mask_name + '_refined')
        else:
            mask_im_dir = osp.join(scan_dir, input_mask_name)
            refined_mask_im_dir = osp.join(scan_dir, input_mask_name + '_refined')
            
        if save_output:
            print("Creating refined mask dir at ", refined_mask_im_dir)
            os.makedirs(refined_mask_im_dir, exist_ok=True)
        
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
            
            # SAVE REFINED MASK IN GIVEN DIR
            im_name = x.m2f_pred_mask_path[0].split("/")[-1]
        
            pred_mask_2d = pred_mask_2d.resize((640, 480), resample=0)
            
            if save_output:
                pred_mask_2d.save(osp.join(refined_mask_im_dir, im_name))

            pred_mask_2d = np.asarray(pred_mask_2d)
            
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
            

        return

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

            
            # Finalise 2D evaluation
            if self._tracker_2d_mvfusion_pred_masks is not None:
            
                log.info("Evaluated scores for 2D refined masks: ")
                self._tracker_2d_mvfusion_pred_masks.finalise(**self.tracker_options)
                self._tracker_2d_mvfusion_pred_masks.print_summary()

                log.info("Evaluated scores for 2D input masks: ")
                self._tracker_2d_model_pred_masks.finalise(**self.tracker_options)
                self._tracker_2d_model_pred_masks.print_summary()
                
    def _test_epoch_3d_seen_points(self, epoch, stage_name: str):
        
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

        
        voting_runs = self._cfg.get("voting_runs", 1)
        if stage_name == "test":
            loaders = self._dataset.test_dataloaders
        else:
            loaders = [self._dataset.val_dataloader]

        self._model.eval()
        if self.enable_dropout:
            self._model.enable_dropout_in_eval()
            
        count = 0

        for loader in loaders:
            print("Input mask type: ", loader.dataset.m2f_preds_dirname)
            
            stage_name = loader.dataset.name
            self._tracker_baseline.reset(stage_name)
            self._tracker_mvfusion.reset(stage_name)
            
            self._tracker_2d_mvfusion_pred_masks.reset(stage_name)
            self._tracker_2d_model_pred_masks.reset(stage_name)

            for i in range(voting_runs):
                with Ctq(loader) as tq_loader:
                    for data in tq_loader:
                        with torch.no_grad():

                            self._model.set_input(data, self._device)
                            with torch.cuda.amp.autocast(enabled=self._model.is_mixed_precision()):
                                self._model.forward(epoch=epoch)
                                
                            data.data.pred = self._model.output.detach().cpu().argmax(1)
                            
                            if count < 5:
                                print("Before subsampling")
                                print(data, flush=True)
                                print(data.data.pred, data.data.pred.shape, flush=True)
                                print(data.data.y, data.data.y.shape, flush=True)
                                
                            data = get_seen_points(data)
                            mode_pred = get_mode_pred(data)
                            
                            count += 1
                            if count < 5:
                                print("After subsampling")
                                print(data, flush=True)
                                print(data.data.pred, data.data.pred.shape, flush=True)
                                print(data.data.y, data.data.y.shape, flush=True)
                            
                            # 3D mIoU
                            self._tracker_baseline.track(pred_labels=mode_pred, gt_labels=data.data.y, model=None)
                            self._tracker_mvfusion.track(pred_labels=data.data.pred, gt_labels=data.data.y, model=None)
                            
                            # 2D mIoU
                            self._track_2d_results(self._model, data, contains_pred=True, save_output=False)
                            
                        tq_loader.set_postfix(**self._tracker_mvfusion.get_metrics(), color=COLORS.TEST_COLOR)


            log.info("Evaluated scores for 3D semantic segmentation on subset of seen points: ")
            self._finalize_epoch(epoch)
            log.info("--- Baseline 3D ---")
            self._tracker_baseline.print_summary()

            log.info("--- Baseline 2D ---")
            self._tracker_2d_model_pred_masks.print_summary()
            
            log.info("--- MVFusion_3D 3D ---")
            self._tracker_mvfusion.print_summary()

            log.info("--- MVFusion_3D 2D ---")
            self._tracker_2d_mvfusion_pred_masks.print_summary()
            
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
    
    def _test_epoch_3d_seen_points_view_selection_experiment(self, epoch, stage_name: str):
        
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
        
        voting_runs = self._cfg.get("voting_runs", 1)
        if stage_name == "test":
            loaders = self._dataset.test_dataloaders
        else:
            loaders = [self._dataset.val_dataloader]

        self._model.eval()
        if self.enable_dropout:
            self._model.enable_dropout_in_eval()
            

        for loader in loaders:
            print("Input mask type: ", loader.dataset.m2f_preds_dirname)
            
            stage_name = loader.dataset.name
            self._tracker_baseline.reset(stage_name)
            self._tracker_mvfusion.reset(stage_name)
            
            self._tracker_2d_mvfusion_pred_masks.reset(stage_name)
            self._tracker_2d_model_pred_masks.reset(stage_name)

            for i in range(voting_runs):
                with Ctq(loader) as tq_loader:
                    for data in tq_loader:
                        with torch.no_grad():
                            
                            ### Store average / random prediction per point
                            random_or_average_pred = data.data.pred.clone()
                            
                                
                            data = get_seen_points(data)
                            mode_pred = get_mode_pred(data)
                            
                            
                            # 3D mIoU
                            self._tracker_baseline.track(pred_labels=mode_pred, gt_labels=data.data.y, model=None)
                            self._tracker_mvfusion.track(pred_labels=random_or_average_pred, gt_labels=data.data.y, model=None)
                            
                            # 2D mIoU
                            self._track_2d_results(self._model, data, contains_pred=True, save_output=False)
                            
                        tq_loader.set_postfix(**self._tracker_mvfusion.get_metrics(), color=COLORS.TEST_COLOR)


            log.info("Evaluated scores for 3D semantic segmentation on subset of seen points: ")
            self._finalize_epoch(epoch)
            log.info("--- Mode Pred 3D ---")
            self._tracker_baseline.print_summary()
            log.info("--- Baseline 2D (input masks) ---")
            self._tracker_2d_model_pred_masks.print_summary()
            log.info("--- Mode Pred 3D ---")
            self._tracker_mvfusion.print_summary()
            log.info("--- Mode Pred projected to 2D views ---")
            self._tracker_2d_mvfusion_pred_masks.print_summary()

    
