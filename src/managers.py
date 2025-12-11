"""
Manager Classes for Metrics, Dataset, and Results Organization.

This module provides manager classes for collecting metrics, handling dataset
file operations, and organizing output directory structures.
"""

from pathlib import Path
from typing import List

import torch
from PIL import Image
import numpy as np

from src.data_models import ImageMetrics, DatasetSummary
from src.metrics import compute_pixel_accuracy, compute_iou, compute_ap
from src.metrics_fidelity import average_drop_increase, insertion_auc, deletion_auc, average_gain


class MetricsCollector:
    """
    Collects and aggregates per-image metrics into dataset-level statistics.
    
    Uses the Collector design pattern to accumulate metrics from individual
    images and compute aggregated statistics (means, counts) at the dataset level.
    Enables flexible metrics tracking without coupling computation to storage.
    
    Attributes:
        image_metrics (List[ImageMetrics]): List of metrics from all images
    """
    
    def __init__(self) -> None:
        """Initialize empty metrics collection."""
        self.image_metrics: List[ImageMetrics] = []
    
    def add_metrics(self, metrics: ImageMetrics) -> None:
        """
        Add metrics from a single image to the collection.
        
        Args:
            metrics (ImageMetrics): Per-image metrics to store
        """
        self.image_metrics.append(metrics)
    
    def get_summary(
        self, 
        dataset_tag: str, 
        method: str, 
        num_overlays_saved: int
    ) -> DatasetSummary:
        """
        Generate aggregated dataset-level summary statistics.
        
        Computes mean values and counts for all metrics across all collected
        images. Handles optional metrics gracefully.
        
        Args:
            dataset_tag (str): Name/tag of the dataset
            method (str): Name of the explanation method
            num_overlays_saved (int): Number of overlay images saved
            
        Returns:
            DatasetSummary: Aggregated statistics for the entire dataset
        """
        if not self.image_metrics:
            return DatasetSummary(
                dataset=dataset_tag,
                method=method,
                num_images=0,
                num_overlays_saved=0,
                count_nonmask=0,
                mean_avgdrop=None,
                avg_increase_ratio_pct=None,
                mean_avg_gain=None,
                count_avg_gain=0,
                mean_insertion_auc=None,
                mean_deletion_auc=None,
                num_masked_images=0,
                mean_pixelacc=None,
                mean_iou=None,
                mean_ap=None,
            )
        
        drops = [m.avg_drop for m in self.image_metrics]
        increases = [m.avg_increase for m in self.image_metrics]
        gains = [m.average_gain for m in self.image_metrics]
        insertions = [m.insertion_auc for m in self.image_metrics]
        deletions = [m.deletion_auc for m in self.image_metrics]
        
        pixel_accs = [m.pixel_acc for m in self.image_metrics if m.pixel_acc is not None]
        ious = [m.iou for m in self.image_metrics if m.iou is not None]
        aps = [m.ap for m in self.image_metrics if m.ap is not None]
        
        return DatasetSummary(
            dataset=dataset_tag,
            method=method,
            num_images=len(self.image_metrics),
            num_overlays_saved=num_overlays_saved,
            count_nonmask=len(drops),
            mean_avgdrop=(sum(drops) / len(drops)) if drops else None,
            avg_increase_ratio_pct=(100.0 * sum(increases) / len(increases)) if increases else None,
            mean_avg_gain=(sum(gains) / len(gains)) if gains else None,
            count_avg_gain=len(gains),
            mean_insertion_auc=(sum(insertions) / len(insertions)) if insertions else None,
            mean_deletion_auc=(sum(deletions) / len(deletions)) if deletions else None,
            num_masked_images=len(pixel_accs),
            mean_pixelacc=(sum(pixel_accs) / len(pixel_accs)) if pixel_accs else None,
            mean_iou=(sum(ious) / len(ious)) if ious else None,
            mean_ap=(sum(aps) / len(aps)) if aps else None,
        )


class DatasetManager:
    """
    Utility manager for dataset file operations.
    
    Provides static methods for common file system operations related to datasets,
    including image discovery and dataset root detection. Centralizes file handling
    logic for easy testing and reusability.
    """
    
    @staticmethod
    def list_images(folder: Path) -> List[Path]:
        """
        List all image files in the specified folder.
        
        Searches for common image formats (jpg, jpeg, png, JPEG) and returns
        them in sorted order.
        
        Args:
            folder (Path): Directory to search for images
            
        Returns:
            List[Path]: Sorted list of image file paths found
        """
        exts = ["*.jpg", "*.jpeg", "*.png", "*.JPEG"]
        files = []
        for pat in exts:
            files += list(folder.glob(pat))
        return sorted(files)
    
    @staticmethod
    def find_tinyimagenet_root(base: Path) -> Path:
        """
        Find and return the root directory of a TinyImageNet dataset.
        
        Searches for the characteristic 'train' and 'val' subdirectories.
        Checks multiple common naming conventions for TinyImageNet.
        
        Args:
            base (Path): Base directory to search for TinyImageNet
            
        Returns:
            Path: Path to the TinyImageNet root directory
            
        Raises:
            FileNotFoundError: If TinyImageNet structure not found
        """
        if (base / "train").exists() and (base / "val").exists():
            return base
        for c in [base / "tiny-imagenet-200", base / "tiny-imagenet"]:
            if (c / "train").exists() and (c / "val").exists():
                return c
        raise FileNotFoundError(f"Could not find train/val under {base}")


class ResultsDirectoryManager:
    """
    Manager for output directory structure organization.
    
    Handles creation and organization of results directories, ensuring
    method-specific subdirectories are created as needed.
    
    Attributes:
        results_root (Path): Root directory for all results
    """
    
    def __init__(self, results_root: Path) -> None:
        """
        Initialize results directory manager.
        
        Args:
            results_root (Path): Root directory where results will be saved.
                                Created if it doesn't exist.
        """
        self.results_root = results_root
        self.results_root.mkdir(parents=True, exist_ok=True)
    
    def ensure_method_dir(self, dataset_tag: str, method: str) -> Path:
        """
        Ensure and return method-specific results directory.
        
        Creates the directory structure results_root/dataset_tag/method/
        if it doesn't already exist.
        
        Args:
            dataset_tag (str): Tag/name of the dataset (e.g., 'local', 'tiny_test')
            method (str): Name of the explanation method (e.g., 'gmar', 'legrad')
            
        Returns:
            Path: Path to the created method-specific directory
        """
        out_dir = self.results_root / dataset_tag / method
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir


class ImageMetricsComputer:
    """
    Computes comprehensive metrics for a single image's explanation.
    
    This class handles the computation of both fidelity metrics and mask-based metrics
    for single image explanations, encapsulating metric computation logic.
    
    Attributes:
        mean (tuple): Image normalization mean values
        std (tuple): Image normalization std values
    """
    
    def __init__(self, mean: tuple, std: tuple) -> None:
        """
        Initialize metrics computer with normalization parameters.
        
        Args:
            mean (tuple): Image normalization mean values
            std (tuple): Image normalization std values
        """
        self.mean = mean
        self.std = std
    
    def compute_all_metrics(
        self,
        vit_model,
        img_tensor: torch.Tensor,
        pred_idx: int,
        final_map: torch.Tensor,
        mask_dir: Path = None,
        mask_file_stem: str = None,
    ) -> ImageMetrics:
        """
        Compute all evaluation metrics for a single image.
        
        Computes both fidelity metrics and mask-based metrics for a single image's
        explanation. Fidelity metrics measure explanation quality by evaluating model
        sensitivity to masked regions. Mask-based metrics evaluate against ground truth.
        
        Args:
            vit_model: Vision Transformer model for computing logits
            img_tensor (torch.Tensor): Preprocessed image tensor (normalized, shape 1×3×H×W)
            pred_idx (int): Predicted class index
            final_map (torch.Tensor): 2D heatmap from explanation method (shape H×W)
            mask_dir (Path): Directory containing ground truth masks (optional)
            mask_file_stem (str): Filename stem to match with mask file
                
        Returns:
            ImageMetrics: Dataclass containing all computed metrics
        """
        drop, inc, s_full, s_masked = None, None, None, None
        ag, p_full, p_masked = None, None, None
        ins_auc, del_auc = None, None
        
        try:
            drop, inc, s_full, s_masked = average_drop_increase(
                model=vit_model,
                x_norm=img_tensor,
                class_idx=pred_idx,
                heatmap_2d=final_map,
                mean=self.mean,
                std=self.std,
                keep_ratio=0.25,
                patchwise=True,
                blur_ksize=21,
                blur_sigma=7.0,
            )
        except (RuntimeError, ValueError) as e:
            print(f"  [WARN] average_drop_increase failed: {e}")
        except Exception as e:
            print(f"  [WARN] Unexpected error in average_drop_increase: {type(e).__name__}: {e}")
        
        try:
            ag, p_full, p_masked = average_gain(
                model=vit_model,
                x_norm=img_tensor,
                class_idx=pred_idx,
                heatmap_2d=final_map,
                mean=self.mean,
                std=self.std,
                keep_ratio=0.25,
                patchwise=True,
                blur_ksize=21,
                blur_sigma=7.0,
            )
        except (RuntimeError, ValueError) as e:
            print(f"  [WARN] average_gain failed: {e}")
        except Exception as e:
            print(f"  [WARN] Unexpected error in average_gain: {type(e).__name__}: {e}")
        
        try:
            ins_auc = insertion_auc(
                model=vit_model,
                x_norm=img_tensor,
                class_idx=pred_idx,
                heatmap_2d=final_map,
                mean=self.mean,
                std=self.std,
                steps=100,
                patchwise=True,
                blur_ksize=21,
                blur_sigma=7.0,
            )
        except (RuntimeError, ValueError) as e:
            print(f"  [WARN] insertion_auc failed: {e}")
        except Exception as e:
            print(f"  [WARN] Unexpected error in insertion_auc: {type(e).__name__}: {e}")
        
        try:
            del_auc = deletion_auc(
                model=vit_model,
                x_norm=img_tensor,
                class_idx=pred_idx,
                heatmap_2d=final_map,
                mean=self.mean,
                std=self.std,
                steps=100,
                patchwise=True,
                blur_ksize=21,
                blur_sigma=7.0,
            )
        except (RuntimeError, ValueError) as e:
            print(f"  [WARN] deletion_auc failed: {e}")
        except Exception as e:
            print(f"  [WARN] Unexpected error in deletion_auc: {type(e).__name__}: {e}")
        
        pixel_acc = None
        iou = None
        ap = None
        
        if mask_dir is not None and mask_file_stem is not None:
            pixel_acc, iou, ap = self._compute_mask_metrics(
                final_map, mask_dir, mask_file_stem
            )
        
        return ImageMetrics(
            avg_drop=drop,
            avg_increase=inc,
            logit_full=s_full,
            logit_masked=s_masked,
            average_gain=ag,
            prob_full=p_full,
            prob_masked=p_masked,
            insertion_auc=ins_auc,
            deletion_auc=del_auc,
            pixel_acc=pixel_acc,
            iou=iou,
            ap=ap,
        )
    
    @staticmethod
    def _compute_mask_metrics(
        final_map: torch.Tensor,
        mask_dir: Path,
        mask_file_stem: str,
    ) -> tuple:
        """
        Compute mask-based metrics if ground truth mask is available.
        
        Args:
            final_map (torch.Tensor): 2D explanation heatmap
            mask_dir (Path): Directory containing masks
            mask_file_stem (str): Filename stem for mask lookup
            
        Returns:
            tuple: (pixel_acc, iou, ap) - all None if mask not found
        """
        pixel_acc = None
        iou = None
        ap = None
        
        try:
            mask_file = mask_dir / (mask_file_stem + ".png")
            if not mask_file.exists():
                return pixel_acc, iou, ap
            
            try:
                gt_mask_img = Image.open(mask_file).convert('L').resize(
                    final_map.shape[::-1], resample=Image.NEAREST
                )
                gt_mask = torch.tensor(np.array(gt_mask_img) > 127).int()
                
                fm = final_map
                fm_min, fm_max = float(fm.min()), float(fm.max())
                fm = (fm - fm_min) / (fm_max - fm_min + 1e-8)
                pred_mask = (fm > 0.5).int()
                
                try:
                    pixel_acc = float(compute_pixel_accuracy(pred_mask, gt_mask))
                except Exception as e:
                    print(f"  [WARN] compute_pixel_accuracy failed: {e}")
                
                try:
                    iou = float(compute_iou(pred_mask, gt_mask))
                except Exception as e:
                    print(f"  [WARN] compute_iou failed: {e}")
                
                try:
                    ap = float(compute_ap(fm, gt_mask))
                except Exception as e:
                    print(f"  [WARN] compute_ap failed: {e}")
            except (IOError, OSError) as e:
                print(f"  [WARN] Failed to load or process mask file {mask_file}: {e}")
            except Exception as e:
                print(f"  [WARN] Unexpected error processing mask: {type(e).__name__}: {e}")
        except Exception as e:
            print(f"  [WARN] Error handling mask directory: {type(e).__name__}: {e}")
        
        return pixel_acc, iou, ap
