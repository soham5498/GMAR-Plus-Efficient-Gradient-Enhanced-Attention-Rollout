"""
Main Explainability Engine - Facade for coordinating all analysis.

This module provides the ExplainabilityEngine class which orchestrates the
complete explainability workflow including model initialization, dataset processing,
and metrics aggregation.
"""

from pathlib import Path
from typing import List, Optional
import torch
from PIL import Image
from src.vit import CustomViT
from src.explainers import ExplainerFactory
from src.data_models import ImageMetrics, DatasetSummary
from src.managers import (
    MetricsCollector,
    ResultsDirectoryManager,
    ImageMetricsComputer,
)


class ExplainabilityEngine:
    """
    Main facade orchestrating explainability analysis across datasets.

    This class provides a unified public API for computing explainability heatmaps
    using multiple interpretability methods (LeGrad, GMAR, Rollout, GMARv2). It
    implements the Facade pattern to simplify interaction with the complex system
    of explainers, models, and metric collectors.

    The engine handles the complete workflow:
    1. Load and preprocess images
    2. Forward pass through Vision Transformer
    3. Compute explainability heatmaps using selected method
    4. Generate overlay visualizations
    5. Compute comprehensive fidelity and mask-based metrics
    6. Aggregate results and print summaries

    Attributes:
        vit (CustomViT): Vision Transformer model wrapper for inference and processing
        results_manager (ResultsDirectoryManager): Manages output directory structure
        mean (tuple): Image normalization mean values for denormalization
        std (tuple): Image normalization std values for denormalization
        metrics_computer (ImageMetricsComputer): Computes metrics for images

    Design Pattern:
        Facade Pattern - Simplifies complex subsystem interactions by providing
        a single unified interface for model initialization, dataset processing,
        and results management.

    Example:
        >>> engine = ExplainabilityEngine(model_name="checkpoints/vit_large/best/")
        >>> methods = engine.get_available_methods()
        >>> summary = engine.process_dataset(
        ...     dataset_tag="local",
        ...     image_files=image_paths,
        ...     method="gmar",
        ...     mask_dir=Path("./masks")
        ... )
    """

    def __init__(self, model_name: str, results_root: Path = Path("./results")):
        """
        Initialize the explainability engine with a Vision Transformer model.

        Sets up the model, preprocessor, image normalization parameters, and
        results directory manager. The engine becomes ready to process datasets
        immediately after initialization.

        Args:
            model_name (str): Path to the model checkpoint directory. Should contain
                config.json, model.safetensors, and preprocessing configuration.
                Example: "checkpoints/vit_large_tinyimagenet/best/"
            results_root (Path, optional): Root directory where results will be saved.
                Subdirectories will be created automatically. Default is "./results"
                relative to current working directory.

        Raises:
            FileNotFoundError: If the model checkpoint path doesn't exist
            ValueError: If model configuration is malformed
            RuntimeError: If model loading fails

        Example:
            >>> from pathlib import Path
            >>> engine = ExplainabilityEngine(
            ...     model_name="checkpoints/vit_large_tinyimagenet/best/",
            ...     results_root=Path("./results")
            ... )
        """
        try:
            # Validate model path
            model_path = Path(model_name)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model checkpoint directory not found: {model_name}"
                )

            # Initialize Vision Transformer model
            try:
                self.vit = CustomViT(model_name=model_name)
            except (RuntimeError, ValueError) as e:
                raise RuntimeError(
                    f"Failed to load Vision Transformer model: {e}")
            except Exception as e:
                raise RuntimeError(
                    f"Unexpected error loading ViT model: {type(e).__name__}: {e}")

            # Initialize results manager
            try:
                self.results_manager = ResultsDirectoryManager(results_root)
            except (OSError, PermissionError) as e:
                raise RuntimeError(
                    f"Failed to initialize results directory manager: {e}")
            except Exception as e:
                raise RuntimeError(
                    f"Unexpected error initializing results manager: {type(e).__name__}: {e}")

            # Extract and validate normalization parameters
            try:
                self.mean = self.vit.processor.image_mean
                self.std = self.vit.processor.image_std

                if self.mean is None or self.std is None:
                    raise ValueError(
                        "Image mean or std is None from processor")
            except AttributeError as e:
                raise ValueError(
                    f"Processor missing required attributes (mean/std): {e}")
            except Exception as e:
                raise ValueError(
                    f"Failed to extract normalization parameters: {e}")

            # Initialize metrics computer
            self.metrics_computer = ImageMetricsComputer(self.mean, self.std)

        except (FileNotFoundError, ValueError, RuntimeError) as e:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error in ExplainabilityEngine initialization: {type(e).__name__}: {e}")

    def get_available_methods(self) -> List[str]:
        """
        Get list of available explainability methods.

        Queries the ExplainerFactory registry to return all supported explanation
        methods. These are the valid choices for the 'method' parameter in
        process_dataset().

        Returns:
            List[str]: List of method names, e.g., ["legrad", "gmar", "rollout", "gmarv2"]

        Example:
            >>> engine = ExplainabilityEngine(model_name="checkpoints/vit_large/best/")
            >>> methods = engine.get_available_methods()
            >>> print(methods)
            ['legrad', 'gmar', 'rollout', 'gmarv2']
        """
        return ExplainerFactory.available_methods()

    def process_dataset(
        self,
        dataset_tag: str,
        image_files: List[Path],
        method: str,
        mask_dir: Optional[Path] = None,
        overlay_limit: Optional[int] = None,
    ) -> DatasetSummary:
        """
        Process a complete dataset with the specified explainability method.

        This is the main entry point for dataset analysis. For each image:
        1. Loads and preprocesses the image to model input format
        2. Runs forward pass to get predictions and attention weights
        3. Computes explainability heatmap using the selected method
        4. Generates and saves overlay visualization (if within overlay_limit)
        5. Computes comprehensive metrics (fidelity and mask-based)
        6. Aggregates metrics across all images

        The method uses the Strategy pattern via ExplainerFactory to support
        multiple explanation methods without code modification. Metrics are
        computed using both fidelity metrics (measures of explanation quality)
        and mask-based metrics (evaluation against ground truth if available).

        Args:
            dataset_tag (str): Identifier/tag for the dataset for output organization.
                Examples: "local", "tiny_test", "custom_dataset". Used in output
                directory paths and console messages.
            image_files (List[Path]): List of image file paths to process. Can be
                in any image format supported by PIL (jpg, png, etc).
            method (str): Name of the explanation method to use. Must be one of the
                values returned by get_available_methods(). Options:
                - "legrad": Layer-wise gradient-based attribution
                - "gmar": Gradient × Attention × Rollout
                - "rollout": Pure attention rollout mechanism
                - "gmarv2": Enhanced GMAR with additional normalization
            mask_dir (Optional[Path]): Directory containing ground truth segmentation
                masks for computing mask-based metrics (pixel_acc, IoU, AP).
                Masks should be PNG files with the same stem as images.
                If None, mask-based metrics will be skipped. Default: None
            overlay_limit (Optional[int]): Maximum number of overlay visualizations
                to save. If len(image_files) > overlay_limit, only first N overlays
                are saved to reduce disk usage. If None, all overlays are saved.
                Default: None

        Returns:
            DatasetSummary: Aggregated metrics object containing:
                - Per-image metrics: average drop/increase, logits, probabilities
                - Fidelity metrics: insertion/deletion AUC
                - Mask metrics (if masks provided): pixel accuracy, IoU, AP
                - Counts: total images, overlays saved, masked images
                - Dataset/method tags for tracking

        Raises:
            ValueError: If method name is not in available_methods()
            FileNotFoundError: If image files don't exist

        Example:
            >>> from pathlib import Path
            >>> engine = ExplainabilityEngine(model_name="checkpoints/vit_large/best/")
            >>> images = list(Path("./images").glob("*.jpg"))
            >>> summary = engine.process_dataset(
            ...     dataset_tag="my_dataset",
            ...     image_files=images,
            ...     method="gmar",
            ...     mask_dir=Path("./masks"),
            ...     overlay_limit=10
            ... )
            >>> print(f"Processed {summary.num_images} images")
            >>> print(f"Mean Insertion AUC: {summary.mean_insertion_auc:.3f}")
        """
        # Validate inputs
        if not image_files:
            print(
                f"[WARN] No image files provided for dataset '{dataset_tag}'")
            return DatasetSummary(
                dataset=dataset_tag, method=method, num_images=0,
                num_overlays_saved=0, mean_avgdrop=None, mean_avg_gain=None,
                mean_insertion_auc=None, mean_deletion_auc=None,
                num_masked_images=0, avg_increase_ratio_pct=0.0,
                count_avg_gain=0, mean_pixelacc=None, mean_iou=None, mean_ap=None,
                count_nonmask=0
            )

        # Ensure method directory exists
        try:
            method_dir = self.results_manager.ensure_method_dir(
                dataset_tag, method)
        except (OSError, PermissionError) as e:
            print(
                f"[ERROR] Failed to create output directory for method '{method}': {e}")
            return DatasetSummary(
                dataset=dataset_tag, method=method, num_images=0,
                num_overlays_saved=0, mean_avgdrop=None, mean_avg_gain=None,
                mean_insertion_auc=None, mean_deletion_auc=None,
                num_masked_images=0, avg_increase_ratio_pct=0.0,
                count_avg_gain=0, mean_pixelacc=None, mean_iou=None, mean_ap=None,
                count_nonmask=0
            )
        except Exception as e:
            print(
                f"[ERROR] Unexpected error creating results directory: {type(e).__name__}: {e}")
            return DatasetSummary(
                dataset=dataset_tag, method=method, num_images=0,
                num_overlays_saved=0, mean_avgdrop=None, mean_avg_gain=None,
                mean_insertion_auc=None, mean_deletion_auc=None,
                num_masked_images=0, avg_increase_ratio_pct=0.0,
                count_avg_gain=0, mean_pixelacc=None, mean_iou=None, mean_ap=None,
                count_nonmask=0
            )

        # Create explainer strategy
        try:
            explainer = ExplainerFactory.create(method)
            if explainer is None:
                raise ValueError(
                    f"ExplainerFactory returned None for method '{method}'")
        except ValueError as e:
            print(f"[ERROR] Invalid explanation method '{method}': {e}")
            return DatasetSummary(
                dataset=dataset_tag, method=method, num_images=0,
                num_overlays_saved=0, mean_avgdrop=None, mean_avg_gain=None,
                mean_insertion_auc=None, mean_deletion_auc=None,
                num_masked_images=0, avg_increase_ratio_pct=0.0,
                count_avg_gain=0, mean_pixelacc=None, mean_iou=None, mean_ap=None,
                count_nonmask=0
            )
        except Exception as e:
            print(
                f"[ERROR] Failed to create explainer: {type(e).__name__}: {e}")
            return DatasetSummary(
                dataset=dataset_tag, method=method, num_images=0,
                num_overlays_saved=0, mean_avgdrop=None, mean_avg_gain=None,
                mean_insertion_auc=None, mean_deletion_auc=None,
                num_masked_images=0, avg_increase_ratio_pct=0.0,
                count_avg_gain=0, mean_pixelacc=None, mean_iou=None, mean_ap=None,
                count_nonmask=0
            )

        metrics_collector = MetricsCollector()

        print(
            f"\n=== [{dataset_tag}] Processing {len(image_files)} image(s) with method '{method}' ===")

        saved_count = 0
        for idx, image_file in enumerate(image_files, 1):
            try:
                # Load and preprocess image
                try:
                    img = Image.open(image_file).convert('RGB')
                except (FileNotFoundError, IOError) as e:
                    print(
                        f"[WARN] Could not open image {image_file.name}: {e}")
                    continue
                except Exception as e:
                    print(
                        f"[WARN] Unexpected error opening image {image_file.name}: {type(e).__name__}: {e}")
                    continue

                try:
                    img_tensor = self.vit.preprocess(img)
                except (ValueError, RuntimeError) as e:
                    print(f"[WARN] Failed to preprocess image {image_file.name}: {e}")
                    continue
                except Exception as e:
                    print(f"[WARN] Unexpected error preprocessing {image_file.name}: {type(e).__name__}: {e}")
                    continue

                try:
                    logits, pred_idx, class_name, attn_weights = self.vit.forward_with_custom_attention(img_tensor)
                except RuntimeError as e:
                    print(f"[WARN] Model forward pass failed for {image_file.name}: {e}")
                    continue
                except Exception as e:
                    print(f"[WARN] Unexpected error in forward pass for {image_file.name}: {type(e).__name__}: {e}")
                    continue

                print(f"[{dataset_tag}] {idx}/{len(image_files)}: {image_file.name} → Pred: {class_name} (idx={pred_idx})")

                # Decide whether to save overlay
                should_save_overlay = True
                if overlay_limit is not None and saved_count >= overlay_limit:
                    should_save_overlay = False

                # Compute heatmap
                try:
                    final_map = explainer.compute_heatmap(logits, pred_idx, attn_weights, self.vit.model)
                except RuntimeError as e:
                    print(f"[WARN] Failed to compute heatmap for {image_file.name}: {e}")
                    continue
                except Exception as e:
                    print(f"[WARN] Unexpected error computing heatmap for {image_file.name}: {type(e).__name__}: {e}")
                    continue

                # Save overlay if requested
                if should_save_overlay:
                    try:
                        save_path = method_dir / image_file.name
                        explainer.save_overlay(final_map, img, save_path)
                        saved_count += 1
                    except (OSError, PermissionError) as e:
                        print(f"[WARN] Failed to save overlay for {image_file.name}: {e}")
                    except Exception as e:
                        print(f"[WARN] Unexpected error saving overlay for {image_file.name}: {type(e).__name__}: {e}")

                # Compute metrics
                try:
                    image_metrics = self.metrics_computer.compute_all_metrics(
                        vit_model=self.vit.model,
                        img_tensor=img_tensor,
                        pred_idx=pred_idx,
                        final_map=final_map,
                        mask_dir=mask_dir,
                        mask_file_stem=image_file.stem,
                    )
                except RuntimeError as e:
                    print(
                        f"[WARN] Failed to compute metrics for {image_file.name}: {e}")
                    continue
                except Exception as e:
                    print(f"[WARN] Unexpected error computing metrics for {image_file.name}: {type(e).__name__}: {e}")
                    continue

                self._print_image_metrics(
                    image_metrics, dataset_tag, idx, len(image_files))
                metrics_collector.add_metrics(image_metrics)

            except KeyboardInterrupt:
                print(f"\n[INFO] Processing interrupted at image {idx}/{len(image_files)}")
                break
            except Exception as e:
                print(f"[ERROR] Unexpected error processing {image_file.name}: {type(e).__name__}: {e}")
                continue

        try:
            summary = metrics_collector.get_summary(
                dataset_tag, method, saved_count)
        except Exception as e:
            print(
                f"[ERROR] Failed to generate summary for dataset '{dataset_tag}': {e}")
            summary = DatasetSummary(
                dataset=dataset_tag, method=method, num_images=len(
                    image_files),
                num_overlays_saved=saved_count, mean_avgdrop=None, mean_avg_gain=None,
                mean_insertion_auc=None, mean_deletion_auc=None,
                num_masked_images=0, avg_increase_ratio_pct=0.0,
                count_avg_gain=0, mean_pixelacc=None, mean_iou=None, mean_ap=None,
                count_nonmask=0
            )

        try:
            self._print_summary(summary)
        except Exception as e:
            print(f"[ERROR] Failed to print summary: {e}")

        return summary

    @staticmethod
    def _print_image_metrics(metrics: ImageMetrics, dataset_tag: str, idx: int, total: int):
        """
        Print formatted metrics for a single image to console.

        This private helper prints per-image metrics in a readable tabular format,
        showing all fidelity and mask-based metrics computed for the image.

        Args:
            metrics (ImageMetrics): Pre-computed metrics for the image
            dataset_tag (str): Dataset identifier
            idx (int): Current image number (1-indexed)
            total (int): Total number of images being processed
        """
        print(f"  AvgDrop: {metrics.avg_drop:.2f}% | AvgIncrease: {metrics.avg_increase} | "
              f"logit(full): {metrics.logit_full:.3f} | logit(masked): {metrics.logit_masked:.3f}")
        print(f"  Average Gain: {metrics.average_gain:.2f}% | p(full): {metrics.prob_full:.4f} | "
              f"p(masked): {metrics.prob_masked:.4f}")
        print(f"  Insertion AUC (↑): {metrics.insertion_auc:.3f} | Deletion AUC (↓): {metrics.deletion_auc:.3f}")

        if metrics.pixel_acc is not None:
            print(f"  PixelAcc: {metrics.pixel_acc:.4f} | IoU: {metrics.iou:.4f} | AP: {metrics.ap:.4f}")
        else:
            print("  No GT mask → skip PixelAcc/IoU/AP")

    @staticmethod
    def _print_summary(summary: DatasetSummary):
        """
        Print formatted dataset-level summary metrics to console.

        Args:
            summary (DatasetSummary): Aggregated dataset statistics
        """
        print(f"\n=== [{summary.dataset}] Summary ({summary.method}) ===")
        print(f"Images processed               : {summary.num_images}")
        print(f"Overlays saved                 : {summary.num_overlays_saved}")

        if summary.mean_avgdrop is not None:
            print(f"Mean AvgDrop               : {summary.mean_avgdrop:.2f}%")
            print(f"AvgIncrease (% of images)  : {summary.avg_increase_ratio_pct:.2f}%")
            print(f"Mean Insertion AUC         : {summary.mean_insertion_auc:.3f}")
            print(f"Mean Deletion AUC          : {summary.mean_deletion_auc:.3f}")

        if summary.mean_avg_gain is not None:
            print(f"Mean Average Gain          : {summary.mean_avg_gain:.2f}%")

        if summary.num_masked_images > 0:
            print(f"Masked images evaluated    : {summary.num_masked_images}")
            print(f"Mean PixelAcc              : {summary.mean_pixelacc:.4f}")
            print(f"Mean IoU                   : {summary.mean_iou:.4f}")
            print(f"Mean AP                    : {summary.mean_ap:.4f}")
        else:
            print("No mask-based metrics computed (no masks found).")
