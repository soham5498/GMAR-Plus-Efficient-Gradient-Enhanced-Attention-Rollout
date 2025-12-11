"""
Data Models for Metrics and Summaries.

This module provides type-safe dataclasses for storing and transferring
metrics and summary statistics throughout the explainability pipeline.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ImageMetrics:
    """
    Type-safe container for metrics computed on a single image.

    Stores both fidelity metrics (average drop, AUC scores) and optional
    mask-based metrics (pixel accuracy, IoU, AP) for a single image.
    Enables structured data passing and type checking throughout the pipeline.

    Attributes:
        avg_drop (float): Average drop in confidence when masking important regions (↓ better)
        avg_increase (float): Average increase in confidence (↑ better)
        logit_full (float): Model logit on full, unmasked image
        logit_masked (float): Model logit on partially masked image
        average_gain (float): Average gain in probability (↑ better)
        prob_full (float): Model probability on full image
        prob_masked (float): Model probability on masked image
        insertion_auc (float): Area under insertion curve (↑ better)
        deletion_auc (float): Area under deletion curve (↓ better)
        pixel_acc (Optional[float]): Pixel-level accuracy vs ground truth (if mask available)
        iou (Optional[float]): Intersection over Union vs ground truth (if mask available)
        ap (Optional[float]): Average Precision vs ground truth (if mask available)
    """
    avg_drop: float
    avg_increase: float
    logit_full: float
    logit_masked: float
    average_gain: float
    prob_full: float
    prob_masked: float
    insertion_auc: float
    deletion_auc: float
    pixel_acc: Optional[float] = None
    iou: Optional[float] = None
    ap: Optional[float] = None


@dataclass
class DatasetSummary:
    """
    Type-safe container for aggregated dataset-level statistics.

    Aggregates metrics from all processed images in a dataset, storing
    mean values and counts for comprehensive statistical summary.
    Provides high-level view of explanation quality across entire dataset.

    Attributes:
        dataset (str): Name/tag of the dataset (e.g., 'local', 'tiny_test')
        method (str): Name of the explanation method used (e.g., 'gmar', 'legrad')
        num_images (int): Total number of images processed
        num_overlays_saved (int): Number of overlay images saved to disk
        count_nonmask (int): Number of images with fidelity metrics
        mean_avgdrop (Optional[float]): Mean average drop across images
        avg_increase_ratio_pct (Optional[float]): Mean average increase as percentage
        mean_avg_gain (Optional[float]): Mean average gain across images
        count_avg_gain (int): Count of images with average gain computed
        mean_insertion_auc (Optional[float]): Mean insertion AUC score
        mean_deletion_auc (Optional[float]): Mean deletion AUC score
        num_masked_images (int): Number of images with ground truth masks
        mean_pixelacc (Optional[float]): Mean pixel accuracy (if masks available)
        mean_iou (Optional[float]): Mean IoU score (if masks available)
        mean_ap (Optional[float]): Mean Average Precision (if masks available)
    """
    dataset: str
    method: str
    num_images: int
    num_overlays_saved: int
    count_nonmask: int
    mean_avgdrop: Optional[float]
    avg_increase_ratio_pct: Optional[float]
    mean_avg_gain: Optional[float]
    count_avg_gain: int
    mean_insertion_auc: Optional[float]
    mean_deletion_auc: Optional[float]
    num_masked_images: int
    mean_pixelacc: Optional[float]
    mean_iou: Optional[float]
    mean_ap: Optional[float]
