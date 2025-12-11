"""Pixel-level Metrics for Mask-based Evaluation

This module provides functions to compute standard computer vision metrics
for comparing predicted explanations (heatmaps) against ground truth masks:
pixel accuracy, IoU, and average precision.
"""

from typing import Union
import torch
from sklearn.metrics import average_precision_score


__all__ = ["compute_pixel_accuracy", "compute_iou", "compute_ap"]


def compute_pixel_accuracy(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor
) -> float:
    """Compute pixel-wise accuracy between two binary masks.

    Parameters
    ----------
    pred_mask
        Predicted binary mask (torch.Tensor [H, W] with values 0/1).
    gt_mask
        Ground truth binary mask (torch.Tensor [H, W] with values 0/1).

    Returns
    -------
    float
        Pixel accuracy in range [0, 1].
    """
    correct = (pred_mask == gt_mask).sum().item()
    total = gt_mask.numel()
    return correct / total


def compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """Compute Intersection over Union (IoU) between two binary masks.

    Parameters
    ----------
    pred_mask
        Predicted binary mask (torch.Tensor [H, W] with values 0/1).
    gt_mask
        Ground truth binary mask (torch.Tensor [H, W] with values 0/1).

    Returns
    -------
    float
        IoU score in range [0, 1]; returns 0.0 if union is empty.
    """
    intersection = ((pred_mask == 1) & (gt_mask == 1)).sum().item()
    union = ((pred_mask == 1) | (gt_mask == 1)).sum().item()
    return intersection / union if union != 0 else 0.0


def compute_ap(heatmap: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """Compute Average Precision using raw heatmap vs. ground truth mask.

    Flattens both the heatmap and mask, then uses scikit-learn's
    ``average_precision_score`` to compute AP.

    Parameters
    ----------
    heatmap
        Raw normalized heatmap (torch.Tensor [H, W] with values in [0, 1]).
    gt_mask
        Ground truth binary mask (torch.Tensor [H, W] with values 0/1).

    Returns
    -------
    float
        Average Precision score.
    """
    y_true = gt_mask.flatten().cpu().numpy()
    y_score = heatmap.flatten().cpu().numpy()
    return average_precision_score(y_true, y_score)
