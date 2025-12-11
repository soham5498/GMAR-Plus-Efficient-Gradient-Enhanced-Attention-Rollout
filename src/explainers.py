"""
Explainability Strategy Pattern - Abstract and Concrete Implementations.

This module provides the Strategy pattern implementation for different explainability
methods including LeGrad, GMAR, Rollout, and GMARv2. Each strategy encapsulates
a specific explanation algorithm with consistent interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
import torch
from PIL import Image
from src.legrad import LeGradExplainer
from src.rollout import AttentionRollout
from src.gmar import GMAR
from src.gmarv2 import GMARv2


class ExplainerStrategy(ABC):
    """
    Abstract base class defining the interface for all explainability strategy implementations.

    This class enforces a contract that all concrete explanation methods must implement
    compute_heatmap() and save_overlay() methods, allowing different algorithms to be
    used interchangeably through the strategy pattern.

    Methods:
        compute_heatmap: Generate explanation heatmap for model predictions
        save_overlay: Save visual overlay of explanation on the original image
    """

    @abstractmethod
    def compute_heatmap(
        self,
        logits: torch.Tensor,
        pred_idx: int,
        attn_weights: torch.Tensor,
        vit_model
    ) -> torch.Tensor:
        """
        Compute explanation heatmap for given inputs.

        Args:
            logits (torch.Tensor): Model output logits for the input
            pred_idx (int): Index of the predicted class
            attn_weights (torch.Tensor): Attention weights from model
            vit_model: Vision Transformer model instance

        Returns:
            torch.Tensor: 2D heatmap of shape (H, W) with explanation scores
        """
        pass

    @abstractmethod
    def save_overlay(
        self,
        final_map: torch.Tensor,
        img: Image.Image,
        save_path: Path
    ) -> None:
        """
        Save visual overlay of heatmap on original image.

        Args:
            final_map (torch.Tensor): 2D explanation heatmap
            img (Image.Image): Original RGB image
            save_path (Path): File path where overlay should be saved
        """
        pass


class LeGradStrategy(ExplainerStrategy):
    """
    LeGrad (Layer-wise Gradient) explanation strategy.

    Uses gradient-based approach by aggregating gradients across layers to generate
    visual explanations. Converts 1D gradient heatmap to 2D for visualization.

    Attributes:
        explainer (LeGradExplainer): Instance of LeGrad explainer

    References:
        LeGrad: Layer-wise Relevance Propagation using gradient information
    """

    def __init__(self) -> None:
        """Initialize LeGrad strategy by creating explainer instance."""
        self.explainer = None

    def compute_heatmap(
        self,
        logits: torch.Tensor,
        pred_idx: int,
        attn_weights: torch.Tensor,
        vit_model
    ) -> torch.Tensor:
        """
        Compute explanation using layer-wise gradient aggregation.

        Args:
            logits (torch.Tensor): Model output logits
            pred_idx (int): Index of predicted class
            attn_weights (torch.Tensor): Attention weights from transformer layers
            vit_model: Vision Transformer model instance

        Returns:
            torch.Tensor: 2D explanation heatmap of shape (H, W)
        """
        self.explainer = LeGradExplainer(attn_weights, logits, pred_idx)
        self.explainer.compute_layer_maps()
        heatmap_1d = self.explainer.merge_heatmap()
        side_len = int(heatmap_1d.shape[0] ** 0.5)
        return torch.tensor(heatmap_1d.reshape(side_len, side_len), dtype=torch.float32)

    def save_overlay(
        self,
        final_map: torch.Tensor,
        img: Image.Image,
        save_path: Path
    ) -> None:
        """
        Save visual overlay of LeGrad heatmap on image.

        Args:
            final_map (torch.Tensor): 2D explanation heatmap
            img (Image.Image): Original image
            save_path (Path): Output file path for overlay
        """
        if self.explainer is not None:
            heatmap_1d = final_map.numpy().reshape(-1)
            self.explainer.save_overlay(
                img, heatmap_1d, original_image_path=save_path)


class GMARStrategy(ExplainerStrategy):
    """
    GMAR (Gradient × Attention × Rollout) explanation strategy.

    Combines gradient information, attention weights, and rollout mechanism
    to generate comprehensive visual explanations.

    Attributes:
        explainer (GMAR): GMAR explainer instance

    Configuration:
        alpha (float): Weighting parameter (default: 1.0)
        norm_type (str): Normalization method (default: 'l2')
    """

    def __init__(self) -> None:
        """Initialize GMAR strategy with default parameters."""
        self.explainer = GMAR(alpha=1.0, norm_type='l2')

    def compute_heatmap(
        self,
        logits: torch.Tensor,
        pred_idx: int,
        attn_weights: torch.Tensor,
        vit_model
    ) -> torch.Tensor:
        """
        Compute GMAR explanation combining gradient, attention, and rollout.

        Args:
            logits (torch.Tensor): Model output logits
            pred_idx (int): Index of predicted class
            attn_weights (torch.Tensor): Attention weights from all layers
            vit_model: Vision Transformer model instance

        Returns:
            torch.Tensor: 2D explanation heatmap
        """
        return self.explainer.compute(logits, pred_idx, attn_weights, model=vit_model).float()

    def save_overlay(
        self,
        final_map: torch.Tensor,
        img: Image.Image,
        save_path: Path
    ) -> None:
        """
        Save GMAR explanation overlay on image.

        Args:
            final_map (torch.Tensor): Computed explanation heatmap
            img (Image.Image): Original image
            save_path (Path): Output file path
        """
        self.explainer.save_overlay(
            final_map, img, original_image_path=save_path)


class RolloutStrategy(ExplainerStrategy):
    """
    Attention Rollout explanation strategy.

    Uses attention flow across transformer layers to generate explanations.
    Rolls out attention from output to input layer to identify important regions.

    Attributes:
        explainer (AttentionRollout): Attention rollout explainer instance

    Configuration:
        add_residual (bool): Whether to add residual connections (default: True)
    """

    def __init__(self) -> None:
        """Initialize Attention Rollout strategy."""
        self.explainer = AttentionRollout(add_residual=True)

    def compute_heatmap(
        self,
        logits: torch.Tensor,
        pred_idx: int,
        attn_weights: torch.Tensor,
        vit_model
    ) -> torch.Tensor:
        """
        Compute explanation using attention rollout mechanism.

        Args:
            logits (torch.Tensor): Model output logits
            pred_idx (int): Index of predicted class
            attn_weights (torch.Tensor): Attention weights from transformer
            vit_model: Vision Transformer model instance

        Returns:
            torch.Tensor: 2D rollout-based explanation heatmap
        """
        return self.explainer.compute_rollout(attn_weights).float()

    def save_overlay(
        self,
        final_map: torch.Tensor,
        img: Image.Image,
        save_path: Path
    ) -> None:
        """
        Save attention rollout explanation overlay.

        Args:
            final_map (torch.Tensor): Rollout explanation heatmap
            img (Image.Image): Original image
            save_path (Path): Output file path
        """
        self.explainer.save_overlay(
            final_map, img, original_image_path=save_path)


class GMARv2Strategy(ExplainerStrategy):
    """
    GMARv2 (Enhanced GMAR) explanation strategy.

    Improved version of GMAR with enhanced gradient-attention integration.
    Provides better fusion of gradient and attention information.

    Attributes:
        explainer (GMARv2): GMARv2 explainer instance

    Configuration:
        alpha (float): Weighting parameter (default: 1.0)
        norm_type (str): Normalization method (default: 'l2')
    """

    def __init__(self) -> None:
        """Initialize GMARv2 strategy with default parameters."""
        self.explainer = GMARv2(alpha=1.0, norm_type='l2')

    def compute_heatmap(
        self,
        logits: torch.Tensor,
        pred_idx: int,
        attn_weights: torch.Tensor,
        vit_model
    ) -> torch.Tensor:
        """
        Compute enhanced GMAR explanation.

        Args:
            logits (torch.Tensor): Model output logits
            pred_idx (int): Index of predicted class
            attn_weights (torch.Tensor): Attention weights from model
            vit_model: Vision Transformer model instance

        Returns:
            torch.Tensor: 2D enhanced explanation heatmap
        """
        return self.explainer.compute(logits, pred_idx, attn_weights, model=vit_model).float()

    def save_overlay(
        self,
        final_map: torch.Tensor,
        img: Image.Image,
        save_path: Path
    ) -> None:
        """
        Save GMARv2 explanation overlay.

        Args:
            final_map (torch.Tensor): GMARv2 explanation heatmap
            img (Image.Image): Original image
            save_path (Path): Output file path
        """
        self.explainer.save_overlay(
            final_map, img, original_image_path=save_path)


class ExplainerFactory:
    """
    Factory for creating explainer strategy instances.

    Uses the Factory design pattern to provide centralized creation logic for
    different explanation methods. Maintains a registry of available strategies
    and enables easy registration of new methods without modifying existing code.

    Class Attributes:
        _strategies (dict): Mapping of method names to strategy classes
    """

    _strategies = {
        "legrad": LeGradStrategy,
        "gmar": GMARStrategy,
        "rollout": RolloutStrategy,
        "gmarv2": GMARv2Strategy,
    }

    @classmethod
    def create(cls, choice: str, *args, **kwargs) -> ExplainerStrategy:
        """
        Create an explainer strategy instance by method name.

        Args:
            choice (str): Name of the explanation method to create.
                         Valid options: 'legrad', 'gmar', 'rollout', 'gmarv2'
            *args: Positional arguments passed to strategy constructor
            **kwargs: Keyword arguments passed to strategy constructor

        Returns: xplainerStrategy: Instantiated strategy object

        Raises: ValueError: If choice is not a registered strategy method
        """
        if choice not in cls._strategies:
            raise ValueError(
                f"Unknown choice: {choice}. "
                f"Available: {', '.join(cls._strategies.keys())}"
            )
        return cls._strategies[choice](*args, **kwargs)

    @classmethod
    def available_methods(cls) -> List[str]:
        """
        Return list of available explainer method names.

        Returns: List[str]: List of registered method names
        """
        return list(cls._strategies.keys())
