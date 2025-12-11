# ViT.py
"""CustomViT: ViT Model Wrapper with Custom Attention Extraction

This module provides a wrapper around the HuggingFace ViT model that:
- Loads fine-tuned checkpoints or hub model IDs
- Matches the image processor from the checkpoint
- Returns logits, predictions, class names, and per-layer attention maps
- Enables custom forward passes for explainability methods
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    AutoConfig,
)


__all__ = ["CustomViT"]


class CustomViT:
    """Wrapper around HuggingFace ViT for attention-based explanations.

    Loads a fine-tuned ViT checkpoint, manages preprocessing, and enables
    custom forward passes that capture per-layer attention matrices.

    Parameters
    ----------
    model_name : str, optional
        Path to fine-tuned checkpoint directory or HF hub ID
        (default: "checkpoints/vit_large_tinyimagenet/best/").
    device : str, optional
        Compute device ('cuda', 'cpu', etc.); auto-selected if None.
    ensure_size : Tuple[int, int], optional
        Target image size (H, W) for preprocessing (default: (224, 224)).
    """

    def __init__(
        self,
        model_name: str = "checkpoints/vit_large_tinyimagenet/best/",
        device: Optional[str] = None,
        ensure_size: Tuple[int, int] = (224, 224),
    ) -> None:
        """Initialize CustomViT with model and processor."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTForImageClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        try:
            self.processor = ViTImageProcessor.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224")

        preproc_path = Path(model_name) / "preproc.json"
        if preproc_path.exists():
            try:
                pp = json.loads(preproc_path.read_text())
                if "image_mean" in pp: self.processor.image_mean = pp["image_mean"]
                if "image_std"  in pp: self.processor.image_std  = pp["image_std"]
            except Exception:
                pass

        
        if ensure_size is not None:
            H, W = ensure_size
            try:
                self.processor.size = {"height": H, "width": W}
            except Exception:
                pass

        cfg = self.model.config
        id2label = cfg.id2label or {}
        try:
            tmp = {int(k): v for k, v in id2label.items()}
        except Exception:
            tmp = id2label
        self.imagenet_classes = [tmp[i] for i in range(cfg.num_labels)]

    def preprocess(self, img) -> torch.Tensor:
        """Preprocess a PIL image for ViT.

        Parameters
        ----------
        img
            PIL Image to preprocess.

        Returns
        -------
        torch.Tensor
            Normalized image tensor of shape [1, 3, H, W].
        """
        inputs = self.processor(images=img, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def forward_with_custom_attention(
        self, img_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, int, str, List[torch.Tensor]]:
        """Manual forward pass extracting per-layer attention maps.

        Implements the ViT forward pass with manual attention computation
        to capture and return attention matrices from each block.

        Parameters
        ----------
        img_tensor
            Preprocessed image tensor of shape [1, 3, H, W].

        Returns
        -------
        Tuple[torch.Tensor, int, str, List[torch.Tensor]]
            (logits, predicted_class_idx, class_name, per_layer_attention_matrices)
        """
        attn_weights: List[torch.Tensor] = []
        x = self.model.vit.embeddings.patch_embeddings(img_tensor)
        cls_token = self.model.vit.embeddings.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.model.vit.embeddings.position_embeddings
        x = self.model.vit.embeddings.dropout(x)

        for blk in self.model.vit.encoder.layer:
            B, N, C = x.shape
            norm_x = blk.layernorm_before(x)

            q = blk.attention.attention.query(norm_x)
            k = blk.attention.attention.key(norm_x)
            v = blk.attention.attention.value(norm_x)

            num_heads = blk.attention.attention.num_attention_heads
            head_dim = C // num_heads

            q = q.view(B, N, num_heads, head_dim).transpose(1, 2)
            k = k.view(B, N, num_heads, head_dim).transpose(1, 2)
            v = v.view(B, N, num_heads, head_dim).transpose(1, 2)

            attn = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = attn.softmax(dim=-1)
            attn.retain_grad()

            context = attn @ v
            context = context.transpose(1, 2).reshape(B, N, C)
            attn_out = blk.attention.output.dense(context)
            x = x + attn_out

            mlp_in = blk.layernorm_after(x)
            mlp_hidden = blk.intermediate.dense(mlp_in)
            mlp_hidden = torch.nn.functional.gelu(mlp_hidden)
            mlp_out = blk.output.dense(mlp_hidden)
            x = x + mlp_out

            attn_weights.append(attn)

        x = self.model.vit.layernorm(x)
        cls_embedding = x[:, 0]
        logits = self.model.classifier(cls_embedding)

        predicted_class = logits.argmax(dim=-1).item()
        class_name = self.imagenet_classes[predicted_class]

        return logits, predicted_class, class_name, attn_weights
