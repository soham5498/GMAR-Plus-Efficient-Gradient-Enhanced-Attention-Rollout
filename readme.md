# GMAR++: Efficient Gradient-Enhanced Attention Rollout

<div align="center">

![GMAR++ Architecture](./images/gmar_architecture.png)

**A cutting-edge explainability framework for Vision Transformers using gradient-enhanced attention mechanisms**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)]()

</div>

---

## 📋 Abstract

GMAR++ is an advanced explainability framework designed to enhance the interpretability of Vision Transformer (ViT) models through efficient gradient-enhanced attention rollout mechanisms. The project implements multiple state-of-the-art explainability methods—including Rollout, GMAR (Gradient-weighted Multi-head Attention Rollout), LeGrad, and GMAR++ (v2)—providing comprehensive visual explanations for transformer-based vision models.

### Key Innovations

- **Gradient-Enhanced Attention Weighting**: Leverages gradient information to weight attention heads based on their importance for the predicted class
- **Efficient Rollout Mechanism**: Implements residual-aware attention rollout to capture interactions across transformer layers
- **Multi-Head Aggregation**: Intelligently combines multi-head attention using learned importance weights
- **Flexible Architecture**: Supports both L1 and L2 normalization for gradient computation
- **Comprehensive Evaluation**: Includes fidelity metrics and faithfulness analysis

The framework achieves superior explanation quality compared to traditional attention rollout methods by incorporating gradient information, enabling more accurate attribution of model predictions to input features.

---

## 🎯 Features

### Explainability Methods

| Method | Description | Characteristics |
|--------|-------------|-----------------|
| **Rollout** | Basic attention rollout without gradient enhancement | Fast, baseline method |
| **GMAR** | Gradient-weighted Multi-head Attention Rollout | Gradient-aware head weighting |
| **LeGrad** | Layer-wise Gradient Attention Rollout | Per-layer gradient weighting |
| **GMAR++** | Enhanced GMAR with improved aggregation | Best explanation quality |

### Capabilities

✅ **Vision Transformer Support** - Optimized for ViT-Large and other transformer architectures  
✅ **Multi-Dataset Processing** - Handles local images and TinyImageNet  
✅ **Heatmap Generation** - Produces high-quality attention heatmaps  
✅ **Visual Overlays** - Generates overlaid visualizations on original images  
✅ **Fidelity Metrics** - Computes explanation faithfulness metrics  
✅ **Batch Processing** - Efficient processing of multiple samples  

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Input Image                        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   Patch Embedding      │
        └─────────────┬──────────┘
                      │
                      ▼
        ┌─────────────────────────────────┐
        │ Vision Transformer Backbone     │
        │ ┌────────────────────────────┐  │
        │ │  Multi-Head Attention (L)  │  │
        │ │  with Gradient Tracking    │  │
        │ └─────────────┬──────────────┘  │
        │               │ (Save Weights)   │
        │               ▼                   │
        │ ┌────────────────────────────┐  │
        │ │   MLP Layers               │  │
        │ └────────────────────────────┘  │
        └─────────────┬──────────────────┘
                      │
                      ▼
        ┌────────────────────────┐
        │   Classification Head  │
        │   (Logits Output)      │
        └─────────────┬──────────┘
                      │
                      ▼
        ┌────────────────────────────────┐
        │   Explainability Engine        │
        │   ┌──────────────────────────┐ │
        │   │ Gradient Backprop        │ │
        │   │ (to Attention Weights)   │ │
        │   └──────────────────────────┘ │
        │   ┌──────────────────────────┐ │
        │   │ Head Importance (∇²/∇)  │ │
        │   └──────────────────────────┘ │
        │   ┌──────────────────────────┐ │
        │   │ Residual-Aware Rollout   │ │
        │   └──────────────────────────┘ │
        └─────────────┬──────────────────┘
                      │
        ┌─────────────┴──────────────────┬──────────────────┐
        │                                │                  │
        ▼                                ▼                  ▼
    ┌────────────┐              ┌────────────┐      ┌────────────┐
    │ Heatmap    │              │ Metrics    │      │ Overlay    │
    │ Generation │              │ Computation│      │ Rendering  │
    └────────────┘              └────────────┘      └────────────┘
```

---

## 📊 Visual Results

### Example Explanations Across Methods

![Comparison Results](./images/referenec_images/comparison_heatmaps.png)

*Above: Attention heatmaps generated by different methods (Rollout, GMAR-L1, GMAR-L2, LeGrad, GMAR++) for various object classes*

The visualization demonstrates:
- **Rollout**: Broad, diffuse attention patterns
- **GMAR-L1/L2**: Focused attention with gradient weighting
- **LeGrad**: Layer-wise gradient-based explanations
- **GMAR++**: Sharp, highly localized attribution maps

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.8 or higher
- **PyTorch**: 2.0 or higher
- **CUDA** (optional): For GPU acceleration

### Installation

#### macOS

```bash
# Clone the repository
git clone https://github.com/soham5498/GMAR-Plus-Efficient-Gradient-Enhanced-Attention-Rollout.git
cd "GMAR++"

# Create a Python virtual environment
python3 -m venv env
source env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirement.txt

# For M1/M2 Mac with GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Linux (Ubuntu/Debian)

```bash
# Clone the repository
git clone https://github.com/soham5498/GMAR-Plus-Efficient-Gradient-Enhanced-Attention-Rollout.git
cd "GMAR++"

# Create a Python virtual environment
python3 -m venv env
source env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirement.txt

# For GPU support (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Windows

```powershell
# Clone the repository
git clone https://github.com/soham5498/GMAR-Plus-Efficient-Gradient-Enhanced-Attention-Rollout.git
cd "GMAR++"

# Create a Python virtual environment
python -m venv env
env\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirement.txt

# For GPU support (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Basic Usage

```bash
# Run the main explainability analysis
python main.py

# Follow the interactive prompts:
# 1. Select an explainability method (rollout/gmar/legrad/gmar++)
# 2. Process images from the local dataset
# 3. Optionally download and process TinyImageNet validation set
```

**Example Interaction:**
```
Which method? (rollout/gmar/legrad/gmar++): gmar

Processing local images...
✓ Processed: goldfish.jpg
✓ Processed: lion.jpg
✓ Processed: african_elephant.jpg
...

Download TinyImageNet test set? (y/n): y
Processing TinyImageNet samples...
```

---

## 📁 Project Structure

```
GMAR++/
├── README.md                 # This file
├── requirement.txt          # Python dependencies
├── main.py                  # Entry point for explainability analysis
│
├── src/                     # Core implementation modules
│   ├── explainers.py       # Strategy pattern for explanation methods
│   ├── gmar.py             # GMAR implementation
│   ├── gmarv2.py           # Enhanced GMAR++ (v2) implementation
│   ├── legrad.py           # LeGrad implementation
│   ├── rollout.py          # Basic attention rollout
│   ├── vit.py              # Vision Transformer model utilities
│   ├── engine.py           # Main explainability engine
│   ├── managers.py         # Dataset and configuration managers
│   ├── metrics.py          # Evaluation metrics
│   ├── metrics_fidelity.py # Fidelity-based metrics
│   └── data_models.py      # Data structures and models
│
├── checkpoints/            # Pre-trained model weights
│   └── vit_large_tinyimagenet/
│       ├── best/          # Best model checkpoint
│       └── final/         # Final model checkpoint
│
├── images/                 # Input images for explanation
│   └── reference_images/  # Example images for comparison
│
├── masks/                  # Attention masks storage
│
└── results/               # Output results and visualizations
    └── local/            # Results for local images
        ├── gmar/         # GMAR method results
        ├── legrad/       # LeGrad method results
        └── ...
```

---

## 🔧 Core Modules

### Explainability Engine (`src/engine.py`)

The main orchestrator that:
- Initializes the ViT model with pre-trained weights
- Manages the explanation workflow
- Coordinates between the model and explainers
- Handles batch processing

### Explainer Strategies (`src/explainers.py`)

Implements the Strategy pattern with:
- `ExplainerStrategy`: Abstract base class
- `RolloutExplainer`: Basic attention rollout
- `GMARExplainer`: Gradient-weighted rollout
- `LeGradExplainer`: Layer-wise gradient attribution
- `GMARv2Explainer`: Enhanced GMAR with improved aggregation

### Data Management (`src/managers.py`)

Handles:
- Dataset loading and preprocessing
- Image normalization (ImageNet stats)
- Batch creation and iteration
- Results organization

---

## 📊 Methods Overview

### 1. **Rollout**
Traditional attention rollout that aggregates attention across layers without gradient weighting.

```python
from src.explainers import ExplainerFactory

explainer = ExplainerFactory.create('rollout')
heatmap = explainer.compute_heatmap(logits, pred_idx, attn_weights, model)
```

### 2. **GMAR** (Gradient-weighted Multi-head Attention Rollout)
Weights attention heads by their gradient magnitude with respect to the predicted class.

- **L2 Norm**: `head_importance = sqrt(sum(∇²))`
- **L1 Norm**: `head_importance = sum(|∇|)`

### 3. **LeGrad** (Layer-wise Gradient Attention)
Applies gradient weighting at each layer before rollout.

### 4. **GMAR++** (Enhanced GMAR)
Advanced version with improved multi-head aggregation and residual scaling.

---

## 💻 Usage Examples

### Example 1: Explain a Single Image

```python
from src.engine import ExplainabilityEngine
from src.explainers import ExplainerFactory
from pathlib import Path

# Initialize engine
engine = ExplainabilityEngine('gmar')

# Load and explain an image
image_path = Path('images/goldfish.jpg')
results = engine.explain(image_path)

print(f"Predicted class: {results['class_name']}")
print(f"Confidence: {results['confidence']:.2%}")
print(f"Heatmap shape: {results['heatmap'].shape}")
```

### Example 2: Batch Processing with Custom Method

```python
from src.managers import DatasetManager

# Load local image dataset
dataset = DatasetManager.load_local_images('images/')

# Process with GMAR++
explainer = ExplainerFactory.create('gmar++')

for image, label in dataset:
    heatmap = explainer.compute_heatmap(
        logits, pred_idx, attn_weights, model
    )
    explainer.save_overlay(image, heatmap, f'results/{label}.png')
```

### Example 3: Compare Methods

```python
methods = ['rollout', 'gmar', 'legrad', 'gmar++']

for method in methods:
    explainer = ExplainerFactory.create(method)
    heatmap = explainer.compute_heatmap(...)
    
    # Compute fidelity score
    fidelity = compute_fidelity(heatmap, model, image)
    print(f"{method}: {fidelity:.4f}")
```

---

## 📈 Performance Metrics

The framework includes several evaluation metrics:

| Metric | Description | Range |
|--------|-------------|-------|
| **Fidelity** | How well the explanation preserves model behavior | 0-1 |
| **Localization** | How localized the attribution is | Higher is better |
| **Sensitivity** | Model's sensitivity to masked regions | 0-1 |
| **Specificity** | Relevance to predicted class | 0-1 |

---

## 🔬 Research Background

This project implements concepts from attention visualization and explainability research:

- **Vision Transformers (ViT)**: Dosovitskiy et al., 2020
- **Attention Rollout**: Abnar & Zuidema, 2020
- **Gradient-based Attribution**: Simonyan et al., 2013

The novel GMAR approach combines:
1. ✨ Gradient weighting for attention head importance
2. 🔄 Residual-aware rollout across layers
3. 📊 Multi-head aggregation with learned weights

---

## 🛠️ Configuration

### Environment Variables

```bash
# GPU/CPU selection
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0, or leave unset for CPU

# Model checkpoint location (optional)
export CHECKPOINT_PATH=./checkpoints/vit_large_tinyimagenet/best
```

### Hyperparameters

Edit `src/engine.py` to modify:
- **alpha** (0.0-2.0): Residual scaling factor for rollout
- **norm_type** ('l1' or 'l2'): Gradient normalization method
- **batch_size**: Number of images to process simultaneously
- **image_size**: Input resolution (default: 224×224)

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

```bash
# Reduce batch size in src/engine.py
BATCH_SIZE = 4  # Default is 8
```

### Issue: Model Checkpoint Not Found

```bash
# Download pre-trained weights
cd checkpoints/vit_large_tinyimagenet/
# Follow instructions in checkpoints/README.md
```

### Issue: Import Errors

```bash
# Ensure virtual environment is activated
source env/bin/activate  # macOS/Linux
env\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirement.txt --force-reinstall
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0 | Deep learning framework |
| torchvision | ≥0.15 | Computer vision utilities |
| transformers | ≥4.30 | Pre-trained models & tokenizers |
| pillow | ≥9.0 | Image processing |
| numpy | ≥1.21 | Numerical computing |
| matplotlib | ≥3.5 | Visualization |
| scikit-learn | ≥1.0 | ML utilities |

---

## 🎓 Citation

If you use GMAR++ in your research, please cite:

```bibtex
@software{gmar_plus_2024,
  title={GMAR++: Efficient Gradient-Enhanced Attention Rollout},
  author={[Your Name]},
  year={2024},
  url={https://github.com/soham5498/GMAR-Plus-Efficient-Gradient-Enhanced-Attention-Rollout}
}
```

---

## 📝 License

This project is licensed under the MIT License. See LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 💡 Future Work

- [ ] Support for additional transformer architectures (BERT, Swin, etc.)
- [ ] Interactive visualization dashboard
- [ ] Real-time explanation generation
- [ ] Quantitative benchmark suite
- [ ] Model-agnostic explanation support
- [ ] Mobile deployment support

---

## 📞 Support & Contact

For questions, issues, or suggestions:

- **GitHub Issues**: [Report a bug](https://github.com/soham5498/GMAR-Plus/issues)
- **Email**: [contact information]
- **Documentation**: Check the `docs/` directory

---

## 🙏 Acknowledgments

- Vision Transformer research community
- PyTorch and Hugging Face teams
- All contributors and users

---

<div align="center">

**Made with ❤️ for Explainable AI**

[⬆ Back to Top](#gmar-efficient-gradient-enhanced-attention-rollout)

</div>
