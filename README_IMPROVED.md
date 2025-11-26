# SimSeg - Improved Semantic Segmentation

Enhanced SimSeg with ViT-L, ViT-H, Swin Transformer, mixed precision training, and a web interface.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Web Interface
```bash
cd web
python app.py
```
Open http://127.0.0.1:5000 in your browser.

### 3. Train Models

**Original (ViT-B baseline):**
```bash
python launch.py --cfg configs/clip/simseg.vit-b.yaml
```

**Improved models:**
```bash
# ViT-L (24 layers, 1024 dim)
python launch.py --cfg configs/clip/simseg.vit-l.yaml

# ViT-H (32 layers, 1280 dim)
python launch.py --cfg configs/clip/simseg.vit-h.yaml

# Swin Transformer
python launch.py --cfg configs/clip/simseg.swin.yaml
```

### 4. View Improvements Comparison
```bash
python tools/compare_improvements.py
```

---

## Improvements Summary

| Feature | Original | Improved |
|---------|----------|----------|
| Backbone | ViT-B (12 layers) | ViT-L/H, Swin |
| Mixed Precision | No | Yes (40-50% memory savings) |
| Flash Attention | No | Yes (20-30% faster) |
| Training Speed | 1x | ~2x |

## Web Interface Features

- Upload images for segmentation
- Real-time colored mask visualization
- Confidence scores per category
- Export masks/overlays as PNG
- Compare original vs improved models

## Project Structure

```
SimSeg/
├── configs/clip/
│   ├── simseg.vit-b.yaml    # Original
│   ├── simseg.vit-l.yaml    # NEW: ViT-L
│   ├── simseg.vit-h.yaml    # NEW: ViT-H
│   └── simseg.swin.yaml     # NEW: Swin
├── web/
│   ├── app.py               # Flask backend
│   └── templates/index.html # Web UI
├── tools/
│   └── compare_improvements.py
└── simseg/models/backbones/mml/
    └── vit_builder.py       # Updated with new backbones
```

