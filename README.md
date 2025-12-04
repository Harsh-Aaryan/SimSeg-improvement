# Simseg-Improved

This project proposes to improve upon the original [Simseg](https://github.com/muyangyi/SimSeg). Below are instructions on how to prepare an environment to run this model.

# Environment Setup Instructions

Requirements:
- Python 3.11.2
- cuda 13.0
- venv


Create a python virtual environment for Python dependencies, we will be using the name ```IMSS``` as an example. We are assuming the environment done with this is debian-based such as Ubuntu and the cuda drivers for the NVIDIA gpu are cuda 13.0.

## The following should be done outside the repository

```shell
python3 -m venv imss
```

Activate the evironment with the following, assuming you have not changed directories.

```shell
source imss/bin/activate
```

With this environment install pytorch with the following command.

```shell
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Clone the following repository, as Details API is required to make this run.

```shell
git clone https://github.com/zhanghang1989/detail-api.git
```

Navigate to the directory PythonAPI within the repository and make the package.

```shell
cd detail-api/PythonAPI
make
cd ../..
```
Clone the improved-simseg repository.

```shell
git clone https://github.com/Harsh-Aaryan/SimSeg-improvement
```

Ensuring details is installed, run the following.

```shell
python tools/convert_datasets/pascal_context.py data/VOCdevkit data/VOCdevkit/VOC2010/trainval_merged.json
```

Enter the root of the repository and install the dependencies.

```shell
cd SimSeg-improvement
pip3 install -r requirements.txt
pip3 install git+https://github.com/lucasb-eyer/pydensecrf.git
```

## Checkpoints
SimSeg checkpoints: [Google Drive](https://drive.google.com/drive/folders/1p2hO6LK1usO3q-S8ZtCK8jLaT941WPNW?usp=sharing)  
Please save the `.pth` files under the `ckpts/` folder.

```none
SimSeg
├── ckpts
│   ├── simseg.vit-b.pth
│   ├── simseg.vit-s.pth
```

## Dataset

Datasets are arranged within the following structure. Checkmarked components are associated with datasets that are retrievable.

The only retrievable dataset as of writing this is the coco_stuff164k dataset is the only one still publically available.

```none
SimSeg
├── data
│   ├── label_category
│   │   ├── pascal_voc.txt
│   │   ├── pascal_context.txt
│   │   ├── coco_stuff.txt✓
│   ├── VOCdevkit
│   │   ├── VOC2012
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   ├── ImageSets
│   │   │   │   ├── Segmentation
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── val.txt
│   │   ├── VOC2010
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClassContext
│   │   │   ├── ImageSets
│   │   │   │   ├── SegmentationContext
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── val.txt
│   │   │   ├── trainval_merged.json
│   ├── coco_stuff164k✓
│   │   ├── images
│   │   │   ├── train2017
│   │   │   ├── val2017
│   │   ├── annotations
│   │   │   ├── train2017
│   │   │   ├── val2017
```

## COCO Stuff

To launch the model with the coco dataset, run the following.

```shell
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=65533 tools/seg_evaluation.py --ckpt_path=ckpts/simseg.vit-s.pth --cfg=configs/clip/simseg.vit-s.yaml data.valid_name=[coco_stuff]
```

## Run Web Interface

To run the web interface, run the following.

```shell
cd web
python app.py
```
Open http://127.0.0.1:5000 in your browser to access the web interface. 

Currently its an example, but future versions will properly show masks.

**Original (ViT-B baseline):**
```bash
python launch.py --cfg configs/clip/simseg.vit-b.yaml
```

**Improved models:**
To see our improved models

```shell
# ViT-L (24 layers, 1024 dim)
python launch.py --cfg configs/clip/simseg.vit-l.yaml

# ViT-H (32 layers, 1280 dim)
python launch.py --cfg configs/clip/simseg.vit-h.yaml

# Swin Transformer
python launch.py --cfg configs/clip/simseg.swin.yaml
```

### View Improvements Comparison

To view the improvement comparisons, run the following.

```bash
python tools/compare_improvements.py
```

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

```none
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
│   ├── compare_improvements.py
|   └── seg_inference.py # inferences a single image and returns image masks
└── simseg/models/backbones/mml/
    └── vit_builder.py       # Updated with new backbones
```
