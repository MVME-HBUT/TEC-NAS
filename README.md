# TEC-NAS

**Training-free Neural Architecture Search for Freight Train Fault Detection**

TEC-NAS explores efficient detection backbones using training-free architecture evaluation. This repository provides the model implementation and evaluation configurations for the Bottom View and Side View freight-train fault detection datasets.

## Repository Structure

```text
TEC-NAS/
├── architectures/res50_like.txt
├── configs/tec_nas/
│   ├── bottom_view.py
│   └── side_view.py
├── mmdet/
├── tools/test.py
├── scripts/
│   ├── test_bottom.sh
│   └── test_side.sh
├── requirements/
├── requirements.txt
└── setup.py
```

## Environment

The code is based on MMDetection 2.6.0. The original development environment used:

* Python 3.8.20
* PyTorch 1.8.0 (CUDA 11.1)
* MMCV 1.3.0

After installing compatible PyTorch and MMCV dependencies, install the project:

```bash
python -m pip install -e .
```

## Data Preparation

Prepare COCO-format validation images and annotations at the following paths:

```text
data/
├── Bottom_View_V2/
│   ├── annotations/instances_val2017.json
│   └── images/val/
└── Side_View_V2/
    ├── annotations/instances_val2017.json
    └── images/val2017/
```

The Bottom View and Side View configurations use 16 and 12 detection classes, respectively.

## Evaluation

Run the following commands from the repository root with the corresponding model checkpoints:

```bash
# Bottom View
bash scripts/test_bottom.sh /path/to/bottom_checkpoint.pth

# Side View
bash scripts/test_side.sh /path/to/side_checkpoint.pth
```

Both scripts perform COCO bounding-box evaluation (`--eval bbox`) on a single GPU.

Datasets and model checkpoints are not included in the repository.

## License

See [LICENSE](LICENSE).
