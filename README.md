# TEC-NAS

**Research code for efficient freight-train visual fault detection using neural architecture search.**

This repository is being prepared for public evaluation of TEC-NAS on the **Bottom View** and **Side View** datasets. It includes an adapted MMDetection 2.6.0 codebase, a custom `MadNas` backbone, its architecture description, and two evaluation configurations.

> **Release status:** Source/configuration checks and dependency-import checks have passed in the original environment. End-to-end public-config checkpoint evaluation and reproduction of paper metrics have **not yet been independently confirmed**. Do not describe the repository as fully reproduced until those checks are completed.

## Repository layout

```text
TEC-NAS/
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
├── requirements/
├── architectures/
│   └── res50_like.txt
├── configs/
│   └── tec_nas/
│       ├── bottom_view.py
│       └── side_view.py
├── mmdet/
├── tools/
│   └── test.py
├── scripts/
│   ├── test_bottom.sh
│   └── test_side.sh
├── checkpoints/
│   └── README.md
└── data/
    └── README.md
```

The files under `mmdet/` are derived from MMDetection and associated research implementations; retain all original upstream copyright and license notices. `LICENSE` and upstream license obligations must be reviewed before release.

## Verified environment (original Linux machine)

| Component | Version / check |
|---|---|
| Python | 3.8.20 |
| PyTorch | 1.8.0+cu111 |
| MMCV | 1.3.0 |
| MMDetection source | 2.6.0 |
| `mmcv.ops.nms` import | PASS |
| `mmdet.apis.single_gpu_test` import | PASS |

The bundled `mmdet/__init__.py` requires **MMCV >=1.1.5, <=1.3.0**. Avoid upgrading to modern MMDetection/MMCV in this environment without a compatibility port.

### Installation

Use a Linux environment with a compatible GPU/CUDA stack. Install Python 3.8, PyTorch 1.8.0 with its matching CUDA build, and **MMCV with compiled operations** for your specific PyTorch/CUDA combination **before** installing this package. The original environment has been tested for imports, but this installation sequence has not yet been independently validated on a clean machine.

From the project root, after installing matching PyTorch and MMCV dependencies:

```bash
python -m pip install -e .
python -c "import torch, mmcv, mmdet; from mmcv.ops import nms; print(torch.__version__, mmcv.__version__, mmdet.__version__)"
```

`requirements.txt` includes optional and test requirements; for a minimal setup, inspect `requirements/runtime.txt` and `setup.py` first. Keep legacy package versions compatible with Python 3.8 and PyTorch 1.8.

## Dataset preparation

Prepare **COCO-style** validation annotations and images, using exactly these project-relative locations:

```text
data/
├── Bottom_View_V2/
│   ├── annotations/
│   │   └── instances_val2017.json
│   └── images/
│       └── val/
│           └── <validation images>
└── Side_View_V2/
    ├── annotations/
    │   └── instances_val2017.json
    └── images/
        └── val2017/
            └── <validation images>
```

The Bottom View configuration expects **16 classes**, and the Side View configuration expects **12 classes**. The original environment contained 5,165 Bottom View validation images and 3,713 Side View validation images. See [`data/README.md`](data/README.md) for integrity notes and split-overlap disclosure.

The datasets are **not** included in this repository. Ensure you have permission to distribute or use the underlying image/annotation data.

## Architecture and evaluation configurations

- Backbone: `MadNas`
- Architecture file: `architectures/res50_like.txt`
- FPN input channels: `[256, 512, 768, 1024]`
- Bottom View config: `configs/tec_nas/bottom_view.py`
- Side View config: `configs/tec_nas/side_view.py`

Run commands **from the repository root** or use the helper scripts, which set the working directory automatically. The `architectures/res50_like.txt` reference is relative to the working directory.

## Evaluation (single GPU)

Supply your own checkpoint path. Neither the dataset nor the pretrained checkpoints are bundled in this repository.

```bash
# Bottom View (16 classes)
bash scripts/test_bottom.sh /absolute/path/to/bottom_checkpoint.pth

# Side View (12 classes)
bash scripts/test_side.sh /absolute/path/to/side_checkpoint.pth
```

The scripts check the necessary paths, set `PYTHONPATH`, and run `tools/test.py` with `--eval bbox`. You may choose a GPU with `CUDA_VISIBLE_DEVICES=0`.

Equivalent command, when already at the project root:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python tools/test.py \
  configs/tec_nas/bottom_view.py \
  /absolute/path/to/bottom_checkpoint.pth \
  --eval bbox
```

**Do not assume a training checkpoint matches the paper's published metrics** merely because its class count and architecture agree. Confirm checkpoint provenance, class ordering, evaluation split, and actual COCO AP output before publishing numerical reproduction claims.

See [`checkpoints/README.md`](checkpoints/README.md) for additional details.

## Reproducibility and limitations

- The public draft configurations were checked for syntax, relative data paths, expected class counts and architecture presence.
- Imports of `mmcv.ops.nms` and the MMDetection test API succeeded on the original Linux environment.
- Model construction, checkpoint-loading compatibility, and end-to-end single-GPU evaluation using these **public** configurations still require final verification.
- A previous dataset integrity audit detected identical image content across train/validation for both views: **89** Bottom View and **94** Side View matches. See [`data/README.md`](data/README.md). This should be disclosed alongside evaluation metrics.
- Paper tables report numbers in different experimental settings; keep them clearly separated rather than mixing results from different comparisons.

## Citation

**Publication metadata / BibTeX to be supplied by the project authors after verification.**

## Acknowledgements and licenses

This codebase builds on the MMDetection 2.6.0 ecosystem and the original GFocal-style detector implementation. Preserve the existing [`LICENSE`](LICENSE) and all third-party source file notices. Review the provenance and terms of any additional copied code before public distribution.
