# Dataset setup

No images or annotation files are shipped in this repository.

## Required test paths

```text
data/Bottom_View_V2/
├── annotations/
│   └── instances_val2017.json
└── images/
    └── val/

data/Side_View_V2/
├── annotations/
│   └── instances_val2017.json
└── images/
    └── val2017/
```

Annotations must use COCO-style detection format. The published evaluation configurations expect:

| Dataset | Validation images in original environment | Detection classes |
|---|---:|---:|
| Bottom View | 5,165 | 16 |
| Side View | 3,713 | 12 |

Use the exact category identifiers and class ordering corresponding to each checkpoint; a matching number of classes alone is insufficient.

## Data leakage / split overlap disclosure

A previous local SHA256-based content comparison of the available train and validation image sets reported:

- **Bottom View:** 9,506 training images, 5,165 validation images, **89** identical-content train/validation matches.
- **Side View:** 6,912 training images, 3,713 validation images, **94** identical-content train/validation matches.

These are observations from the inspected local datasets, not a guarantee about any future public redistribution or revised split. Confirm dataset provenance and report this overlap when publishing validation results. Do not silently modify splits to chase a target benchmark number.

## Distribution

Only publish or link to datasets where access, privacy, and redistribution rights have been confirmed.
