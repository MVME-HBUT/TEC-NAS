# TEC-NAS model checkpoints

Pretrained checkpoints are **not included** in this repository.

The evaluation scripts accept an explicit checkpoint path, so weights may be located anywhere accessible to the Linux machine.

```bash
bash scripts/test_bottom.sh /path/to/bottom_checkpoint.pth
bash scripts/test_side.sh /path/to/side_checkpoint.pth
```

## Previously identified local candidates (not reproduced/approved for release)

The historical source project contained these candidate files:

- **Bottom View (16 classes, saved epoch 20):**
  `work_dirs_bottom/Resnet50/Budget_FLOPs_Params/e+l+f/custom_maedet_s_4x_lr0.02_r50_3/epoch_20.pth`
- **Side View (12 classes, saved epoch 46):**
  `work_dirs/Resnet50/Budget_FLOPs_Params/e+l+f/custom_maedet_s_4x_lr0.02_r50/epoch_46.pth`

These paths are **relative to the historical source project**, not to this public draft. The class metadata and existence of these files were previously inspected, but **final compatibility with the new public configurations, evaluation results, rights, and distribution links still need verification**.

Before distributing weights, confirm:
1. Ownership and distribution permission.
2. Dataset version and exact class ordering.
3. Model architecture and checkpoint loading.
4. Full evaluation AP values against the intended validation set.
5. Checkpoint file hashes and published download location.

Do not add large `.pth` files to Git without an explicit distribution decision.
