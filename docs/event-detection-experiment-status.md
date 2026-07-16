# Event detection experiment status

Last saved: 2026-07-16 (Asia/Seoul)

## Goal

Build a video-to-punch-event detector using video-extracted pose, and exceed the
BoxMind paper's reported event-detection F1 of 0.783 at temporal IoU 0.5. Model
selection is performed only on the last four BoxingWeb training matches; the
held-out test labels have not been used for model selection.

## Current result

- Best GT-free validation result: F1 0.7069 (precision 0.6925, recall 0.7220).
  This is an equal-weight ensemble of two existing RTMW 2D models, the composite
  2D+depth TCN, and the composite 2D+depth BiGRU.
- Best single full-frame-rate RTMW 2D model: F1 0.6602 (precision 0.7165,
  recall 0.6121), using actor-only hybrid-motion features and a TCN. A
  multi-scale TCN subsequently improved the best single-model result to F1
  0.6659 (precision 0.6824, recall 0.6502).
- Best single stride-3 composite 2D+depth model: F1 0.6505.
- GT-pose diagnostics peak at about F1 0.7495 with BiGRU. This is diagnostic
  only and is not part of the deployable ensemble.
- Target remaining: F1 greater than 0.783 on validation, followed by one final
  held-out test evaluation.

## Persisted state

All large experimental data was moved out of `/tmp` so it survives a restart:

`/Users/kgw7401/.local/share/atom-experiments/2026-07-16`

- `rtmw2d-stride3`: complete RTMW-m 2D pose for 40 train + 10 test matches.
- `rtmw3d-stride3`: complete RTMW3D-L pose for 40 train + 10 test matches.
- `rtmw2d3d-stride3`: complete RTMW-m 2D plus RTMW3D depth composite data.
- `rtmw2d-fullrate`: complete full-rate RTMW-m extraction for 40 train + 10
  test matches.
- `rtmw2d-fullrate-3d-stride3`: complete full-rate 2D plus stride-3 RTMW3D
  depth composite data for 40 train + 10 test matches.
- `artifacts`: 79 checkpoints and experiment reports, including the best
  GT-free ensemble and full-rate TCN.
- `models`, `mmpose-source`, and `venv`: pose weights, configs/source, and a
  verified runnable Python environment. The environment currently imports
  PyTorch 2.13.0 and MMPose 1.3.2 successfully.

## Resume point

Full-rate extraction is complete. The following command can be used only for an
integrity-preserving cached rerun if files are later found to be missing:

```bash
EXP=/Users/kgw7401/.local/share/atom-experiments/2026-07-16
$EXP/venv/bin/python scripts/extract_boxingweb_rtmw.py \
  --data-root /Users/kgw7401/boxingweb \
  --output-root $EXP/rtmw2d-fullrate \
  --split test \
  --python $EXP/venv/bin/python \
  --pose-config $EXP/mmpose-source/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-m_8xb1024-270e_cocktail14-256x192.py \
  --pose-checkpoint $EXP/models/rtmw2d-m.pth \
  --yolo $EXP/models/yolo11n.pt \
  --device mps \
  --workers 1 \
  --batch-frames 48 \
  --inference-stride 1 \
  --smooth-window 3
```

Latest validation experiments (held-out training matches only):

- Full-rate 2D TCN: F1 0.6602.
- Full-rate 2D BiGRU: F1 0.6409.
- Full-rate 2D multi-scale TCN: F1 0.6659.
- Ensemble of those three full-rate models: F1 0.6875.
- Full-rate 2D plus stride-3 depth TCN: F1 0.6482.
- Three positive windows per event: F1 0.6458.
- Focal alpha 0.99: F1 0.6127.
- Full-rate multi-lag input features: F1 0.6471.

For correctly matched events from the best full-rate single model, mean
temporal IoU is 0.7604. Of 446 validation events, 145 have no same-hand
prediction near IoU 0.25, while only 11 fall between IoU 0.25 and 0.5. The
dominant remaining failure is therefore candidate classification/recall rather
than boundary regression.

Resume validation work in this order:

1. Add and validate explicit arm-extension/acceleration and opponent-distance
   kinematic features, targeting the 145 missing candidates.
2. Combine full-rate and stride-3 model predictions using per-component pose
   roots, then select weights and thresholds using validation only.
3. Retrain the selected configuration on all 40 training matches and evaluate
   the held-out test set exactly once.
