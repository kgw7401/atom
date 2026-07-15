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
  recall 0.6121), using actor-only hybrid-motion features and a TCN.
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
- `rtmw2d-fullrate`: full-rate RTMW-m extraction; 40 train and 3 test matches
  are complete. The interrupted fourth test match was not retained.
- `artifacts`: 79 checkpoints and experiment reports, including the best
  GT-free ensemble and full-rate TCN.
- `models`, `mmpose-source`, and `venv`: pose weights, configs/source, and a
  verified runnable Python environment. The environment currently imports
  PyTorch 2.13.0 and MMPose 1.3.2 successfully.

## Resume point

First complete the seven remaining full-rate test matches. Existing outputs are
cached, so rerunning the test split will skip the first three:

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

Then resume validation experiments in this order:

1. Train full-rate BiGRU and MSTCN variants using 2D hybrid-motion features.
2. Create a full-rate 2D+depth composite and test TCN/BiGRU variants.
3. Analyze validation false positives and misses by duration, boxer, and motion
   pattern; adjust candidate/window sampling based on that evidence.
4. Select the ensemble and thresholds using validation only.
5. Retrain the selected configuration on all 40 training matches and evaluate
   the held-out test set exactly once.

The interrupted full-rate BiGRU had not completed an epoch and produced no
checkpoint, so it should be restarted from the beginning.
