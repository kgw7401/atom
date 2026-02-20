# Notes

## Data Annotation

Manual annotation is painful. Labeling frame-level action segments from video is slow, tedious, and error-prone — but the quality of training data directly determines model performance. Garbage in, garbage out.

### Pain points
- Scrubbing through video frame-by-frame to find exact action boundaries is mind-numbing
- Each video has mixed actions (guard → jab → guard → cross → ...), all needing individual labels
- Subjective boundary decisions: when exactly does a jab "start" and "end"?
- Scale problem: need hundreds of labeled segments across multiple actions, angles, speeds

### Idea: automatic annotation tool
An auto-labeling pipeline would be hugely valuable. Possible approaches:
- **Velocity-based segmentation**: detect wrist acceleration spikes to find punch start/end automatically, then only require human to confirm the action *type*
- **Semi-supervised**: label a small seed set manually, train a weak classifier, use it to pre-label the rest, then human just corrects mistakes
- **Active learning**: model picks the most uncertain segments and asks for labels only on those
- **LLM-assisted**: feed pose sequence descriptions to an LLM to suggest labels

Even a tool that just auto-detects "something happened here" (motion spike) and presents candidate segments for quick label assignment would cut annotation time dramatically.

### Lesson
Data labeling is the real bottleneck, not model architecture. Investing in annotation tooling pays off more than tuning hyperparameters.

---

## First Training Result (2026-02-20)

### Dataset
- 159 annotated segments across 3 actions (guard, jab, cross)
- 3 camera angles (front, left45, right45)
- 525 total windows after preprocessing
- Class distribution: guard 64%, cross 20%, jab 16% (guard segments are longer → more windows)

### Result: 98.1% accuracy (Random Forest baseline)

| | precision | recall | f1 |
|---|---|---|---|
| cross | 1.00 | 0.95 | 0.98 |
| guard | 0.97 | 1.00 | 0.99 |
| jab | 1.00 | 0.94 | 0.97 |

Only 2 misclassifications out of 105 test windows (1 cross → guard, 1 jab → guard).

### Key fix: window_size must match data
- Original config had `window_size: 30` but punches are only 6-11 frames long
- 157/159 segments were shorter than 30 frames → pipeline discarded almost everything → training crashed
- Fix: `window_size: 6` (= min segment length), `stride: 2`
- Also reduced `smoothing_window: 7 → 5` (must be <= window_size)

### Config that worked
- 11 keypoints (added ears for guard detection — fist-to-face proximity)
- window_size=6, stride=2
- Savitzky-Golay smoothing: window=5, polyorder=2
- Hip-center normalization + shoulder-width scaling
- RF: n_estimators=100, max_depth=20

### Takeaway
Always check that `window_size` fits your actual segment lengths. A mismatch silently drops all data.

---

## Biggest Lesson: Data Annotation Changes Everything

Before proper annotation, the classifier was useless. Each raw video mixed multiple actions (guard → jab → guard → cross → ...), so the model was trained on mislabeled windows — a "jab" video was mostly guard frames labeled as "jab". The model couldn't learn anything meaningful.

After annotating each action segment individually with the trim tool:
- **Before annotation**: model couldn't classify anything (training crashed or gave random results)
- **After annotation**: 98.1% accuracy, real-time classification actually works on live webcam

The model architecture (Random Forest) and preprocessing pipeline didn't change at all. The only difference was **clean, correctly-labeled data**.

This is the most visceral proof that **data quality > model complexity**. A simple RF with clean labels crushes a fancy model with noisy labels. The hours spent on annotation felt tedious, but it was the single highest-leverage activity in the entire project.
