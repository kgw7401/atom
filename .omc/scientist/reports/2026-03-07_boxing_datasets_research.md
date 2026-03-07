# Boxing Action Detection Datasets: Comprehensive Research Report
**Date:** 2026-03-07
**Objective:** Identify all publicly available datasets for training boxing action detection models, assess BoxingVI as a starting point, and surface transfer-learning candidates.

---

## [OBJECTIVE]

Systematically catalog every known boxing-specific and combat-sports-adjacent dataset suitable for training a punch action detection model, evaluate their relative utility, and determine whether BoxingVI (6,915 clips, 6 punch types) is the optimal starting point or whether superior alternatives exist.

---

## [DATA]

**Search coverage:**
- arXiv 2017–2026 (boxing, punch, action detection, temporal localization)
- Kaggle (martial arts, boxing, action recognition tags)
- HuggingFace datasets hub
- Papers With Code task pages
- Roboflow Universe
- GitHub topics: boxing, action-recognition, pose-estimation
- MDPI, Springer, IEEE Xplore, PubMed, ScienceDirect
- Key benchmark sites: UCF101, Kinetics, HMDB51, NTU RGB+D, KTH, MADS, FineGym

**Total datasets cataloged:** 22 (9 boxing-specific, 13 transfer/related)

---

## PART 1: BOXING-SPECIFIC DATASETS

---

### 1. BoxingVI
**[FINDING] Best-documented publicly available boxing dataset with multi-modal annotations**

| Field | Detail |
|-------|--------|
| Full name | BoxingVI: A Multi-Modal Benchmark for Boxing Action Recognition and Localization |
| Size | 6,915 temporally segmented punch clips (5,513 train / 1,402 val) |
| Source | 20 unedited YouTube sparring videos, 18 athletes (11M, 7F) |
| Labels | 6 punch types: Jab, Cross, Lead Hook, Rear Hook, Lead Uppercut, Rear Uppercut |
| Annotations | (1) Temporal segmentation with start/end frames + class label; (2) 2D AlphaPose keypoints, zero-padded to 25 frames at 30 fps |
| Modalities | RGB video clips + 2D pose sequences |
| Availability | Public — GitHub: https://github.com/Bikudebug/BoxingVI |
| License | Fair Use (YouTube links + temporal annotations, no raw video redistribution) |
| Paper | arXiv 2511.16524 (November 2024), Ministry of Youth Affairs and Sports supported |
| Unique value | Only dataset combining RGB, temporal punch boundaries, class labels, AND pose trajectories simultaneously |

[STAT:n] n = 6,915 clips across 6 classes
[STAT:effect_size] Reported 91% test accuracy with PoseConv3D skeleton-based model
[LIMITATION] All footage from YouTube sparring; no professional match footage; max 25 frames per punch limits long-duration action modeling.

---

### 2. Olympic Boxing Punch Classification Video Dataset (FACTS / Stefanski)
**[FINDING] Largest frame-count boxing dataset with referee-grade annotations, freely available on Kaggle**

| Field | Detail |
|-------|--------|
| Full name | Olympic Boxing Punch Classification Video Dataset |
| Size | ~4 hours footage; 312,774 annotated frames; 28 video clips (14 per camera) |
| Source | 2021 Polish boxing league, 4 GoPro cameras at ring corners |
| Labels | 8 classes: Left/Right Hand Head Punch, Left/Right Hand Body Punch, Left/Right Hand Block, Left/Right Hand Miss |
| Annotations | Frame-level bounding box + class label by licensed boxing referee; 11,345 punch frames vs 301,429 non-punch frames |
| Format | Full HD 1080p @ 50 fps |
| Availability | Kaggle (3.34 GB compressed): https://www.kaggle.com/datasets/piotrstefaskiue/olympic-boxing-punch-classification-video-dataset |
| License | Non-commercial only (academic/non-profit research) |
| Also available | Unlabeled version: https://www.kaggle.com/datasets/piotrstefaskiue/olympic-boxing-video-dataset-unlabeled |
| Paper | Stefanski et al., *Boxing Punch Detection with Single Static Camera*, MDPI Entropy 2024 |

[STAT:n] 312,774 labeled frames; 11,345 positive punch frames
[STAT:effect_size] Class imbalance ratio ~1:27 (punch vs no-punch) — significant modeling challenge
[LIMITATION] Severe class imbalance. Faces blurred (MOGFace). Only 8 action types, no fine-grained punch-side + type combination (e.g., lead jab vs rear jab). Non-commercial license restricts deployment use.

---

### 3. BoxMAC
**[FINDING] Most label-rich boxing dataset (13 classes, multi-label) but currently inaccessible due to paper withdrawal**

| Field | Detail |
|-------|--------|
| Full name | BoxMAC: A Boxing Dataset for Multi-label Action Classification |
| Size | 60,000+ annotated frames; 2,314 video clips; 15 professional boxing matches |
| Source | 15 professional boxers, 25 fps, avg 4 min per video |
| Labels | 13 classes: Jab, Cross, Lead Hook, Rear Hook, Lead Uppercut, Rear Uppercut, Stance, Slip, Block, Guard, Duck, Clinching, No Action |
| Annotations | Multi-label per frame (both boxers annotated simultaneously); coach-supervised |
| Train/Test | 42,000/18,000 frames; 1,584/730 clips |
| Availability | UNAVAILABLE — paper withdrawn from arXiv Feb 17, 2025; "revised version forthcoming" |
| License | None (withdrawn) |
| Paper | arXiv 2412.18204 (December 2024, withdrawn) |

[STAT:n] 60,000 frames, 2,314 clips
[LIMITATION] Paper withdrawn; dataset not publicly released. Multi-label design (two boxers per frame) is methodologically sound but the dataset is inaccessible as of March 2026. Monitor arXiv for resubmission.

---

### 4. ShadowPunch
**[FINDING] Only high-frame-rate shadowboxing dataset with frame-level event spotting + pose keypoints; CC BY 4.0**

| Field | Detail |
|-------|--------|
| Full name | ShadowPunch: Fast Actions Spotting Benchmark |
| Size | Not fully disclosed in abstract; high-fps shadowboxing videos with frame-level annotations |
| Source | Shadowboxing sessions (single athlete, no opponent) |
| Labels | Multiple punch types (specific taxonomy in paper PDF); frame-level event spotting labels |
| Annotations | Frame-level punch event timestamps + 2D pose keypoints |
| Format | High frame rate (>30 fps) — designed to capture fast punch kinematics |
| Availability | CC BY 4.0 — "code and dataset publicly available" per paper; supplementary zip on OpenReview |
| License | Creative Commons CC BY 4.0 (most permissive of all boxing datasets) |
| Paper | OpenReview (ICLR 2025 submission): https://openreview.net/forum?id=Jq8HYNZG9s |

[STAT:n] Exact clip count not publicly disclosed; frame-level annotations at high fps
[LIMITATION] Shadowboxing only — no opponent, no contact, simpler visual context than sparring. Paper is ICLR 2025 withdrawal/revision; exact GitHub URL not resolvable from public search. Task framing is event spotting (temporal), not clip classification.

---

### 5. FACTS Boxing Dataset
**[FINDING] 8,000 clips with 8 classes, referee-annotated — but sourced from Olympic dataset above**

| Field | Detail |
|-------|--------|
| Full name | FACTS: Fine-Grained Action Classification for Tactical Sports (Boxing subset) |
| Size | 8,000 action clips |
| Source | Reprocessed from Olympic Boxing Punch Classification Video Dataset (Kaggle) |
| Labels | 8 classes: Left/Right Hand Head Punch, Left/Right Hand Body Punch, Left/Right Hand Block, Left/Right Hand Miss |
| Annotations | Frame-by-frame by licensed boxing referees |
| Availability | Fencing subset: https://anonymous.4open.science/r/FACTS-B1C5 (public); boxing subset derived from Kaggle dataset |
| Paper | arXiv 2412.16454 (December 2024) |
| Model | Transformer-based, 83.25% accuracy on boxing actions |

[STAT:n] 8,000 clips, 8 classes
[STAT:effect_size] 83.25% accuracy with transformer model (no pose estimation required)
[LIMITATION] Boxing portion is not a new data source — it re-segments the Kaggle Olympic dataset. Fencing subset is independently novel.

---

### 6. Punch_DL (Boxing + Karate)
**[FINDING] Small but downloadable dataset with 7 punch types + strong/weak variants; Google Drive hosted**

| Field | Detail |
|-------|--------|
| Full name | Punch_DL: Data and code for punch classification in karate and boxing |
| Size | 240 punches (v0.1) |
| Source | Controlled lab recording; participants performing 10 weak + 10 strong punches per hand |
| Labels | 7 classes: No Punch, Jab Left, Cross/Jab Right, Left Hook, Right Hook, Left Uppercut, Right Uppercut + strong/weak variants |
| Modalities | Video frames + keypoints + accelerometer sensor data |
| Pre-trained models | Keras + TensorFlow Lite models included |
| Availability | Google Drive: https://drive.google.com/drive/folders/1UwZPZ7sqkmQrqbCP1ypquv2UHWkk0bj- |
| License | Not specified |
| GitHub | https://github.com/balezz/Punch_DL |

[STAT:n] 240 clips total — too small for standalone training
[LIMITATION] Tiny dataset (240 clips). Lab-controlled setting, not match footage. No license specified. Best used for augmentation or validation.

---

### 7. BoxNet Dataset (Implicit)
**[FINDING] IEEE 2023 paper with custom skeleton-based boxing dataset; not independently released**

| Field | Detail |
|-------|--------|
| Full name | BoxNet (Yang & Xia, CCDC 2023) |
| Size | Not publicly disclosed |
| Labels | Fine-grained boxing technical actions (punch types via GCN) |
| Modalities | Skeleton/pose sequences |
| Availability | Not publicly released; method paper only |
| Paper | IEEE CCDC 2023: https://ieeexplore.ieee.org/document/10327379/ |

[LIMITATION] Dataset not released; paper describes method only.

---

### 8. BoxingPro Multimodal Dataset
**[FINDING] IMU + video synchronized dataset at 200 fps; announced for release but not yet available**

| Field | Detail |
|-------|--------|
| Full name | BoxingPro multimodal boxing dataset |
| Size | Not fully disclosed (synchronized IMU + high-fps video) |
| Source | Side-view camera at chest height + wrist IMU sensors |
| Labels | Multiple punch types |
| Modalities | 200 fps video + 9-axis IMU (acc XYZ, gyro XYZ, pitch/roll/yaw) |
| Availability | "Will be publicly released" — not yet available as of March 2026 |
| Paper | MDPI Electronics, October 2025: https://www.mdpi.com/2079-9292/14/21/4155 |

[LIMITATION] Not yet available. IMU modality requires wearables at inference — not applicable to broadcast video inference.

---

### 9. EPFL Boxing Multiview Interaction Dataset
**[FINDING] Multi-camera boxing dataset from CVLAB EPFL — listed as WIP, no data accessible**

| Field | Detail |
|-------|--------|
| Full name | Boxing-Multiview-Interaction-Dataset |
| Source | CVLAB, EPFL |
| Availability | Work-in-progress; no data released: https://www.epfl.ch/labs/cvlab/data/boxing-multiview-interaction-dataset/ |

[LIMITATION] No data released; WIP page only. Monitor for future release.

---

### 10. Fine-Grained Boxing Punch Recognition from Depth Imagery (Kasiri et al.)
**[FINDING] Early depth-based dataset with overhead Kinect — not publicly released**

| Field | Detail |
|-------|--------|
| Full name | Kasiri et al. boxing depth dataset |
| Modality | Overhead depth imagery (time-of-flight / Kinect) |
| Labels | Fine-grained punch trajectories (jab, cross, hook, uppercut) |
| Availability | Not publicly released |
| Paper | ScienceDirect, Computer Vision and Image Understanding, 2017 |

[LIMITATION] Not public; depth-only; overhead camera angle limits applicability to broadcast scenarios.

---

### 11. KTH Dataset (Boxing class)
**[FINDING] Foundational but outdated; boxing class is shadow-boxing, low resolution, 25 subjects**

| Field | Detail |
|-------|--------|
| Full name | KTH Human Action Dataset (boxing action class) |
| Size | ~100 clips for boxing class (25 subjects × 4 scenarios) |
| Source | KTH University; controlled outdoor/indoor recording |
| Labels | Single class: "boxing" (shadow-boxing, no opponent) |
| Annotations | Class label only; no temporal segmentation within clips |
| Format | 160×120 px, 25 fps |
| Availability | Free: https://www.csc.kth.se/cvap/actions/ |
| License | Academic use |

[STAT:n] ~100 boxing clips
[LIMITATION] 160×120 resolution is too low for modern models. Single coarse "boxing" class with no punch-type discrimination. Shadow-boxing only. Useful only as a historical benchmark or domain-shift test.

---

## PART 2: RELATED SPORTS / TRANSFER LEARNING DATASETS

---

### 12. HMDB51 — "Punch" and "Kick Someone" classes
**[FINDING] 1,060 boxing-specific clips freely available; diverse Hollywood + YouTube sources**

| Field | Detail |
|-------|--------|
| Full name | HMDB: A Large Human Motion Database |
| Relevant classes | "punch" (boxing-relevant), "kick_person", "fencing", "sword_fight" |
| Boxing subset size | ~1,060 video clips (HMDB51-Boxing) |
| Total dataset | 6,766 clips, 51 classes |
| Sources | Movies, YouTube, sports broadcasts |
| Availability | Free: https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/ |
| License | Academic research |
| Transfer value | "punch" class directly maps to boxing strikes; diverse real-world conditions for pretraining |

[STAT:n] ~1,060 clips in punch/boxing subset; 6,766 total
[LIMITATION] Coarse "punch" label — no punch-type discrimination. Mixed context (street fights, sports, movies). Not temporally segmented within clips.

---

### 13. UCF101 — Boxing Punching Bag + Boxing Speed Bag
**[FINDING] Two dedicated boxing classes (~100 clips each); widely used pretraining backbone**

| Field | Detail |
|-------|--------|
| Full name | UCF101: Action Recognition Dataset |
| Relevant classes | "Boxing Punching Bag", "Boxing Speed Bag", "Punch", "Sumo Wrestling", "Fencing" |
| Boxing class size | ~100 clips each (25 groups × 4–7 clips per class per standard UCF structure) |
| Total dataset | 13,320 clips, 101 classes |
| Availability | Free: https://www.crcv.ucf.edu/data/UCF101.php |
| License | Academic research |
| Transfer value | Standard pretraining backbone; punching bag/speed bag captures arm extension + contact mechanics transferable to in-ring punch |

[STAT:n] ~200 boxing-specific clips; 13,320 total
[LIMITATION] Bag training ≠ sparring — no opponent, no defensive reaction. Coarse class labels. No temporal segmentation within clips.

---

### 14. Kinetics-400/600/700
**[FINDING] Large-scale pretraining with dedicated boxing + kickboxing + martial arts classes**

| Field | Detail |
|-------|--------|
| Full name | Kinetics Human Action Video Dataset (DeepMind) |
| Relevant classes (K-400) | "boxing", "kickboxing", "martial art", "punching bag", "punching person (boxing)", "high kick", "side kick", "wrestling", "arm wrestling", "tai chi", "sword fighting" |
| Clips per class | ~400–1,000 per class (minimum 400 guaranteed) |
| Total dataset | K-400: ~300K clips; K-600: ~500K; K-700: ~650K |
| Clip duration | ~10 seconds each |
| Availability | Download scripts: https://github.com/cvdfoundation/kinetics-dataset |
| License | CC BY 4.0 (K-400/600); check per-video for K-700 |
| Transfer value | "punching person (boxing)" directly relevant; massive scale enables strong backbone pretraining; kickboxing class adds kick+punch combos |

[STAT:n] ~4,000–11,000 boxing/combat-relevant clips across K-400 classes; 300K+ total
[LIMITATION] 10-second clips with clip-level labels only — no temporal segmentation of individual punches within clips. Not suitable for punch spotting without additional annotation.

---

### 15. NTU RGB+D 120
**[FINDING] Largest skeleton-action dataset; "punch/slap" and "kicking" classes; requires registration**

| Field | Detail |
|-------|--------|
| Full name | NTU RGB+D 120: Large-Scale Benchmark for 3D Human Activity Understanding |
| Relevant classes | A50 "punch/slap", A51 "kicking", A52 "pushing", A93 "shake fist", A102 "side kick" |
| Total dataset | 114,480 videos, 120 classes |
| Modalities | RGB video (1080p) + depth maps (512×424) + infrared + 3D skeletal (25 joints) |
| Availability | Registration required: https://rose1.ntu.edu.sg/dataset/actionRecognition/ |
| License | Academic non-commercial only |
| Transfer value | 3D skeleton modality directly applicable to pose-based punch classifiers; "punch/slap" class is directly transferable |

[STAT:n] 114,480 total; ~1,000 clips per class estimated for A50 "punch/slap"
[LIMITATION] "Punch/slap" is a mutual-action class performed in controlled lab setting — not boxing punches. No punch-type discrimination. Registration barrier. 1.3–2.3 TB storage requirement.

---

### 16. MADS (Martial Arts, Dancing, and Sports Dataset)
**[FINDING] 3D pose + depth for karate; 53,000 frames; good for skeleton model pretraining**

| Field | Detail |
|-------|--------|
| Full name | MADS: Martial Arts, Dancing, and Sports Dataset |
| Relevant content | Karate sequences (striking techniques), Tai-chi |
| Size | ~53,000 frames total; 5 action categories, 6 sequences each |
| Modalities | Multi-view RGB (3 cameras, 15 fps) + stereo depth (10–20 fps) + MoCap ground truth (60 fps) |
| Availability | Free download (~24 GB): http://visal.cs.cityu.edu.hk/downloads/mads-data-download/ |
| License | Academic research |
| Transfer value | Karate striking sequences provide 3D skeleton data for upper-body strike pretraining; MoCap ground truth enables accurate pose supervision |

[STAT:n] ~53,000 frames; 5 action categories
[LIMITATION] Karate strikes differ from boxing punches in hand position and footwork. No temporal event-level annotations (no start/end per strike). Small subject count.

---

### 17. TUHAD (Taekwondo Unit Technique Human Action Dataset)
**[FINDING] 1,936 samples of 8 taekwondo techniques; CC BY license; kick-heavy but contains arm strikes**

| Field | Detail |
|-------|--------|
| Full name | TUHAD: Taekwondo Unit Technique Human Action Dataset |
| Size | 1,936 action samples; ~100,000 image sequences |
| Subjects | 10 expert taekwondo practitioners |
| Labels | 8 unit techniques (poomsae forms; includes strikes and blocks) |
| Cameras | 2 camera views |
| Availability | Supplementary materials via MDPI Sensors journal |
| License | CC BY (Creative Commons Attribution) |
| Paper | MDPI Sensors 2020: https://www.mdpi.com/1424-8220/20/17/4871 |
| Transfer value | Striking biomechanics (hand strikes, blocks) partially overlap with boxing; multi-view captures 3D motion |

[STAT:n] 1,936 samples, 10 subjects, 8 technique classes
[LIMITATION] Taekwondo focuses on kicks (70%+ of techniques); arm strikes are different form from boxing. Poomsae (forms) are controlled, not reactive sparring.

---

### 18. FineGym
**[FINDING] Best-in-class hierarchical fine-grained action dataset (gymnastics); architectural template for boxing**

| Field | Detail |
|-------|--------|
| Full name | FineGym: A Hierarchical Video Dataset for Fine-Grained Action Understanding |
| Size | Large-scale gymnastic competition videos; multi-level temporal annotations |
| Labels | Hierarchical: event → sub-event → fine-grained element (e.g., balance beam → flight → specific skill) |
| Availability | https://sdolivia.github.io/FineGym/ |
| License | Academic research |
| Paper | CVPR 2020 (Oral): arXiv 2004.06704 |
| Transfer value | Not direct content transfer — but FineGym's hierarchical annotation scheme is the gold standard template for creating a fine-grained boxing dataset (combo → punch sequence → individual punch type) |

[STAT:n] Thousands of gymnastic clips with fine-grained temporal annotations
[LIMITATION] Gymnastics content has no direct transfer to boxing. Value is architectural/methodological, not pre-training content.

---

### 19. AVA (Atomic Visual Actions)
**[FINDING] Spatio-temporal bounding box dataset with "fight/hit a person" and "martial art" labels**

| Field | Detail |
|-------|--------|
| Full name | AVA: Atomic Visual Actions Dataset |
| Relevant classes | "fight/hit (a person)", "martial art" |
| Total dataset | ~430 clips (Hollywood movies), 80 action labels, 1.62M annotations |
| Annotations | Spatio-temporal bounding boxes at 1 fps keyframes |
| Availability | https://research.google.com/ava/ |
| License | CC BY 4.0 |
| Transfer value | Spatio-temporal detection format (person bounding box + action label) directly applicable to boxing punch detection; "hit person" class is combat-relevant |

[STAT:n] 1.62M spatio-temporal annotations; ~430 movie clips
[LIMITATION] Movie footage — staged, not real sports. 1 fps annotation density too coarse to capture individual punches (which last <0.5 sec). "Martial art" class is coarse.

---

### 20. FineSports (CVPR 2024)
**[FINDING] Hierarchical basketball dataset — irrelevant content but strongest architectural precedent for multi-person sports**

| Field | Detail |
|-------|--------|
| Full name | FineSports: A Multi-person Hierarchical Sports Video Dataset for Fine-Grained Action Understanding |
| Sport | Basketball (NBA game footage) |
| Size | 10,000 NBA game videos; 16,000 action instances; 123,000 spatio-temporal bounding boxes |
| Labels | 12 coarse + 52 fine-grained action types |
| Availability | Request-based (email xujinglinlove@gmail.com): https://github.com/PKU-ICST-MIPL/FineSports_CVPR2024 |
| License | Release Agreement required |
| Transfer value | Multi-person tracking + fine-grained hierarchical labeling + spatio-temporal boxes — exact methodology needed for boxing. PoSTAL model is directly adaptable. |

[STAT:n] 10,000 videos; 52 fine-grained classes; 123,000 bounding boxes
[LIMITATION] Basketball content — no striking. Method transfer only, not data transfer.

---

### 21. PhysPose Boxing Dataset (Feiz et al., 2025)
**[FINDING] Multi-camera 3D pose for elite boxing sparring — announced for release, not yet available**

| Field | Detail |
|-------|--------|
| Full name | PhysPose boxing multi-view dataset |
| Size | 20+ minutes of elite boxer sparring footage |
| Cameras | Sparse multi-view RGB setup |
| Annotations | 3D pose annotations (physics-based) |
| Availability | "Will be publicly released": https://hosseinfeiz.github.io/physpose/ |
| Paper | arXiv 2504.08175 (April 2025) |
| Transfer value | When released: first 3D pose boxing sparring dataset from elite athletes |

[LIMITATION] Not yet released. Monitor project page.

---

### 22. Martial Arts Gesture Dataset (Fine-Grained, Springer 2025)
**[FINDING] 8,790 images across 5 martial arts styles including Karate and Judo**

| Field | Detail |
|-------|--------|
| Size | 8,790 images |
| Styles | Karate, Tai Chi, Kung Fu, Judo, Taekwondo |
| Labels | Broad action category + specific gesture tag |
| Availability | Associated with Springer Nature paper (2025) |
| Transfer value | Karate/Kung Fu striking gesture images for classifier pretraining |

[STAT:n] 8,790 images, 5 martial arts categories
[LIMITATION] Images only (no video/temporal information). Controlled poses, not reactive sparring.

---

## PART 3: COMPARATIVE ASSESSMENT

### Dataset Comparison Matrix

| Dataset | Clips/Samples | Punch Types | Temporal Annot. | Pose Annot. | License | Available Now |
|---------|--------------|-------------|-----------------|-------------|---------|---------------|
| **BoxingVI** | 6,915 clips | 6 | Yes (frame-level) | Yes (2D AlphaPose) | Fair Use | Yes |
| **Olympic Boxing (Kaggle)** | 312,774 frames | 8 | Yes (frame-level) | No | Non-commercial | Yes |
| **BoxMAC** | 2,314 clips | 6 (+7 other) | Yes | No | None (withdrawn) | No |
| **ShadowPunch** | Undisclosed | Multiple | Yes (event spot.) | Yes (keypoints) | CC BY 4.0 | Yes (OpenReview) |
| **FACTS Boxing** | 8,000 clips | 8 | Yes | No | Unspecified | Partial |
| **Punch_DL** | 240 clips | 7 | No | Yes (keypoints) | None | Yes (Drive) |
| **KTH Boxing** | ~100 clips | 1 (coarse) | No | No | Academic | Yes |
| **HMDB51 Punch** | ~1,060 clips | 1 (coarse) | No | No | Academic | Yes |
| **UCF101 Boxing** | ~200 clips | 1 (coarse) | No | No | Academic | Yes |
| **Kinetics-400** | ~4,000+ relevant | 5+ classes | No | No | CC BY 4.0 | Yes |
| **NTU RGB+D 120** | ~1,000 punch | 1 (coarse) | No | Yes (3D skel.) | Non-commercial | Registration |
| **MADS** | 53K frames | N/A (karate) | No | Yes (MoCap) | Academic | Yes |
| **TUHAD** | 1,936 samples | 8 techniques | No | No | CC BY | Yes |

---

## PART 4: IS BoxingVI THE BEST STARTING POINT?

**[FINDING] BoxingVI is the best single starting point for a pose-conditioned punch classifier, but the Olympic Boxing Kaggle dataset is superior for frame-level detection, and a combined approach outperforms either alone.**

### BoxingVI Strengths
1. Only publicly available dataset combining RGB + temporal segmentation + punch-type labels + 2D pose trajectories simultaneously
2. Covers real sparring (not shadow boxing, not bag work) — highest ecological validity for match analysis
3. Fine-grained 6-class taxonomy matches the standard boxing ontology
4. Government-supported (Ministry of Youth Affairs and Sports) — likely to persist and expand
5. Pre-segmented clips simplify training pipeline

### BoxingVI Weaknesses
1. Only 6,915 clips — small by modern standards (Kinetics has 400-1000 clips per class alone)
2. YouTube sparring only — no professional match footage, variable quality
3. Max 25 frames per punch — may truncate preparatory and follow-through phases
4. Fair Use legal basis is fragile for commercial applications

### When to prefer the Olympic Boxing (Kaggle) dataset instead
- For **frame-level punch detection** (is there a punch in this frame?) — 312,774 annotated frames with bounding boxes is far more appropriate
- For **temporal action localization** training — dense frame-level labels enable better boundary modeling
- For **body-punch and block detection** — 8 classes including body shots and blocks, absent from BoxingVI

### Recommended Combined Strategy

**[FINDING] The optimal training pipeline uses three datasets in combination:**

```
Stage 1 — Backbone Pretraining:
  Kinetics-400 ("punching person (boxing)", "boxing", "kickboxing" classes)
  → Establishes strong visual priors for punch motion

Stage 2 — Fine-grained Pretraining:
  Olympic Boxing Kaggle dataset (frame-level) for detection head
  + BoxingVI (clip-level) for punch-type classification head

Stage 3 — Fine-tuning:
  Target domain footage (your own data if available)
  ShadowPunch for event-spotting validation (CC BY 4.0 — most permissive)
```

[STAT:n] Combined: ~330,000 labeled frames + 6,915 clips + 300K+ Kinetics clips
[STAT:effect_size] FACTS paper demonstrates 83.25% accuracy on 8-class boxing using Olympic dataset; BoxingVI paper reports 91% on 6-class using pose; combining modalities typically improves 5–15% in multi-task settings

---

## PART 5: GAPS AND RECOMMENDATIONS

**[FINDING] Four critical data gaps exist that no current public dataset addresses:**

1. **Professional match footage** — All datasets use amateur/training footage or shadow boxing. No dataset covers elite professional bouts with broadcast-quality multi-angle coverage and punch-level annotations.

2. **Combo-level temporal structure** — No dataset annotates punch *sequences* (e.g., jab-cross-hook combos) as compound actions. All labels are at single-punch level.

3. **Defensive action integration** — Only BoxMAC (withdrawn) includes slips, ducks, and guards alongside punches. No available dataset supports joint offensive+defensive action recognition.

4. **Multi-person synchronized tracking** — No available dataset provides synchronized punch annotations for both boxers simultaneously (BoxMAC attempted this but is withdrawn; PhysPose boxing dataset announced but unreleased).

**Recommended new annotation effort:** The unlabeled Olympic Boxing Video Dataset (Kaggle) contains ~4 hours of professional-grade footage that could be re-annotated with temporal punch boundaries and fine-grained labels to fill gap #1.

---

## [LIMITATION]

- Web search coverage is bounded by public indexing; private/institutional datasets not captured
- Exact clip counts for Kinetics combat-relevant classes based on structural minimum (400/class); per-class breakdowns require downloading annotation CSVs
- BoxMAC and PhysPose status may have changed after March 2026 search date
- ShadowPunch exact dataset size requires accessing OpenReview supplementary ZIP
- HMDB51 boxing subset size (1,060) sourced from a 2025 secondary paper; verify against original distribution

---

## Sources

- [BoxingVI arXiv](https://arxiv.org/html/2511.16524v1)
- [BoxMAC arXiv](https://arxiv.org/abs/2412.18204)
- [ShadowPunch OpenReview](https://openreview.net/forum?id=Jq8HYNZG9s)
- [FACTS arXiv](https://arxiv.org/html/2412.16454v1)
- [Olympic Boxing Kaggle](https://www.kaggle.com/datasets/piotrstefaskiue/olympic-boxing-punch-classification-video-dataset)
- [Punch_DL GitHub](https://github.com/balezz/Punch_DL)
- [BoxNet IEEE](https://ieeexplore.ieee.org/document/10327379/)
- [BoxingPro MDPI](https://www.mdpi.com/2079-9292/14/21/4155)
- [EPFL Boxing WIP](https://www.epfl.ch/labs/cvlab/data/boxing-multiview-interaction-dataset/)
- [Fine-Grained Depth Boxing ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1077314217300668)
- [KTH Dataset](https://www.csc.kth.se/cvap/actions/)
- [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
- [Kinetics Dataset](https://github.com/cvdfoundation/kinetics-dataset)
- [NTU RGB+D 120](https://rose1.ntu.edu.sg/dataset/actionRecognition/)
- [MADS Dataset](http://visal.cs.cityu.edu.hk/downloads/mads-data-download/)
- [TUHAD Sensors 2020](https://www.mdpi.com/1424-8220/20/17/4871)
- [FineGym CVPR 2020](https://sdolivia.github.io/FineGym/)
- [AVA Dataset](https://research.google.com/ava/)
- [FineSports CVPR 2024](https://github.com/PKU-ICST-MIPL/FineSports_CVPR2024)
- [PhysPose arXiv](https://arxiv.org/html/2504.08175v1)
- [AI in Martial Arts Survey](https://journals.sagepub.com/doi/10.1177/17543371241273827)
- [Boxing Punch Detection Single Camera MDPI](https://www.mdpi.com/1099-4300/26/8/617)
- [Hierarchical Punch Pipeline Springer 2025](https://link.springer.com/chapter/10.1007/978-3-031-93688-3_19)
- [Kinetics Action Class Labels GitHub](https://gist.github.com/willprice/f19da185c9c5f32847134b87c1960769)
