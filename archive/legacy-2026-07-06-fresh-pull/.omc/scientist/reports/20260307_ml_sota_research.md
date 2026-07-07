# ML State-of-the-Art Research Report
**Date:** 2026-03-07
**Scope:** Boxing/combat sports AI — validated current information as of 2025-2026
**Method:** Web search + primary source retrieval across 5 research domains

---

## [OBJECTIVE]
Survey the current state of the art for five ML domains relevant to a boxing training application: (1) combat sports action recognition, (2) real-time pose estimation, (3) VLMs for sports video analysis, (4) LLM-powered coaching/drill generation, and (5) action detection terminology. Identify the current best approach, key papers/tools, and practical solo-developer feasibility for each.

---

## Domain 1: Boxing/Combat Sports Action Recognition from Video

### [DATA]
- Primary benchmark: BoxingVI (arXiv:2511.16524, Nov 2024) — 6,915 labeled punch clips, 6 punch types, 18 athletes, YouTube sparring footage
- Secondary benchmark: BoxMAC (arXiv:2412.18204, Dec 2024) — multi-label boxing action classification dataset
- Active research from Springer Nature, Nature Scientific Reports, PLOS ONE, PMC — all 2025

### Current Best Approach

**The field is NOT monolithic.** Two parallel streams co-exist:

**Stream A — Pose/Skeleton-Based (still dominant for structured punch classification):**
- Extract 2D/3D joint keypoints (MediaPipe, BlazePose, or physics-based trackers)
- Feed temporal sequences to lightweight classifiers: XGBoost, LSTM, GRU, or GCN
- Accuracy: 89–92% for binary hand classification (lead vs. rear punch) with active learning at only 15% labeling effort
- 2025 paper (PLOS ONE, PMC:12061147): Query-by-Committee active learning achieves 91.41% rear-hand, 91.91% lead-hand punch accuracy
- Still the standard for real-time, on-device, resource-constrained deployments

**Stream B — Video Foundation Models (emerging for richer understanding):**
- VideoMAE V2 (CVPR 2023 baseline, still dominant backbone as of 2025): 87.4% on Kinetics-400
- InternVideo2 (arXiv:2403.15377): SOTA across 60+ video/audio tasks via multi-stage web+YouTube pre-training
- InternVideo-Next (arXiv:2512.01342, Dec 2025): General video foundation models without video-text supervision
- 3D-ResNet used in boxing-specific work (Nature 2025) for spatiotemporal feature extraction
- These models require GPU for training; inference on CPU is feasible for non-real-time use

**Stream C — Physics-Based Pose (new, 2025):**
- "Multi-person Physics-based Pose Estimation for Combat Sports" (arXiv:2504.08175, Apr 2025)
- Trajectory optimization handles rapid motions, occlusions, and close interactions specific to boxing
- New benchmark of elite boxing footage released alongside

**Key 2025 Papers:**
- BoxingVI benchmark: https://arxiv.org/html/2511.16524v1
- Physics-based combat pose: https://arxiv.org/html/2504.08175v1
- Active learning boxing: https://pmc.ncbi.nlm.nih.gov/articles/PMC12061147/
- Contrastive learning + CBAM for boxing: https://link.springer.com/article/10.1007/s11760-025-05018-2
- Multimodal (3D-ResNet + BERT psychology): https://www.nature.com/articles/s41598-025-34771-0

### [FINDING 1] Pose-based classification remains the standard for real-time punch recognition; video foundation models are emerging but not yet dominant for on-device deployment.
[STAT:effect_size] Active learning pose classifier: 91.4–91.9% accuracy on punch classification
[STAT:n] BoxingVI: 6,915 clips; active learning study: achieved with 15% of typical labeling effort
[STAT:p_value] Not reported in summaries; peer-reviewed in PLOS ONE (2025)

### [FINDING 2] Video foundation models (VideoMAE, InternVideo2) achieve SOTA on general benchmarks but have not yet displaced pose-based approaches in boxing-specific literature.
[STAT:effect_size] VideoMAE V2: 87.4% top-1 on Kinetics-400 (general action recognition)
[STAT:n] Kinetics-400: ~306,000 clips across 400 classes

### [LIMITATION]
- Most boxing-specific papers still use small, lab-controlled datasets
- VideoMAE fine-tuning on <500 clips causes immediate overfitting (GitHub issue #129 confirmed)
- Physics-based pose estimation (arXiv:2504.08175) is very new; no wide adoption yet
- No direct apples-to-apples comparison between pose-classifier and video-transformer on the same boxing dataset was found in the literature as of this search

---

## Domain 2: Real-Time Pose Estimation for Sports Training Apps

### [DATA]
- Sources: Roboflow (2025), Kite Metric comparison, PMC narrative review (PMC:11566680), multiple benchmark papers
- Models evaluated: MediaPipe BlazePose, MoveNet Lightning/Thunder, YOLOv11 Pose, YOLOv8 Pose, OpenPose

### Current Best Options

| Model | Latency (mobile) | Keypoints | Multi-person | Best For |
|---|---|---|---|---|
| MoveNet Lightning | <7ms @ 192×256 | 17 | No | Latency-critical mobile |
| MoveNet Thunder | ~20ms @ 256×256 | 17 | No | Accuracy + mobile |
| MediaPipe BlazePose | 30+ FPS on mid-range phone | 33 | No (single-person) | Sports coaching apps |
| YOLOv11 Pose | GPU-dependent | 17 | Yes | Multi-person, server |
| YOLOv8 Pose | GPU-dependent | 17 | Yes | Dense/occluded scenes |

### [FINDING 3] MediaPipe BlazePose remains the practical standard for single-person sports coaching apps in 2025; MoveNet Lightning is preferred when latency is the primary constraint.
[STAT:effect_size] MoveNet Lightning: <7ms inference on mobile; MediaPipe: 30+ FPS on mid-range devices
[STAT:n] Based on multiple independent benchmark evaluations (Roboflow, Kite Metric, PMC review 2025)

### [FINDING 4] YOLOv11 Pose (89.4% mAP@0.5 on COCO Keypoints) surpasses YOLOv8 in accuracy with 22% fewer parameters, but requires GPU for real-time use.
[STAT:effect_size] YOLOv11m: 89.4% mAP@0.5 on COCO Keypoints
[STAT:n] COCO Keypoints benchmark (standard)

**Practical Recommendation for Boxing App:**
- Single boxer on-device: MediaPipe BlazePose (33 keypoints, well-supported Python/JS SDK, IMAGE mode stable)
- Two boxers (sparring): YOLOv8/v11 Pose on server-side, or MediaPipe run twice with ROI crops
- The project's existing MediaPipe IMAGE mode setup is correctly aligned with 2025 practice

### [LIMITATION]
- All fast on-device models (MoveNet, BlazePose) are single-person; multi-person boxing requires server-side inference or creative ROI cropping
- Occlusion during clinches and grappling still causes keypoint failures in all 2D models
- 3D pose (depth) requires specialized hardware (LiDAR, stereo) or 3D lifting models with added latency

---

## Domain 3: VLMs for Sports Video Analysis

### [DATA]
- Sources: GameRun AI architecture analysis, Gemini 2.5 video understanding blog, Towards AI MMA use case (Feb 2026), ICCV 2025 VLM4D paper, Moonshine VLM deployment guide
- Models: Gemini 2.5 Pro, GPT-4.1, GPT-4V (legacy)

### Current Capabilities

**Gemini 2.5 Pro (current SOTA for video):**
- Context window: 1M+ tokens; video processing up to ~45 minutes
- Default frame sampling: **1 FPS** (critical limitation for fast sports)
- Leads LMArena and video understanding benchmarks vs. GPT-4.1
- Processes video from file upload or YouTube URL
- Confirmed use case: MMA footage analysis with structured captioning (Towards AI, Feb 2026)

**GPT-4.1:**
- Video via frame sampling; handles ~30 minutes depending on compression
- Strong at visual question answering, object counting
- Weaker than Gemini 2.5 Pro on temporal video understanding benchmarks

**Hard limitations for boxing:**
- Default 1 FPS sampling misses sub-second punch events (a jab takes ~150-300ms)
- Higher frame sampling = proportionally more tokens = higher cost + latency
- Models lack biomechanical ground truth; they describe what they see, not kinematic correctness
- No real-time capability — all current VLMs require upload → API call → response cycle

**What VLMs can do for boxing (validated):**
- Describe visible tactics ("fighter A circled left after combo")
- Identify general punch types from clear video
- Generate structured round-by-round summaries
- Answer questions about fighter positioning and patterns

**What VLMs cannot reliably do:**
- Frame-precise temporal localization of punch start/end
- Kinematic analysis (wrist angle, hip rotation, shoulder drop)
- Differentiate a jab from a cross in low-quality or occluded footage
- Real-time feedback

### [FINDING 5] VLMs can produce tactical narrative analysis of boxing footage but are fundamentally unsuitable for real-time or precise temporal punch detection due to 1 FPS default sampling and API latency.
[STAT:effect_size] Gemini 2.5 Pro: SOTA on video understanding benchmarks (ICCV 2025 evaluation)
[STAT:n] Confirmed by Google Developers Blog (Gemini 2.5 video understanding, 2025); MMA use case (Towards AI, Feb 2026)

### [FINDING 6] A hybrid architecture — pose classifier for punch detection + VLM for tactical narrative — is the architecturally sound approach for a boxing coaching app.
[STAT:n] Supported by GameRun AI architectural analysis; Towards AI MMA multi-agent workflow (Feb 2026)

### [LIMITATION]
- VLM video capabilities are evolving rapidly; frame sampling limits may improve in 2026
- Cost at higher frame rates ($0.002–0.015/minute of video at 1fps, higher at more fps) may be prohibitive for frequent analysis
- No peer-reviewed evaluations specifically on boxing punch classification accuracy with VLMs found

---

## Domain 4: LLM-Powered Coaching and Drill Generation

### [DATA]
- Primary: GPTCoach (Stanford HCI, CHI 2025) — peer-reviewed, published at top HCI venue
- Secondary: LLM-SPTRec (Nature Scientific Reports, 2026) — knowledge-graph augmented sports training plan generation
- Tertiary: Half-marathon LLM coach (arXiv:2509.26593), Multiple Medium/blog practical implementations
- Evaluation: JMIR scoping review on LLM exercise coaching strategies (PMC:12520646)

### Current Approaches

**GPTCoach (CHI 2025 — Stanford, most rigorous study):**
- LLM chatbot implementing Active Choices evidence-based health coaching program
- Uses motivational interviewing (MI) counseling strategies
- Tool use: queries and visualizes wearable device health data
- Evaluation: n=16, lab study, 3 months historical tracking data
- Result: MI-consistent or neutral behavior 93% of the time; outperformed vanilla GPT-4 on MI-consistency
- Participants rated guidance as highly personalized and actionable

**LLM-SPTRec (Nature 2026):**
- LLM augmented with domain-specific Sports Science Knowledge Graph
- Integrates multi-source user data for personalization
- Generates structured training plans with progressive overload, recovery, specificity principles

**Practical implementations confirmed:**
- Running coach: weekly plan generation from Strava/Garmin data, WhatsApp + Google Calendar delivery
- Triathlon planning: availability + goal-based weekly scheduling
- Half-marathon: 9-week roadmap with base building, endurance extension, taper phases

**Drill generation pattern (validated approach):**
1. Input: session data + athlete profile + goals + constraints
2. LLM generates: weekly plan with progressive overload
3. Feedback loop: post-session data → LLM adjusts next week
4. Wearable integration: heart rate, load, recovery data as context

### [FINDING 7] LLM-based coaching is an active, validated research area. The GPTCoach approach (MI + tool use + wearable data) achieves 93% MI-consistent behavior and is rated highly personalized by users.
[STAT:effect_size] 93% MI-consistent or neutral behavior codes
[STAT:n] n=16 lab study; peer-reviewed CHI 2025
[STAT:p_value] Counterfactual comparison vs. vanilla GPT-4 shows significant improvement in MI-consistency

### [FINDING 8] Knowledge-graph-augmented LLMs (LLM-SPTRec) outperform plain LLMs for sports training plan generation by grounding outputs in sports science principles.
[STAT:n] Published Nature Scientific Reports 2026; peer-reviewed

### [LIMITATION]
- LLM-generated plans cannot account for individual physiological adaptation, injury response, or athlete safety without human expert supervision (PMC evaluation confirmed)
- All studies use general fitness or endurance sports; no peer-reviewed boxing-specific LLM coaching study found
- Fine-tuning on behavioral psychology data (PMC:12454129) improves adherence but requires labeled dialogue datasets

**Solo Developer Feasibility: HIGH.** GPTCoach is open-source (GitHub: StanfordHCI/GPTCoach-CHI2025). The pattern of "feed session data → LLM → generate next session plan" is implementable with any LLM API in a weekend. The hard part is the session data capture, not the LLM integration.

---

## Domain 5: Action Detection Terminology

### [DATA]
- Sources: MDPI Electronics 2025, Springer AI Review 2023 (still definitive), ACM Computing Surveys 2024, ICCV 2025 BinEgo-360 Challenge

### Validated Terminology Map

```
Video Understanding (umbrella term)
├── Action Recognition / Action Classification
│   └── Given a TRIMMED clip → predict class label
│       Examples: "this is a jab", "this is a cross"
│
├── Temporal Action Detection (TAD) / Temporal Action Localization (TAL)
│   └── Given an UNTRIMMED video → find [start, end] + class for each action instance
│       Evaluation: mAP across multiple tIoU thresholds (standard protocol)
│       This is what "action detection" means in the literature
│
├── Action Segmentation
│   └── Frame-by-frame labeling of action classes across the full video
│
└── Action Prediction / Anticipation
    └── Predict future actions before they are complete
```

**Key distinctions:**
- "Action recognition" = clip-level classification (trimmed input)
- "Action detection" = temporal localization in untrimmed video (find WHEN things happen)
- TAD is a **subtask** of video understanding, not a synonym for it
- The term used in ICCV 2025 challenge: "Temporal Action Localization (TAL)" — TAL and TAD are used interchangeably in current literature

**For a boxing app, the correct framing is:**
- Classifying a pre-segmented punch clip: **Action Recognition**
- Finding where punches occur in a full sparring video: **Temporal Action Detection / Localization**
- Both together: **Temporal Action Localization + Classification** (the full TAD pipeline)

### [FINDING 9] "Action detection" in ML literature specifically means Temporal Action Detection — locating action instances in untrimmed video with start/end timestamps + class labels. It is a subtask of video understanding.
[STAT:n] Confirmed by: Springer AI Review survey; ACM Computing Surveys 2024; ICCV 2025 BinEgo-360 TAL challenge; MDPI Electronics 2025

### [LIMITATION]
- Terminology is used inconsistently in blog posts and product descriptions; in peer-reviewed literature the above definitions are stable
- "Spatial-temporal action detection" (finding WHERE in frame + WHEN in time) is a separate, harder task not yet practical for solo deployment

---

## Summary Table: Solo Developer Feasibility

| Domain | Current Best Approach | Solo Dev Feasibility | Notes |
|---|---|---|---|
| Boxing action recognition | Pose keypoints → XGBoost/LSTM | HIGH | Current project approach is correct |
| Video foundation models | VideoMAE fine-tune | MEDIUM | Needs GPU, overfitting risk <500 clips |
| Physics-based combat pose | arXiv:2504.08175 | LOW | Very new, no SDK |
| Real-time pose estimation | MediaPipe BlazePose | HIGH | IMAGE mode, Python SDK, stable |
| Multi-person pose | YOLOv11 server-side | MEDIUM | Needs GPU inference server |
| VLM tactical analysis | Gemini 2.5 Pro API | HIGH | API call, 1fps limit, cost concern |
| LLM drill generation | GPT-4o/Claude + session data | HIGH | Open-source reference: GPTCoach |
| TAD (finding punches in video) | Pose-based sliding window | MEDIUM | No off-the-shelf boxing TAD model |

---

## Strategic Recommendation for Atom Project

Based on validated 2025-2026 evidence:

1. **Keep the XGBoost pose-based classifier** — it is scientifically sound, aligns with published SOTA for boxing-specific classification, and achieves 89–92% accuracy in comparable work.

2. **Add Temporal Action Detection on top** — use a sliding window over MediaPipe pose sequences to locate punch events in untrimmed sparring video. This is the correct ML framing for "finding punches in a sparring session."

3. **VLM for narrative, not detection** — Gemini 2.5 Pro can analyze a full sparring session video and produce round-by-round tactical summaries. Feed it sampled frames at 2-4fps with structured prompts. Use this for post-session coaching narratives, not real-time feedback.

4. **LLM drill generation is well-validated** — Use GPTCoach's open-source architecture (Stanford HCI) as the reference. The key insight: feed structured session output (punch counts, accuracy, round performance) as context to the LLM for next-session plan generation.

5. **The hybrid architecture is the right one:** MediaPipe (pose) → XGBoost (punch classification) → TAD (temporal localization) → LLM (coaching narrative + drill generation) → (optionally) Gemini (tactical video summary).

---

## Sources

- [BoxingVI Benchmark (arXiv Nov 2024)](https://arxiv.org/html/2511.16524v1)
- [Physics-Based Combat Pose Estimation (arXiv Apr 2025)](https://arxiv.org/html/2504.08175v1)
- [Active Learning Boxing Punch Recognition (PMC/PLOS ONE 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12061147/)
- [Contrastive Learning + CBAM for Boxing Video (Springer 2025)](https://link.springer.com/article/10.1007/s11760-025-05018-2)
- [Multimodal Deep Learning for Boxing Action Recognition (Nature 2025)](https://www.nature.com/articles/s41598-025-34771-0)
- [BoxMAC Multi-Label Boxing Dataset (arXiv Dec 2024)](https://arxiv.org/html/2412.18204v1)
- [InternVideo2 (arXiv 2024)](https://arxiv.org/abs/2403.15377)
- [InternVideo-Next (arXiv Dec 2025)](https://arxiv.org/html/2512.01342)
- [Roboflow: Best Pose Estimation Models 2025](https://blog.roboflow.com/best-pose-estimation-models/)
- [Kite Metric: BlazePose vs MoveNet vs YOLOv11](https://kitemetric.com/blogs/open-source-pose-detection-a-deep-dive-into-blazepose-movenet-and-yolov11)
- [PMC Narrative Review: ML Pose Estimation Models (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11566680/)
- [GameRun AI: VLMs for Sports Architecture](https://gamerun.ai/blog/the-spatio-temporal-frontier-architecting-vlms-for-the-unstructured-realism-of-sports)
- [Gemini 2.5 Video Understanding Blog (Google Developers)](https://developers.googleblog.com/en/gemini-2-5-video-understanding/)
- [Towards AI: MMA Analysis with Gemini (Feb 2026)](https://pub.towardsai.net/structured-video-captioning-with-gemini-an-mma-analysis-use-case-bfbb8fd91a26)
- [VLM4D: Spatiotemporal Awareness in VLMs (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhou_VLM4D_Towards_Spatiotemporal_Awareness_in_Vision_Language_Models_ICCV_2025_paper.pdf)
- [GPTCoach: LLM-Based Physical Activity Coaching (CHI 2025)](https://dl.acm.org/doi/10.1145/3706598.3713819)
- [GPTCoach arXiv](https://arxiv.org/abs/2405.06061)
- [GPTCoach GitHub](https://github.com/StanfordHCI/GPTCoach-CHI2025)
- [LLM-SPTRec: Knowledge-Grounded Sports Training Plan Generation (Nature 2026)](https://www.nature.com/articles/s41598-026-37075-z)
- [Half Marathon LLM Coach (arXiv 2025)](https://arxiv.org/html/2509.26593v1)
- [JMIR: Evaluation Strategies for LLM Exercise Coaching (2025)](https://www.jmir.org/2025/1/e79217)
- [MDPI: Adaptive Temporal Action Localization (2025)](https://www.mdpi.com/2079-9292/14/13/2645)
- [Springer: Overview of Temporal Action Detection (2023, still canonical)](https://link.springer.com/article/10.1007/s10462-023-10650-w)
- [VideoMAE Fine-tuning Overfitting Issue (GitHub)](https://github.com/MCG-NJU/VideoMAE/issues/129)
- [FACTS: Fine-Grained Action Classification for Tactical Sports (arXiv Dec 2024)](https://arxiv.org/html/2412.16454v1)
