# ATOM — Technical Specification

> PRD에서 약속한 것을 기술로 구현하기 위한 문서

Version 1.0 | February 2026

---

## 1. Technical Objective

이 제품이 기술적으로 해야 하는 것은 세 가지입니다:

1. **영상에서 "이 사람이 뭘 쳤는지" 알아내기**
2. **그 데이터로 "다음에 뭘 시킬지" 결정하기**
3. **시간이 지나면서 "뭐가 달라졌는지" 보여주기**

나머지(앱, 서버, TTS)는 이 데이터 엔진 위에서 돌아가는 껍데기입니다.

---

## 2. Data Engine

### 2.1 Analysis Pipeline

영상에서 복싱 데이터를 추출하는 핵심 파이프라인입니다.

```
영상 입력
  ↓
① 포즈 추출 — 프레임마다 관절 좌표 추출
  ↓
② 동작 분류 — 관절 움직임에서 펀치/방어 동작 판단
  ↓
③ 시퀀스 인식 — 연속 동작을 콤보로 묶기
  ↓
④ 세션 매칭 — TTS 지시와 실제 수행 비교
  ↓
구조화된 데이터 출력
```

#### Stage 1: 포즈 추출

| 항목 | 선택 |
|------|------|
| 모델 | MediaPipe Pose |
| 키포인트 | 33개 (전신) |
| 실행 위치 | 서버 (세션 후 처리) |
| 입력 | 영상 파일 (mp4) |
| 출력 | 프레임별 33 키포인트 × 3좌표 (x, y, z) + confidence |

**왜 서버에서 하는가:** 세션 중 실시간 분석이 필요 없습니다. 세션 중에는 TTS만 돌리고, 세션 끝나면 영상을 서버로 보내서 분석합니다. 모바일에서 촬영 + 포즈 추출을 동시에 하면 배터리/발열 이슈가 있습니다.

**전처리:**
- confidence가 낮은 프레임 제거 (threshold: 0.5)
- 키포인트 좌표 정규화 (사람 크기/위치 보정)
- 프레임 레이트 통일 (30fps)

#### Stage 2: 동작 분류

| 항목 | 선택 |
|------|------|
| 모델 | CTR-GCN (Channel-wise Topology Refinement GCN) |
| 입력 | 키포인트 시계열 (윈도우 단위) |
| 출력 | 동작 클래스 + confidence |
| 실행 위치 | 서버 (GPU) |

**왜 CTR-GCN인가:**

CTR-GCN은 사람의 관절 연결 구조(그래프)를 이해하고, 관절 간 관계를 학습합니다. 복싱에서 "오른손목 + 오른팔꿈치 + 오른어깨 + 오른엉덩이가 연쇄적으로 회전" 같은 패턴을 잡아내는 데 강합니다.

단순 1D CNN 대비 장점:
- 비슷한 동작 구분에 강함 (잽 vs 크로스, 리드훅 vs 리어훅)
- 관절 연결 관계를 고정하지 않고 학습 — 복싱에 최적화된 관계를 스스로 찾음
- 동작 종류가 늘어나도 성능 유지 (확장성)

**분류 클래스 (MVP):**

| 클래스 | 설명 |
|--------|------|
| jab | 앞 손 직선 펀치 |
| cross | 뒷 손 직선 펀치 |
| lead_hook | 앞 손 훅 |
| rear_hook | 뒷 손 훅 |
| lead_uppercut | 앞 손 어퍼컷 |
| rear_uppercut | 뒷 손 어퍼컷 |
| body_shot | 바디 공격 (높이 기반 구분) |
| slip | 슬립 (Phase 2) |
| duck | 덕킹 (Phase 2) |
| backstep | 백스텝 (Phase 2) |
| idle | 동작 없음 / 가드 |

**슬라이딩 윈도우:**
- 윈도우 크기: 30프레임 (1초)
- 스트라이드: 5프레임
- 각 윈도우마다 동작 분류 + confidence 출력
- confidence threshold: 0.7 이상만 유효한 동작으로 채택

#### Stage 3: 시퀀스 인식

| 항목 | 선택 |
|------|------|
| 방식 | Rule-based |
| 로직 | 시간 간격 기반 콤보 묶기 |

```
규칙:
- 동작 사이 간격 < 0.8초 → 같은 콤보
- 동작 사이 간격 ≥ 0.8초 → 콤보 종료, 새 콤보 시작
- idle이 0.5초 이상 → 콤보 종료
```

출력: `[{combo: ["jab", "cross", "lead_hook"], start_time: 12.3, end_time: 13.1}]`

**왜 Rule-based인가:** 시퀀스 인식은 "이미 분류된 동작을 묶는 것"이라 복잡한 모델이 필요 없습니다. 시간 간격만으로 정확하게 끊을 수 있습니다.

#### Stage 4: 세션 매칭

| 항목 | 선택 |
|------|------|
| 방식 | Rule-based |
| 입력 | TTS 지시 로그 + 인식된 콤보 시퀀스 |
| 출력 | 매칭 결과 (성공/실패/부분성공) |

```
로직:
1. TTS 지시 타임스탬프를 기준으로 매칭 윈도우 설정 (지시 후 3초)
2. 해당 윈도우 내에서 인식된 콤보를 추출
3. 지시한 콤보와 비교:
   - 완전 일치: success
   - 부분 일치 (일부 펀치 누락/추가): partial
   - 불일치: miss
```

예시:
```
지시: "원-투-바디" (t=12.0)
인식: ["jab", "cross", "lead_hook"] (t=12.3~13.1)
매칭: partial — 바디 대신 훅이 나감
```

### 2.2 Session Generation Engine

사용자 데이터를 기반으로 다음 훈련 세션을 생성합니다.

#### LLM 기반 세션 생성

| 항목 | 선택 |
|------|------|
| 모델 | Claude Haiku / GPT-4o-mini |
| 호출 시점 | 세션 시작 전 1회 |
| 예상 비용 | ~$0.005/세션 (입력 ~1500 토큰, 출력 ~1000 토큰) |

**3-Layer Context 구조:**

LLM에 전달하는 사용자 히스토리를 3단계로 압축합니다.

**Layer 1: User Profile (항상 전달, ~500 토큰)**

사용자의 현재 상태 스냅샷. 매 세션 후 업데이트.

```json
{
  "experience": "8개월",
  "total_sessions": 23,
  "current_project": {
    "name": "바디 콤보 확장",
    "started_at": "2026-02-10",
    "sessions_in_project": 5
  },
  "combo_mastery": {
    "원-투": {"status": "mastered", "drill_rate": 0.95, "shadow_rate": 0.40},
    "원-투-쓰리": {"status": "proficient", "drill_rate": 0.82, "shadow_rate": 0.25},
    "원-투-바디": {"status": "learning", "drill_rate": 0.45, "shadow_rate": 0.05},
    "잽-바디-잽-크로스": {"status": "learning", "drill_rate": 0.20, "shadow_rate": null},
    "원-투-슬립-크로스": {"status": "new", "drill_rate": null, "shadow_rate": null}
  },
  "shadow_stats": {
    "defense_reaction_rate": 0.60,
    "response_diversity": 0.35,
    "common_pattern": "상대 잽에 매번 백스텝"
  },
  "strengths": ["직선 펀치 정확도", "기본 콤보 안정성"],
  "weaknesses": ["바디샷 부재", "반응 단조로움"]
}
```

**Layer 2: Recent Context (항상 전달, 최근 3세션, ~500 토큰)**

```json
{
  "recent_sessions": [
    {
      "date": "2/27",
      "type": "drill",
      "focus": "바디 콤보",
      "key_results": "원-투-바디 성공률 30%→45%. 바디훅 지시 시 헤드훅으로 간 경우 7회.",
      "feedback_given": "바디를 의식적으로 노리라고 피드백"
    },
    {
      "date": "2/25",
      "type": "shadow",
      "focus": "상황 대응",
      "key_results": "방어 반응률 60%. 추격 찬스에서 잽만 사용.",
      "feedback_given": "드릴에서 배운 콤보를 쉐도우에서 꺼내보라고 피드백"
    }
  ]
}
```

**Layer 3: Long-term Trends (프로젝트 완료/전환 시에만, ~300 토큰)**

```json
{
  "weekly_summary": [
    {"week": "W1", "sessions": 3, "new_combos_learned": 2, "avg_success": 0.45},
    {"week": "W2", "sessions": 2, "new_combos_learned": 1, "avg_success": 0.55},
    {"week": "W3", "sessions": 3, "new_combos_learned": 1, "avg_success": 0.62}
  ],
  "trend": "성공률 상승 중. 새 콤보 도입 속도 안정적."
}
```

#### LLM 출력 형식

**콤비네이션 드릴 세션:**

```json
{
  "session_type": "drill",
  "total_rounds": 4,
  "round_duration": 180,
  "rest_duration": 30,
  "focus": "바디 콤보 확장",
  "focus_message": "오늘은 바디 콤보에 집중합니다. 저번에 바디훅이 헤드로 간 경우가 많았어요. 높이를 의식해보세요.",
  "rounds": [
    {
      "round": 1,
      "theme": "워밍업",
      "instructions": [
        {"time_offset": 0, "combo": "원-투", "repeat": 3},
        {"time_offset": 20, "combo": "원-투-쓰리", "repeat": 3},
        {"time_offset": 45, "combo": "원-투-바디", "repeat": 2}
      ],
      "focus_reminders": [
        {"time_offset": 50, "message": "바디 칠 때 높이 낮춰!"}
      ],
      "motivation": [
        {"time_offset": 150, "message": "좋아, 마무리!"}
      ]
    }
  ],
  "post_session_focus": "바디훅 높이에 특히 주목해서 분석할게요."
}
```

**쉐도우 복싱 세션:**

```json
{
  "session_type": "shadow",
  "total_rounds": 4,
  "round_duration": 180,
  "rest_duration": 30,
  "focus": "상황별 콤보 선택",
  "focus_message": "오늘은 상대 반응에 맞는 콤보를 골라 쓰는 연습입니다.",
  "rounds": [
    {
      "round": 1,
      "theme": "기본 공방",
      "scenarios": [
        {"time_offset": 5, "type": "opponent_attack", "call": "잽이 왔어!"},
        {"time_offset": 15, "type": "opening", "call": "가드가 열렸어, 찬스!"},
        {"time_offset": 30, "type": "pressure", "call": "상대가 압박해!"},
        {"time_offset": 50, "type": "retreat", "call": "상대가 물러나, 추격!"}
      ],
      "pace_cues": [
        {"time_offset": 90, "message": "템포 올려!"},
        {"time_offset": 150, "message": "마지막 30초, 집중!"}
      ]
    }
  ]
}
```

### 2.3 Feedback Generation

세션 분석 결과를 코치 언어 피드백으로 변환합니다.

| 항목 | 선택 |
|------|------|
| 모델 | Claude Haiku / GPT-4o-mini (세션 생성과 동일) |
| 호출 시점 | 세션 분석 완료 후 1회 |
| 입력 | User Profile + 이번 세션 분석 결과 |
| 출력 | 코치 언어 피드백 (3~5문장) |

**프롬프트 구조:**

```
[System]
너는 복싱 코치야. 세션 분석 결과를 보고 사용자에게 피드백을 줘.
코치가 하는 것처럼 자연스럽게. 숫자는 근거로 쓰되, 기술 용어는 금지.

규칙:
- 구체적: "방어가 약해요"가 아니라 "훅 치고 나서 왼손이 턱 옆으로 안 돌아와요"
- 변화 중심: 이전과 비교해서 뭐가 달라졌는지
- 행동 가능: 다음 훈련에서 바로 의식할 수 있는 수준
- 절제된 격려: 과하지 않은 긍정
- 3~5문장으로 간결하게

금지:
- 내부 수치 직접 노출 (0.34 같은)
- 기술 용어 (entropy, diversity_index)
- "대단해요!", "최고예요!" 같은 과한 표현

[User Profile]
{Layer 1 데이터}

[이번 세션 분석 결과]
{세션 분석 JSON}

[요청]
이번 세션 피드백을 만들어줘.
```

---

## 3. Data Model

### 3.1 Core Entities

**User**
```
user:
  id: uuid
  device_id: string
  experience_level: string (beginner/intermediate/advanced)
  created_at: timestamp
  onboarding_completed: boolean
  self_assessment: json (온보딩 자가진단 결과)
```

**Session**
```
session:
  id: uuid
  user_id: uuid → user.id
  type: enum (drill/shadow)
  mode: enum (video/audio_only)
  project_id: uuid → growth_project.id (nullable)
  session_plan: json (LLM이 생성한 세션 플랜)
  duration_seconds: int
  rounds: int
  created_at: timestamp
  completed_at: timestamp (nullable)
  video_url: string (nullable)
  analysis_status: enum (pending/processing/completed/failed)
```

**Session Analysis — Drill**
```
drill_analysis:
  id: uuid
  session_id: uuid → session.id
  instructions: json
    [{timestamp: float, combo_name: string}]
  detected_actions: json
    [{timestamp: float, action: string, confidence: float}]
  detected_combos: json
    [{start_time: float, end_time: float, combo: [string]}]
  matches: json
    [{instruction_idx: int, combo_idx: int, result: string (success/partial/miss), detail: string}]
  combo_stats: json
    {combo_name: {attempts: int, successes: int, partials: int, success_rate: float}}
  feedback_text: string (LLM 생성)
  created_at: timestamp
```

**Session Analysis — Shadow**
```
shadow_analysis:
  id: uuid
  session_id: uuid → session.id
  scenarios: json
    [{timestamp: float, type: string, call: string}]
  detected_responses: json
    [{timestamp: float, action_type: string, combo_used: [string]}]
  scenario_matches: json
    [{scenario_idx: int, response_idx: int, reacted: boolean, reaction_time: float}]
  stats: json
    {defense_reaction_rate: float, response_diversity: float, combo_distribution: {combo: count}}
  feedback_text: string (LLM 생성)
  created_at: timestamp
```

**Combo Mastery**
```
combo_mastery:
  id: uuid
  user_id: uuid → user.id
  combo_name: string
  status: enum (new/learning/proficient/mastered)
  drill_success_rate: float (EMA, 최근 세션 기반)
  shadow_usage_rate: float (최근 쉐도우 세션 기반)
  total_attempts: int
  total_successes: int
  first_attempted_at: timestamp
  last_attempted_at: timestamp
  updated_at: timestamp
```

**Growth Project**
```
growth_project:
  id: uuid
  user_id: uuid → user.id
  name: string ("바디 콤보 확장")
  target_combos: [string]
  started_at: timestamp
  completed_at: timestamp (nullable)
  before_snapshot: json ({combo: {drill_rate, shadow_rate}})
  current_snapshot: json ({combo: {drill_rate, shadow_rate}})
  sessions_count: int
```

### 3.2 Combo Mastery 상태 전이

```
new (안 해봄)
  → learning (시도했지만 성공률 < 50%)
    → proficient (성공률 50~80%, 최소 3세션)
      → mastered (성공률 80%+, 3세션 연속 유지)
```

업데이트 타이밍: 매 세션 분석 완료 후.

EMA (지수이동평균) 계산:
```
new_rate = α × current_session_rate + (1 - α) × previous_rate
α = 0.3 (최근 세션 가중치)
```

### 3.3 Growth Project 완료 조건

다음 중 하나 충족 시 완료:

- **target_reached**: 모든 target_combos의 drill_success_rate ≥ 0.7
- **significant_gain**: target_combos 평균 drill_success_rate의 before → current delta ≥ 0.25
- **max_sessions**: 프로젝트 내 세션 ≥ 10회

---

## 4. Data Collection Strategy

### 4.1 초기 학습 데이터 확보

**Phase 0: 출시 전 데이터**

| 소스 | 방법 | 예상 데이터량 | 소요 |
|------|------|-------------|------|
| 직접 촬영 | 각 펀치 200회 + 콤보 20종 × 50회 | ~2000 클립 | 3~5일 |
| 유튜브 | 드릴 영상에서 키포인트 추출 | ~2000~3000 클립 | 파이프라인 구축 2~3일 |

직접 촬영 시 조건 변화:
- 카메라 각도: 정면, 좌측 45°, 우측 45°
- 거리: 2m, 3m, 4m
- 조명: 밝은 곳, 어두운 곳
- 복장: 다양하게

**유튜브 데이터 파이프라인:**

```
1. 복싱 드릴 영상 목록 수집
   - 검색어: "boxing combination drill", "boxing pad work tutorial"
   - 프로 코치 채널 위주

2. 음성에서 콤보 레이블 추출
   - Whisper (STT) → 코치 음성에서 콤보 이름 탐지
   - "잽! 크로스! 훅!" → 해당 구간에 레이블 부여

3. 포즈 추출 + 저장
   - MediaPipe로 키포인트만 추출
   - 원본 영상 저장하지 않음 (저작권)
   - 키포인트 시계열 + 레이블만 저장

4. 정제
   - 여러 사람 등장 시 주요 인물만
   - confidence 낮은 프레임 제거
   - 레이블 정확도 샘플링 검증
```

### 4.2 사용자 데이터 수집 (운영 중)

Atom 세션 자체가 데이터 수집 파이프라인입니다.

```
TTS "원-투-쓰리!" 지시 (t=12.0)
  ↓
사용자 수행 (영상)
  ↓
분석: 인식된 콤보 + confidence
  ↓
필터링:
  - confidence ≥ 0.8
  - 매칭 결과 success인 것만
  ↓
학습 데이터로 축적
```

이 데이터의 고유 가치:
- **"지시 + 수행" 쌍 데이터**: 세상에 없는 데이터. Atom에만 존재
- **종단 데이터**: 한 사람의 성장 궤적. 초급 → 중급 과정의 데이터
- **다양한 실환경**: 실제 사용자의 체육관/자택 환경

### 4.3 모델 개선 사이클

```
v1 (출시): 직접 촬영 + 유튜브 데이터로 학습한 CTR-GCN
  ↓
운영하면서 사용자 데이터 축적
  ↓
v2 (데이터 5000개+): 사용자 데이터 포함 재학습. 정확도 향상
  ↓
v3 (데이터 10000개+): 방어 동작 분류 추가 (Phase 2)
  ↓
v4 (데이터 50000개+): 폼 품질 분석 도입 검토 (Future)
```

---

## 5. System Architecture

```
┌─────────────────────────────────────────────┐
│                  Mobile App                  │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐ │
│  │   TTS   │  │  Camera  │  │    UI      │ │
│  │  Player │  │ Recorder │  │ (RN/Expo)  │ │
│  └─────────┘  └──────────┘  └────────────┘ │
└─────────────────┬───────────────────────────┘
                  │ API calls + Video upload
                  ▼
┌─────────────────────────────────────────────┐
│               API Server (FastAPI)           │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐ │
│  │  Session  │  │  User     │  │  Project │ │
│  │  Mgmt    │  │  Profile  │  │  Mgmt    │ │
│  └──────────┘  └───────────┘  └──────────┘ │
└─────────┬──────────┬────────────────────────┘
          │          │
          ▼          ▼
┌──────────────┐  ┌─────────────────────────────┐
│  PostgreSQL  │  │     Analysis Pipeline        │
│  ─────────── │  │  ┌──────────┐  ┌──────────┐ │
│  users       │  │  │MediaPipe │→ │ CTR-GCN  │ │
│  sessions    │  │  │  Pose    │  │ Classify │ │
│  analyses    │  │  └──────────┘  └──────────┘ │
│  combo_mast. │  │       ↓             ↓       │
│  projects    │  │  ┌──────────┐  ┌──────────┐ │
│              │  │  │ Sequence │  │ Session  │ │
│              │  │  │  Recog.  │  │ Matcher  │ │
│              │  │  └──────────┘  └──────────┘ │
│              │  └─────────────────────────────┘
│              │
│              │  ┌─────────────────────────────┐
│              │  │        LLM Service           │
└──────────────┘  │  ┌──────────┐  ┌──────────┐ │
                  │  │ Session  │  │ Feedback │ │
  ┌────────────┐  │  │Generator │  │Generator │ │
  │    GCS     │  │  └──────────┘  └──────────┘ │
  │  (Video)   │  └─────────────────────────────┘
  └────────────┘
```

### 5.1 Technology Stack

**Mobile:**

| 기술 | 용도 |
|------|------|
| React Native (Expo) | iOS/Android 앱 |
| Expo Router | 파일 기반 라우팅 |
| expo-av | TTS 오디오 재생 |
| expo-camera | 세션 중 촬영 |
| AsyncStorage | 로컬 캐시 (MVP) |
| Zustand | 상태 관리 |
| TanStack Query | 서버 데이터 캐시 |

**Backend:**

| 기술 | 용도 |
|------|------|
| Python (FastAPI) | API 서버 |
| PostgreSQL | 메인 데이터베이스 |
| Google Cloud Storage | 영상 파일 저장 |
| Redis | 분석 상태 폴링 + 캐시 |
| Cloud Run | API 서버 호스팅 |
| Cloud Tasks | 비동기 분석 트리거 |

**AI/ML:**

| 기술 | 용도 |
|------|------|
| MediaPipe Pose | 키포인트 추출 |
| CTR-GCN (PyTorch) | 동작 분류 |
| Claude Haiku / GPT-4o-mini | 세션 생성 + 피드백 생성 |
| Google Cloud TTS | 훈련 세션 음성 (MVP) |
| Whisper | 유튜브 데이터 수집 시 STT |

**Infra:**

| 기술 | 용도 |
|------|------|
| GCP | 전체 클라우드 인프라 |
| Cloud Run (GPU) | 분석 파이프라인 실행 |
| Artifact Registry | ML 모델 버전 관리 |
| Cloud Monitoring | 분석 파이프라인 모니터링 |

### 5.2 분석 플로우

```
1. 세션 완료 → 앱이 영상을 GCS에 업로드
2. 업로드 완료 → API가 Cloud Tasks로 분석 트리거
3. 분석 파이프라인 (Cloud Run GPU):
   a. GCS에서 영상 다운로드
   b. MediaPipe 포즈 추출
   c. CTR-GCN 동작 분류
   d. 시퀀스 인식 + 세션 매칭
   e. 분석 결과 → PostgreSQL 저장
4. LLM 피드백 생성 → PostgreSQL 저장
5. 앱이 폴링으로 분석 완료 확인 → 피드백 표시

예상 처리 시간: 1~2분 (12분 세션 기준)
```

---

## 6. API Specification

### 6.1 Endpoints

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/users` | device_id로 유저 생성 |
| POST | `/api/v1/users/:id/onboarding` | 자가진단 결과 저장 |
| GET | `/api/v1/users/:id/profile` | User Profile (Layer 1) |
| GET | `/api/v1/users/:id/combo-mastery` | 콤보 숙련도 목록 |
| POST | `/api/v1/sessions/generate` | LLM 기반 세션 생성 |
| POST | `/api/v1/sessions` | 세션 시작 기록 |
| POST | `/api/v1/sessions/:id/complete` | 세션 완료 + 영상 업로드 |
| GET | `/api/v1/sessions/:id/analysis` | 분석 결과 + 피드백 (폴링) |
| GET | `/api/v1/users/:id/sessions` | 세션 히스토리 (달력용) |
| GET | `/api/v1/users/:id/project` | 현재 Growth Project |
| POST | `/api/v1/users/:id/project/complete` | 프로젝트 완료 처리 |
| GET | `/api/v1/users/:id/project/next` | 다음 프로젝트 제안 |

### 6.2 Key API Flows

**세션 생성 → 수행 → 분석 플로우:**

```
1. GET  /users/:id/profile          → User Profile 확인
2. POST /sessions/generate          → LLM이 세션 플랜 생성
3. POST /sessions                   → 세션 시작 기록
4. (사용자 훈련 수행)
5. POST /sessions/:id/complete      → 세션 완료 + 영상 업로드
6. GET  /sessions/:id/analysis      → 폴링 (2초 간격)
7. (분석 완료 시) 피드백 + 다음 세션 자동 생성
```

---

## 7. Constraints & Decisions

### 7.1 분석 제약

| 제약 | 영향 | 대응 |
|------|------|------|
| 촬영 환경 비통제 | 포즈 추출 품질 편차 | 촬영 가이드라인 + confidence threshold |
| 카메라 각도 다양 | 동작 분류 정확도 저하 | 다양한 각도 학습 데이터 확보 |
| 실시간 분석 불가 | 세션 중 피드백 불가 | 세션 후 분석으로 설계 (1~2분 처리) |
| 바디샷 vs 헤드샷 구분 | 높이 기반 구분의 한계 | 키포인트 y좌표 기반 + 팔꿈치 각도 보조 |
| 방어 동작 인식 (Phase 2) | 슬립/덕 구분 어려움 | 타이밍 기반 휴리스틱 → ML 전환 |

### 7.2 비용 추정 (월간, 사용자 1000명 기준)

| 항목 | 단가 | 월 사용량 | 월 비용 |
|------|------|----------|---------|
| LLM (세션 생성 + 피드백) | $0.01/세션 | 8000 세션 | $80 |
| Cloud Run GPU (분석) | $0.50/시간 | ~200시간 | $100 |
| GCS (영상 저장) | $0.02/GB | ~2TB | $40 |
| PostgreSQL (Cloud SQL) | — | — | $50 |
| 기타 (Redis, Tasks 등) | — | — | $30 |
| **합계** | | | **~$300/월** |

사용자당 $0.30/월. 충분히 합리적입니다.

### 7.3 핵심 기술 결정 요약

| 결정 | 선택 | 이유 |
|------|------|------|
| 포즈 추출 | MediaPipe (서버) | 성숙, 무료, 33 키포인트 |
| 동작 분류 | CTR-GCN | 확장성, 관절 관계 학습, 정확도 |
| 시퀀스/매칭 | Rule-based | 단순, ML 불필요 |
| 세션 생성 | LLM | variation 무한, 비용 낮음 |
| 피드백 생성 | LLM | 자연스러운 코치 언어 |
| 분석 위치 | 서버 (후처리) | 실시간 불필요, 배터리/발열 회피 |
| 초기 데이터 | 직접 촬영 + 유튜브 키포인트 | 1주 내 확보 가능 |
| 히스토리 관리 | 3-Layer Context | 토큰 효율적 |

---

## 8. Implementation Phases

### Phase 1: Analysis Pipeline + Drill Loop (5주)

| 주차 | 항목 |
|------|------|
| W1 | 데이터 수집 (직접 촬영 + 유튜브 파이프라인) |
| W2 | CTR-GCN 학습 + 평가 (5가지 펀치 분류) |
| W2-3 | 분석 파이프라인 구축 (MediaPipe → CTR-GCN → 시퀀스 → 매칭) |
| W3 | LLM 세션 생성 + 피드백 생성 프롬프트 설계 |
| W3-4 | API 서버 + DB + 분석 비동기 플로우 |
| W4-5 | 모바일 앱: 온보딩 + 드릴 세션 + 촬영 + 피드백 |

### Phase 2: Shadow Session + Gap Analysis (3주)

| 주차 | 항목 |
|------|------|
| W6 | 쉐도우 TTS 시나리오 설계 + LLM 프롬프트 |
| W6-7 | 방어 동작 인식 (타이밍 기반 휴리스틱) |
| W7 | 쉐도우 분석 + 드릴-쉐도우 갭 분석 |
| W8 | 모바일 앱: 쉐도우 세션 + 피드백 |

### Phase 3: Growth Proof (2주)

| 주차 | 항목 |
|------|------|
| W9 | Growth Project 완료 판단 + Before/After |
| W9-10 | 달력 뷰 + 장기 추이 + 다음 프로젝트 제안 |

**총: 약 10주**

---

*Atom Tech Spec — 데이터가 핵심이다.*