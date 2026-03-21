# TTS Spec: ElevenLabs Pre-generated Audio

## Overview

각 라운드를 하나의 연속 오디오 파일로 사전 생성한다.
세그먼트별 개별 TTS 호출 대신, SSML로 하나의 스크립트를 구성하여
실제 코치가 패드워크하는 듯한 자연스러운 흐름을 만든다.

```
Plan Generation (LLM)
  → SSML Build (segments → one script per round)
    → ElevenLabs API (SSML → MP3)
      → Static File Serve (/audio/{plan_id}/round_N.mp3)
        → Mobile Audio Playback (expo-audio)
          → UI Sync (timestamps → text/intensity display)
```

---

## 1. SSML 구성

각 라운드의 세그먼트를 하나의 `<speak>` 블록으로 결합한다.

### 매핑 규칙 (overview.md Section 5)

| 필드 | 값 | SSML `rate` | SSML `pitch` | SSML `volume` |
|------|-----|-------------|--------------|---------------|
| tempo | slow | 85% | - | - |
| tempo | medium | 100% | - | - |
| tempo | fast | 120% | - | - |
| intensity | low | - | -2st | soft |
| intensity | medium | - | 0st | medium |
| intensity | high | - | +2st | loud |

### SSML 예시

```xml
<speak>
  <prosody rate="100%" pitch="0st" volume="medium">잽 잽 크로스</prosody>
  <break time="6500ms"/>
  <prosody rate="85%" pitch="-2st" volume="soft">가드 올려요</prosody>
  <break time="4500ms"/>
  <prosody rate="120%" pitch="+2st" volume="loud">원투훅! 계속!</prosody>
  <break time="5000ms"/>
  <prosody rate="120%" pitch="+2st" volume="loud">더블잽 크로스!</prosody>
</speak>
```

### Break 계산

```
break_ms = (segment.duration * 1000) - estimated_tts_ms
estimated_tts_ms = max(800, len(segment.text) * 150)  # ~150ms per character
break_ms = max(500, break_ms)  # minimum 500ms
```

마지막 세그먼트 뒤에는 break를 넣지 않는다.

---

## 2. Timestamp 동기화

모바일 UI가 오디오 재생 위치에 맞춰 텍스트를 표시하기 위해,
각 세그먼트의 시작/종료 시간을 사전 계산한다.

```
cursor_ms = 0
for each segment:
  tts_ms = max(800, len(text) * 150)
  start_ms = cursor_ms
  end_ms = cursor_ms + segment.duration * 1000
  timestamps.append({ start_ms, end_ms, text, tempo, intensity })
  cursor_ms = end_ms
```

### 응답 예시

```json
{
  "round": 1,
  "segments": [...],
  "audio_url": "/audio/abc123/round_1.mp3",
  "timestamps": [
    { "start_ms": 0,     "end_ms": 8000,  "text": "잽 잽 크로스",    "tempo": "medium", "intensity": "low" },
    { "start_ms": 8000,  "end_ms": 14000, "text": "가드 올려요",     "tempo": "slow",   "intensity": "low" },
    { "start_ms": 14000, "end_ms": 24000, "text": "원투훅! 계속!",   "tempo": "fast",   "intensity": "high" }
  ]
}
```

---

## 3. ElevenLabs API

- **Model**: `eleven_multilingual_v2` (한국어 지원)
- **Voice**: 한국어 가능한 에너지 있는 남성 보이스 (자동 선택 또는 `ELEVENLABS_VOICE_ID`로 지정)
- **Input**: SSML 텍스트
- **Output**: MP3 오디오 스트림
- **비용**: ~$0.30/1000자 → 세션당 ~$0.05-0.10 (3라운드 기준)

### 환경변수

| 변수 | 필수 | 설명 |
|------|------|------|
| `ELEVENLABS_API_KEY` | Y | ElevenLabs API 키 |
| `ELEVENLABS_VOICE_ID` | N | 보이스 ID (미설정 시 자동 선택) |

### SDK

```
pip install elevenlabs>=1.0.0
```

---

## 4. 백엔드 API 변경

### 스키마 추가

```python
class SegmentTimestamp(BaseModel):
    start_ms: int
    end_ms: int
    text: str
    tempo: str       # slow | medium | fast
    intensity: str   # low | medium | high

class RoundResponse(BaseModel):
    round: int
    segments: list[SegmentResponse]
    audio_url: str | None = None
    timestamps: list[SegmentTimestamp] | None = None

class PlanResponse(BaseModel):
    # ... 기존 필드 ...
    audio_ready: bool = False
```

### 플로우

```
POST /api/sessions/plan
  1. LLM으로 plan JSON 생성 (기존)
  2. ELEVENLABS_API_KEY가 있으면:
     a. 각 라운드 SSML 빌드
     b. ElevenLabs API 호출 → MP3 저장 (data/audio/{plan_id}/round_N.mp3)
     c. timestamps 계산
     d. plan에 audio_url, timestamps 추가
     e. audio_ready = true
  3. 키가 없으면: audio_ready = false, 기존 응답 그대로

GET /audio/{plan_id}/round_N.mp3
  → FastAPI StaticFiles (data/audio/ 마운트)
```

---

## 5. 모바일 재생

### Audio 모드 (audio_url이 있을 때)

```
expo-audio의 useAudioPlayer 사용:
1. player.replace({ uri: serverUrl + round.audio_url })
2. player.play()
3. 50ms 간격으로 player.currentTime 폴링
4. timestamps 배열에서 현재 위치에 해당하는 세그먼트 찾기
5. UI 텍스트 + intensity 업데이트
6. 라운드 타이머 만료 시 player.pause()
```

### Fallback 모드 (audio_url이 없을 때)

기존 `expo-speech` 세그먼트별 재생 유지 (변경 없음).

### 일시정지/중단

- 일시정지: `player.pause()` / `player.play()`
- 중단: `player.pause()`, 세션 종료 플로우

---

## 6. 파일 구조

```
data/audio/
  {plan_id}/
    round_1.mp3
    round_2.mp3
    round_3.mp3
```

서버 재시작 시 기존 오디오 파일 유지. 정리는 별도 배치로 (MVP에서는 수동 삭제).

---

## 7. Fallback 전략

| 상황 | 동작 |
|------|------|
| `ELEVENLABS_API_KEY` 미설정 | expo-speech fallback |
| ElevenLabs API 에러 | expo-speech fallback, 로그 출력 |
| 네트워크 에러 (오디오 다운로드 실패) | expo-speech fallback |
| 오디오 파일 손상 | expo-speech fallback |

모든 경우에 세션은 중단되지 않는다.
