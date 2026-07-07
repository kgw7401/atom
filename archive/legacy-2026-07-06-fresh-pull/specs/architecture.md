전체 아키텍처: 세션 생성 → TTS 출력

┌─────────────────────────────────────────────────────────────────────┐
│ MOBILE (React Native / Expo) │
│ │
│ ① SessionSetup → 유저가 라운드/시간 설정 후 "시작" │
│ │ │
│ ▼ │
│ ② POST /api/sessions/plan ──────────────────────────────────┐ │
│ { rounds: 3, round_duration_sec: 180, rest_sec: 30 } │ │
│ │ │
└────────────────────────────────────────────────────────────────┼─────┘
│
┌────────────────────────────────────────────────────────────────┼─────┐
│ BACKEND (FastAPI) │ │
│ ▼ │
│ ③ sessions.py router │
│ ├─ \_get_llm_client() → LLMClient (Anthropic/OpenAI) │
│ └─ \_get_tts_service() → TTSService (ELEVENLABS_API_KEY) │
│ │ │
│ ▼ │
│ ④ SessionService.generate_plan() │
│ │ │
│ ├─ 4a. \_get_profile() → 유저 프로필 (경험, 목표) │
│ ├─ 4b. \_get_recent_sessions() → 최근 훈련 기록 │
│ ├─ 4c. \_build_prompt() → 시스템+유저 프롬프트 조합 │
│ │ │
│ ▼ │
│ ⑤ LLMClient.generate_json() │
│ │ system: session_system.md (코치 규칙, 세그먼트 규칙) │
│ │ user: 프로필 + 기록 + 설정 + 유저 요청 │
│ │ │
│ ▼ │
│ ⑥ LLM → Structured JSON │
│ { │
│ "rounds": [{ │
│ "round": 1, │
│ "segments": [ │
│ { "text": "잽잽", "duration": 4, │
│ "tempo": "medium", "intensity": "low" }, │
│ { "text": "원투훅!", "duration": 5, │
│ "tempo": "fast", "intensity": "high" }, │
│ ... │
│ ] │
│ }] │
│ } │
│ │ │
│ ├─ \_validate_plan() → 구조 검증 │
│ ├─ DB 저장 (DrillPlan 테이블) │
│ │ │
│ ▼ │
│ ⑦ TTSService.generate_session_audio(plan, plan_id) │
│ │ │
│ │ For each round: │
│ │ ┌──────────────────────────────────────────┐ │
│ │ │ For each segment: │ │
│ │ │ │ │
│ │ │ ⑦a. ElevenLabs API 호출 │ │
│ │ │ voice_settings: │ │
│ │ │ speed = tempo 매핑 │ │
│ │ │ slow→0.85, medium→1.0, fast→1.15│ │
│ │ │ stability = intensity 매핑 │ │
│ │ │ high→0.30(격렬), low→0.60(차분) │ │
│ │ │ → \_tmp/r1_s1.mp3 (개별 세그먼트) │ │
│ │ │ │ │
│ │ │ ⑦b. pydub로 실제 오디오 길이 측정 │ │
│ │ │ actual_ms = len(audio) │ │
│ │ │ │ │
│ │ │ ⑦c. silence 계산 │ │
│ │ │ gap = max(300, duration\*1000 │ │
│ │ │ - actual_ms) │ │
│ │ └──────────────────────────────────────────┘ │
│ │ │
│ │ ⑦d. pydub 연결 (concatenate) │
│ │ [seg1_audio][silence][seg2_audio][silence][seg3_audio] │
│ │ → data/audio/{plan_id}/round_1.mp3 │
│ │ │
│ │ ⑦e. 정확한 timestamps 계산 │
│ │ [{ start_ms: 0, end_ms: 1440, text: "잽잽" }, │
│ │ { start_ms: 4000, end_ms: 4789, text: "원투훅!" }, ...] │
│ │ │
│ │ ⑦f. \_tmp/ 폴더 삭제 (개별 파일 정리) │
│ │ │
│ ▼ │
│ ⑧ 응답 반환 │
│ { │
│ id: "abc123", │
│ audio_ready: true, │
│ plan: { │
│ rounds: [{ │
│ round: 1, │
│ segments: [...], │
│ audio_url: "/audio/abc123/round_1.mp3", │
│ timestamps: [ │
│ { start_ms: 0, end_ms: 1440, text: "잽잽", ... }, │
│ { start_ms: 4000, end_ms: 4789, text: "원투훅!", ... } │
│ ] │
│ }] │
│ } │
│ } │
│ │
│ ⑨ StaticFiles 서빙 │
│ GET /audio/abc123/round_1.mp3 → data/audio/abc123/round_1.mp3 │
│ │
└─────────────────────────────────────────────────────────────────────┘
│
┌────────────────────────────────┼────────────────────────────────────┐
│ MOBILE (ActiveSessionScreen) ▼ │
│ │
│ ⑩ PlanPreview → 유저가 플랜 확인 후 "시작" │
│ │ │
│ ▼ │
│ ⑪ ActiveSession — 라운드 루프 │
│ │ │
│ ├─ audio_url 있으면 → runAudioRound() │
│ │ ┌────────────────────────────────────────┐ │
│ │ │ 1. expo-av로 round MP3 로드+재생 │ │
│ │ │ 2. 50ms 폴링: currentTime 확인 │ │
│ │ │ 3. timestamps에서 현재 세그먼트 매칭 │ │
│ │ │ 4. UI 업데이트 (텍스트+intensity+haptic)│ │
│ │ │ 5. 라운드 타이머 만료 시 오디오 정지 │ │
│ │ └────────────────────────────────────────┘ │
│ │ │
│ └─ audio_url 없으면 → runFallbackSegments() │
│ (기존 expo-speech 세그먼트별 재생) │
│ │
│ ⑫ 라운드 간 휴식 → "휴식" TTS → 대기 │
│ │ │
│ ▼ │
│ ⑬ SessionEnd → POST /api/sessions/log │
│ │
└─────────────────────────────────────────────────────────────────────┘
