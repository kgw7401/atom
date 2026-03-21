# Spec: Frontend — React Native App

> Status: APPROVED
> Created: 2026-03-08
> Last Updated: 2026-03-08

## TL;DR

- **Goal:** Build a React Native (Expo) mobile app that replaces the CLI for running drill sessions
- **Approach:** Expo app connects to local FastAPI server; real-time session events streamed via WebSocket; device TTS via `expo-speech`
- **Scope:** 7 tasks, overall complexity L
- **Key risk:** WebSocket-based real-time session streaming requires `SessionEngine` refactor to emit structured events
- **Decisions needing input:** None — all decisions made autonomously below

---

## 1. Objective

- **What:** A React Native mobile app for running boxing drill sessions with real-time combo calling and training history
- **Why:** The CLI is unusable during shadowboxing — you can't see the terminal while moving. A phone screen you can glance at is essential.
- **Who:** The solo user shadowboxing with their phone nearby
- **Success Criteria:**
  - [ ] User can start a session from phone: select template → optional prompt → preview → run
  - [ ] Active session screen shows current combo, round timer, and progress in real time
  - [ ] Device TTS speaks combo names in Korean (via `expo-speech`)
  - [ ] User can view training history and profile on phone
  - [ ] Session abort (tap button) saves log with `abandoned` status

---

## 2. Technical Design

### Architecture

```
┌─────────────────────────────────────────┐
│         iPhone / Android                │
│  ┌───────────────────────────────────┐  │
│  │     Expo React Native App         │  │
│  │  - React Navigation (screens)     │  │
│  │  - Zustand (state)                │  │
│  │  - expo-speech (TTS)              │  │
│  │  - WebSocket client               │  │
│  └───────────────────────────────────┘  │
│           │ HTTP + WebSocket            │
└───────────┼─────────────────────────────┘
            │ (same WiFi / localhost)
┌───────────┼─────────────────────────────┐
│  FastAPI Backend (existing)             │
│  ├── /api/templates       (GET)         │
│  ├── /api/sessions/plan   (POST)        │
│  ├── /ws/sessions/{id}    (WebSocket) ← NEW │
│  ├── /api/sessions        (GET)         │
│  ├── /api/sessions/{id}   (GET)         │
│  ├── /api/profile         (GET/PUT)     │
│  └── /api/combos          (CRUD)        │
└─────────────────────────────────────────┘
```

### WebSocket Session Protocol

Server → Client (JSON events):
```json
{"type": "session_start", "plan": {...}}
{"type": "round_start", "round": 1, "total": 3, "duration_sec": 120}
{"type": "combo_call", "name": "원투", "actions": ["jab", "cross"], "ts": 5.2}
{"type": "round_end", "round": 1}
{"type": "rest_start", "round": 1, "rest_sec": 45}
{"type": "rest_end", "round": 1}
{"type": "session_end", "status": "completed", "rounds": 3, "combos": 24, "duration_sec": 430}
```

Client → Server:
```json
{"type": "abort"}
```

**Reconnection protocol:**
- Backend buffers last 10 emitted events in memory per session
- Client sends `{"type": "reconnect", "last_ts": 5.2}` on reconnect
- Server replays events with `ts > last_ts` before resuming live stream
- If session already ended: server sends final `session_end` event immediately

### App Screens

```
HomeScreen
  └─ SessionSetupScreen → PlanPreviewScreen → ActiveSessionScreen → SessionEndScreen
  └─ HistoryScreen → SessionDetailScreen
  └─ ProfileScreen
  └─ ComboListScreen → ComboAddScreen
```

### Directory Structure

```
mobile/
├── app.json
├── package.json
├── src/
│   ├── api/          # API client (fetch + WebSocket)
│   ├── screens/      # Screen components
│   ├── components/   # Shared UI components
│   ├── store/        # Zustand state
│   └── hooks/        # Custom hooks
```

### Decisions Made Autonomously

| Decision | Choice | Reason |
|---|---|---|
| Framework | Expo (managed) | Fastest setup, no native build tools needed for MVP |
| Navigation | React Navigation v6 | Standard, well-documented |
| State | Zustand | Simple, no boilerplate |
| Styling | StyleSheet (built-in) | No extra deps, sufficient for this scope |
| TTS | expo-speech | Free, offline, Korean supported on iOS/Android |
| Backend connection | localhost:8000 default | Works out of box on simulator; Settings screen to change IP for physical device |
| Session streaming | WebSocket | Native support in React Native, bidirectional |

### Backend Changes Required

1. **`SessionEngine`**: Add `on_event: Callable[[dict], Awaitable[None]] | None` callback, called whenever a `SessionEvent` is recorded. Allows WebSocket handler to stream events.
2. **New WebSocket endpoint**: `GET /ws/sessions/{plan_id}` — fetches plan from DB, runs engine with `tts_enabled=False`, streams events to client.
3. **CORS**: Add `CORSMiddleware` to FastAPI app for all origins.
4. **`atom serve` CLI command**: Starts uvicorn server on `0.0.0.0:8000`.

---

## 3. Implementation Plan

- [ ] **Task 1: Backend — WebSocket session endpoint**
  - Scope: `session_engine.py` (add `on_event` callback), new `src/atom/api/routers/ws_session.py`, `app.py` (include router + CORS), `main.py` (add `atom serve`)
  - Verification: `atom serve` starts server; `wscat ws://localhost:8000/ws/sessions/{id}` receives events in real time
  - Complexity: M

- [ ] **Task 2: Expo app scaffold**
  - Scope: `mobile/` directory — `package.json`, `app.json`, React Navigation, Zustand, API client (`src/api/client.ts`, `src/api/session.ts`), `SettingsScreen.tsx` (backend IP input, stored in AsyncStorage)
  - Verification: `npx expo start` shows app on simulator; API client can `GET /api/templates` using configured base URL; Settings screen lets user change IP and persists it
  - Complexity: S

- [ ] **Task 3: Home + Session Setup screens**
  - Scope: `HomeScreen.tsx`, `SessionSetupScreen.tsx`, `PlanPreviewScreen.tsx`
  - Verification: User can pick template, enter prompt, call `POST /api/sessions/plan`, and see plan preview with round/combo count
  - Complexity: M

- [ ] **Task 4: Active Session screen (WebSocket + TTS)**
  - Scope: `ActiveSessionScreen.tsx`, `src/hooks/useSessionStream.ts`, `src/components/ConnectionLostOverlay.tsx`
  - Verification: Screen connects to WebSocket, displays current combo and round timer live, `expo-speech` speaks Korean combo names, Abort button sends `{"type":"abort"}` and navigates to summary; disconnection shows "Connection lost — session still running" overlay with auto-reconnect countdown; reconnect replays missed events
  - Complexity: L

- [ ] **Task 5: Session End screen**
  - Scope: `SessionEndScreen.tsx`
  - Verification: After session completes/aborts, shows rounds completed, combos delivered, duration, and "Start Again" button
  - Complexity: S

- [ ] **Task 6: History + Profile screens**
  - Scope: `HistoryScreen.tsx`, `SessionDetailScreen.tsx`, `ProfileScreen.tsx`
  - Verification: History lists past sessions from `GET /api/sessions`; Profile shows stats from `GET /api/profile`
  - Complexity: M

- [ ] **Task 7: Combo management screen**
  - Scope: `ComboListScreen.tsx`, `ComboAddScreen.tsx`
  - Verification: Combos listed from `GET /api/combos`; user can add a custom combo via `POST /api/combos`
  - Complexity: S

---

## 4. Boundaries

- ✅ **Always:**
  - App connects to backend via env-configured base URL (not hardcoded localhost)
  - TTS is handled on-device via `expo-speech` — backend does NOT speak during WebSocket sessions
  - Backend session log is always saved (completed or abandoned)
  - System combos are read-only in the UI (no edit/delete)

- ⚠️ **Ask first:**
  - Adding user auth (login/accounts) — out of scope for now, single-user
  - Switching to a cloud-hosted backend (vs local network)
  - Adding pause/resume to the session engine

- 🚫 **Never:**
  - Hardcode the backend URL — always use env/config
  - Delete session logs from the UI
  - Run the session engine client-side (it must run on the FastAPI server)

---

## 5. Testing Strategy

- **Unit:** API client functions (`fetchTemplates`, `generatePlan`, etc.) — mock fetch responses
- **Integration:** WebSocket hook (`useSessionStream`) — mock WebSocket server, verify event handling
- **Conformance:**
  - Input: WebSocket receives `{"type":"combo_call","name":"원투","actions":["jab","cross"],"ts":5.2}`
  - Expected: Screen updates to show "원투", actions "jab → cross"; `expo-speech` called with "원투"
  - Input: User taps Abort during round 2
  - Expected: `{"type":"abort"}` sent over WebSocket; screen navigates to SessionEnd with status "abandoned"

---

## 6. Open Questions

- [ ] Should the app show a countdown timer before round 1 starts (3-2-1 animation)?
- [ ] Should combo history be shown during the active session (scrolling log of called combos)?
- [x] What happens if the WebSocket connection drops mid-session? → Overlay + auto-reconnect with event replay (resolved)
- [ ] Is Android support required, or iOS-first?

---

## Changelog

| Date | Change | Reason |
|------|--------|--------|
| 2026-03-08 | Initial draft | Phase 2 complete |
| 2026-03-08 | Added reconnection protocol, settings screen for IP, ConnectionLostOverlay | Review R1 fixes |
