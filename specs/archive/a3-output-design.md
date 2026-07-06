# A3 Design: Voice & Visual Output Interface

> Status: DRAFT
> Created: 2026-03-07
> Author: Track A design phase

## 1. Objective

Define the interface contract between the **Session Engine** and its **output layer** (terminal display + voice), so that:
- The current macOS CLI implementation can be swapped for other frontends (mobile, web, GUI) without changing the engine.
- The TTS backend can be upgraded independently.
- Future frontends can subscribe to session events in real time.

---

## 2. Output Event Protocol

The Session Engine emits typed events. Any output layer subscribes by implementing the `OutputHandler` protocol.

### 2.1 Event Types

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class OutputHandler(Protocol):
    def on_session_start(self, plan: dict) -> None: ...
    def on_round_start(self, round_num: int, total_rounds: int, duration_sec: int) -> None: ...
    def on_combo_call(self, combo_display_name: str, actions: list[str], ts: float) -> None: ...
    def on_round_end(self, round_num: int) -> None: ...
    def on_rest_start(self, round_num: int, rest_sec: int) -> None: ...
    def on_rest_end(self, round_num: int) -> None: ...
    def on_session_end(self, status: str, rounds_completed: int, combos_delivered: int, duration_sec: float) -> None: ...
```

### 2.2 Event Payload Summary

| Event | Key Data |
|-------|----------|
| `session_start` | plan dict (template, focus, rounds) |
| `round_start` | round_num, total_rounds, duration_sec |
| `combo_call` | combo_display_name, actions, timestamp_offset |
| `round_end` | round_num |
| `rest_start` | round_num, rest_sec |
| `rest_end` | round_num |
| `session_end` | status (completed/abandoned), rounds_completed, combos_delivered, duration_sec |

### 2.3 Current Implementation: `CliOutputHandler`

```
CliOutputHandler
├── on_combo_call  → click.echo(name + actions) + say(name)
├── on_round_start → click.echo("Round N/M (Xs)")
├── on_rest_start  → click.echo("Rest: Xs") + say("휴식")
└── on_session_end → click.echo summary
```

The current `SessionEngine.on_output` callback and inline TTS calls should be **refactored to use this protocol** when a second frontend is added. The current monolithic design is acceptable for single-frontend use.

---

## 3. TTS Evaluation

### 3.1 Current: macOS `say`

**Implementation:** `subprocess.Popen(["say", "-r", "200", text])`

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Latency | ✅ <100ms | Spawns instantly |
| Korean quality | ⚠️ Fair | Limited Korean voice (Yuna). Sounds robotic for longer phrases. |
| Cost | ✅ Free | Built into macOS |
| Cross-platform | ❌ macOS only | Unavailable on Linux/Windows |
| Offline | ✅ Yes | No internet required |
| Setup | ✅ Zero | Pre-installed |

**Verdict:** Sufficient for solo developer use on macOS. Not suitable for distribution.

---

### 3.2 Option B: Edge-TTS (Microsoft Edge neural TTS)

**Package:** `edge-tts` (pip installable, free)

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Latency | ⚠️ ~300-500ms | Requires internet, streams audio |
| Korean quality | ✅ Excellent | `ko-KR-SunHiNeural` — natural, expressive |
| Cost | ✅ Free | Uses Microsoft's public endpoint |
| Cross-platform | ✅ Yes | Python package, works on macOS/Linux/Windows |
| Offline | ❌ No | Requires internet connection |
| Setup | ✅ `pip install edge-tts` | Straightforward |

**Integration sketch:**
```python
import edge_tts, asyncio, tempfile, subprocess

async def speak_edge(text: str) -> None:
    communicate = edge_tts.Communicate(text, voice="ko-KR-SunHiNeural")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        await communicate.save(f.name)
        subprocess.Popen(["afplay", f.name])  # macOS; use mpg123 on Linux
```

**Verdict:** Best Korean voice quality. Recommended upgrade path when internet is available.

---

### 3.3 Option C: Google Cloud TTS

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Latency | ⚠️ ~200-400ms | REST API call |
| Korean quality | ✅ Excellent | WaveNet/Neural2 voices |
| Cost | ⚠️ $4/1M chars | Free tier: 1M chars/month (sufficient for personal use) |
| Cross-platform | ✅ Yes | REST API |
| Offline | ❌ No | Requires internet |
| Setup | ⚠️ Medium | API key, `google-cloud-texttospeech` package |

**Verdict:** Overkill for personal use. Google Cloud TTS adds operational complexity for no meaningful quality gain over edge-tts.

---

### 3.4 TTS Recommendation

| Use Case | Recommended |
|----------|-------------|
| Solo developer, macOS, offline | `say` (current) |
| Higher voice quality, internet available | `edge-tts` |
| Production / distributed app | Google Cloud TTS or Azure Cognitive Services |

**Immediate recommendation:** Keep `say` for Track A. Add `edge-tts` as an optional backend behind an `--tts-engine` flag (`say` / `edge`).

---

## 4. Frontend Platform Decision Criteria

When Track A moves to a visual frontend (A3), evaluate platforms by:

| Criterion | Weight | Notes |
|-----------|--------|-------|
| Real-time event subscription | High | Must receive `combo_call` events with <100ms latency |
| Korean text rendering | High | Display combo names and instructions |
| Audio playback | High | Can play TTS audio or trigger native TTS |
| Development speed | Medium | Time to working prototype |
| Offline capability | Medium | Local SQLite, no cloud required |
| iOS/macOS native | Low | Nice-to-have for phone use while training |

**Candidate platforms:**

| Platform | Real-time | Korean | Audio | Speed | Offline |
|----------|-----------|--------|-------|-------|---------|
| **Terminal (current)** | ✅ | ✅ | ✅ (say) | ✅ | ✅ |
| **FastAPI + HTMX** | ✅ SSE | ✅ | ✅ Web Audio | ✅ | ✅ |
| **Textual (TUI)** | ✅ | ✅ | ✅ (say) | ✅ | ✅ |
| **React/Next.js** | ✅ SSE/WS | ✅ | ✅ Web Audio | ⚠️ | ⚠️ |
| **Swift (iOS/macOS)** | ✅ | ✅ | ✅ AVSpeech | ❌ slow | ✅ |

**A3 Recommendation:** Start with **Textual** (Python TUI library) for a rich terminal UI with progress bars, round timers, and live combo display — no web stack required, offline, fast to build. Migrate to FastAPI + HTMX if a browser UI is needed.

---

## 5. Interface Contract (Session Engine → Frontend)

The Session Engine's responsibility ends at emitting events. The frontend is responsible for:

1. **Rendering** — displaying round timers, current combo, session progress
2. **Audio** — playing TTS or audio cues
3. **Input** — handling pause/abort signals (Ctrl+C, button press)

The current `on_output: Callable[[str], None]` in `SessionEngine` is a minimal text-only contract. The full `OutputHandler` protocol above should be adopted when adding a second frontend.

**Event ordering guarantee:** Events are emitted in strict chronological order within a session. No concurrent events.

**Abort contract:** `engine.abort()` is safe to call from any thread/coroutine. The engine will complete the current sleep interval, then stop gracefully.

---

## 6. Open Questions

- [ ] Should pause/resume be added before A3? Current design doesn't support it.
- [ ] Should TTS audio be pre-generated at plan generation time (lower latency, larger files)?
- [ ] Is a phone display (iOS) needed, or is terminal sufficient for shadow boxing use?

---

## Changelog

| Date | Change | Reason |
|------|--------|--------|
| 2026-03-07 | Initial draft | Task 6 complete |
