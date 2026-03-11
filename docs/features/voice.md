# Voice Input

Say a wake word and Arc listens. No terminal or browser tab needs to be focused.

```
You:  "Hey Jarvis"
🔔    *chime*
You:  "Search for flights from Delhi to Mumbai next week"
Arc:  *runs the query, sends you a desktop notification with results*
You:  "Sort by cheapest"    ← no wake word needed, conversation is still active
Arc:  *responds*
      ... 30 seconds of silence ...
Arc:  *goes back to sleep*
```

## How It Works

- **Wake word** — `openwakeword` detects "Hey Jarvis" (configurable) at ~2% CPU idle
- **Speech-to-text** — `faster-whisper` (offline, local, ~150MB model) transcribes after wake word
- **Gateway client** — sends transcribed text to `arc gateway` via WebSocket
- **Conversation flow** — 4-state machine: SLEEPING → ACTIVE → PROCESSING → LISTENING (30s follow-up)
- **Cross-platform** — conversations appear live in WebChat and Telegram
- **Mic goes deaf** during processing — your meetings and background chatter are never captured

## Setup

```bash
pip install arc-agent[voice]

# Terminal 1:
arc gateway

# Terminal 2:
arc listen
```

## Configuration

```toml
[voice]
wake_model = "hey_jarvis"     # also: alexa, hey_mycroft, hey_rhasspy
whisper_model = "base.en"     # tiny.en (faster) or small.en (more accurate)
silence_duration = 1.5        # seconds of silence = end of speech
listen_timeout = 30.0         # seconds before going back to sleep
```
