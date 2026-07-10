# Gemini Sparring Analyzer

A small personal-analysis CLI for boxing sparring video. It uploads one video to the Gemini Files API, waits for processing to complete, asks Gemini for structured decision-making feedback, saves the result as JSON, and deletes the remote file by default.

## Setup

1. Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Change into this directory, create the project virtual environment, and install the dependency:

   ```bash
   cd /Users/kgw7401/atom
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   ```

3. Export the API key for the current shell. Do not put it in source code or commit it.

   ```bash
   export GEMINI_API_KEY="your-key"
   ```

## Analyze One Round

```bash
.venv/bin/python analyze_sparring.py /path/to/round.mp4 \
  --me "blue headgear and black T-shirt" \
  --context "I freeze after defending against pressure"
```

The default output is next to the input video:

```text
round.analysis.json
```

The JSON contains the candidate exchange timestamps, visible opponent trigger, your visible response, category, confidence, repeated pattern, one next-round rule, and model limitations.

## Options

```bash
.venv/bin/python analyze_sparring.py --help
```

Use `--keep-remote-file` only when you intentionally want to reuse the file for another request. Otherwise the script deletes it after analysis. Gemini also automatically deletes Files API uploads after 48 hours.

## First Test

Use one 2-3 minute round where both boxers and their feet remain in frame. Pass a concrete description in `--me`; do not rely on "the boxer on the left", since positions change during sparring.

The output is evidence for review, not ground truth. Verify the listed timestamps yourself. Gemini's default video sampling can miss fast punch-level details, so interpret the output as decision-sequence feedback rather than exact punch or scoring analysis.
