# plaudnico

**Local meeting transcription on a Mac: who said what, plus a written report.**

Record a meeting with your phone or Mac. `plaud` transcribes it, separates the speakers and asks Claude to write the minutes.
Audio never leaves your machine; only the transcript text is sent to Claude.

```console
$ plaud process meeting.m4a
~/plaud/archive/2026/2026-10-03_1451_meeting
Converting audio…
Diarizing…
Transcribing with mlx-community/whisper-large-v3-mlx…
Transcript: …/transcript.md
Writing report 'minutes' with Claude…
Report: …/report-minutes.md
```

```markdown
**[00:00:00] Speaker 1:** Good morning. Today we review the simulation budget for next year.

**[00:00:04] Speaker 2:** Thanks. I think we need two more weeks on the cluster.

**[00:00:07] Speaker 1:** Agreed. Please send the request before Friday.
```

## Features

- **Speech to text on the Mac's GPU** with [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (Whisper large-v3). French, English and the 90+ other Whisper languages, detected automatically.
- **Speaker separation** with [pyannote](https://github.com/pyannote/pyannote-audio) (`speaker-diarization-community-1`): speakers get anonymous labels (`Speaker 1`, `Speaker 2`, …).
- **Reports written by Claude** from Markdown templates, in the language of the meeting.
- **One folder per meeting**, with every intermediate result kept. Rerunning is instant unless something changed.

## Privacy

This tool is built for recordings that may contain sensitive conversations.

- Transcription and speaker separation run **locally**. The models are downloaded once from Hugging Face, and inference runs offline.
- pyannote's and Hugging Face's **usage telemetry is switched off**.
- The report step runs `claude -p` **isolated**:
  - no tools, no MCP servers or connectors;
  - no user or project settings, no `CLAUDE.md`;
  - an empty working directory.
  
  Claude sees the transcript text and the template, nothing else.
- Claude is told never to guess speakers' real names.
- Audio, transcripts and reports stay in a local archive (`~/plaud/archive` by default).

**Recording people may require their consent.** In France, it does (Code pénal, art. 226-1). Announce that you are recording.

## Requirements

- macOS on Apple Silicon (M1 or later)
- [uv](https://docs.astral.sh/uv/) and [ffmpeg](https://ffmpeg.org/): `brew install uv ffmpeg`
- [Claude Code](https://claude.com/claude-code), logged in. Only needed for reports.
- A free [Hugging Face](https://huggingface.co) account, to download the speaker-separation model once:
  1. Accept the conditions of [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
  2. Create a **Read** token under *Settings → Access Tokens*.
  3. Add it to your shell profile: `export HF_TOKEN=hf_...`

## Install

```bash
git clone https://github.com/nicolasaunai/plaudnico.git
cd plaudnico
uv sync
```

The first run downloads the models (about 3 GB for Whisper large-v3, plus the pyannote model).

## Usage

```bash
uv run plaud process <audio file> [options]
```

| Option | Default | Description |
|---|---|---|
| `--template NAME[,NAME…]` | `minutes` | Report template(s) from `templates/`, or `none` for transcript only |
| `--language CODE` | auto | Force the language (`fr`, `en`, …) if detection gets it wrong |
| `--speakers N` | auto | Number of speakers, if you know it (helps separation) |
| `--model REPO` | `mlx-community/whisper-large-v3-mlx` | Any mlx-whisper model, e.g. `mlx-community/whisper-large-v3-turbo` (faster, less accurate) |
| `--device mps\|cpu` | `mps` | Device for speaker separation |
| `--force` | off | Recompute every stage |

Any format ffmpeg reads works: `.m4a` from iPhone Voice Memos (*Dictaphone* in French), `.mp3`, `.wav`… For best results, set Voice Memos to **Lossless** audio quality and put the phone in the middle of the table.

### Output

Each recording gets a folder named after its date and file name:

```
~/plaud/archive/2026/2026-10-03_1451_meeting/
├── audio.m4a           original recording (never modified)
├── meeting.json        id (content hash), source, start time, title
├── audio16k.wav        16 kHz mono copy used by the models
├── diar.json           speaker turns
├── asr.json            words with timestamps
├── turns.json          words with their speaker
├── transcript.md       readable transcript
└── report-minutes.md   report
```

Processing the same file again (even under another name) reuses its folder. Stages are only recomputed if their options changed or with `--force`. Set `PLAUD_ARCHIVE` to store the archive elsewhere.

### Templates

A template is a Markdown file of instructions in `templates/`. The shipped `minutes` template asks for a summary, the points discussed, decisions, an actions table (with owner) and open questions.

To add a format, drop a new file, e.g. `templates/actions.md`, then:

```bash
uv run plaud process meeting.m4a --template minutes,actions
```

## How it works

```
audio ─► ffmpeg (16 kHz mono) ─► pyannote: who speaks when ─┐
                               └► Whisper: words + times ────┴─► align ─► transcript.md ─► claude -p ─► report
```

Each word is given to the speaker turn it overlaps most. Words that fall in a gap go to the nearest turn less than 1 s away; otherwise they are marked `Unknown speaker`.

On an M-series MacBook, a 13-minute French meeting took:
- about 50 s for speaker separation,
- 2 min 15 s for transcription with large-v3 (about 6× real time),
- 3.5 min in total, report included.

## Development

```bash
uv run pytest            # fast unit tests, no models needed
uv run pytest -m slow    # real models; needs HF_TOKEN and macOS `say` voices
```

## Status and roadmap

Early but working (v0). Planned:
- naming speakers from a local voice library;
- a review page to fix speaker names;
- a glossary for jargon and names;
- calendar metadata;
- automatic pickup of new Voice Memos;
- recording online calls on the Mac;
- export to Obsidian.

Known limits:
- a speaker change can cut a sentence in two;
- language is detected once per file, so mixed-language meetings are not handled yet;
- overlapping speech and distant voices in large rooms reduce accuracy.
