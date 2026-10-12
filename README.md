# Pitch Correction

Auto-tune with pitch detection, key-aware snapping and natural-sounding resynthesis.

A Python auto-tune-style tool: detect a mono vocal's pitch, snap it to the notes of a
musical key, and resynthesize with formants preserved (so the voice doesn't sound
"chipmunked"). Comes with a command-line tool and a Gradio web UI, both with batch
processing for whole folders of takes.

## How it works

1. **Pitch detection** — [`librosa.pyin`](https://librosa.org/) (probabilistic YIN)
   estimates the fundamental frequency of the vocal, frame by frame.
2. **Key-aware snapping** — each frame's pitch is converted to a MIDI note number and
   snapped to the *closest* note that's actually in your target key/scale (e.g. only
   C, D, E, F, G, A, B for C major).
3. **Smoothing / "retune speed"** — like a hardware auto-tune's retune-speed knob, the
   correction eases in over a configurable number of milliseconds instead of snapping
   instantly, and a fraction of the singer's original vibrato is optionally re-added
   on top so the result doesn't sound robotic.
4. **Resynthesis** — the corrected pitch curve is applied via Praat-based
   pitch-synchronous overlap-add (PSOLA, through the [`psola`](https://pypi.org/project/psola/)
   package), which preserves the vocal's formants/timbre far better than a naive
   resample-based pitch shift.

There's also an optional **key auto-detector**: point it at the instrumental/full
song and it estimates the key using chroma features correlated against the
Krumhansl-Kessler tonal-hierarchy profiles (the standard music-cognition algorithm for
this), so you don't have to know music theory to figure out what key you're in.

## Install

Requires Python 3.10 or newer (developed on 3.13).

```bash
git clone https://github.com/JohnnyRaha/pitch-correction.git
cd pitch-correction
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

`psola` uses Praat under the hood (via `parselmouth`) — no separate Praat install
needed, it's bundled. Reading `.mp3` / `.m4a` input may additionally need
[FFmpeg](https://ffmpeg.org/) on your PATH; `.wav` and `.flac` work out of the box.

## Usage

### Command line (`cli.py`)

```bash
# Manually specify the key
python cli.py vocal.wav -o vocal_tuned.wav --key "A minor"

# Auto-detect the key from the song/instrumental instead
python cli.py vocal.wav -o vocal_tuned.wav --key-from song_instrumental.wav

# Looser / more natural-sounding correction
python cli.py vocal.wav -o vocal_tuned.wav --key "C major" --strength 0.6 --retune-ms 60
```

Key names accept things like `"C major"`, `"A minor"`, `"F# dorian"`, `"Bb"` (defaults
to major), `"G#m"` (minor shorthand). Supported scales: `major`, `minor`,
`harmonic_minor`, `melodic_minor`, `dorian`, `mixolydian`, `chromatic` (chromatic =
no correction, passes every note — useful as a no-op sanity check).

Run `python cli.py -h` for the full flag list (pitch-detection range, humanize toggle, etc).

#### Batch mode (whole folder)

Point the CLI at a folder instead of a single file to correct every audio file in it.
Each output keeps its **original filename** — it's written into the output folder you
specify, so your source files are never touched:

```bash
python cli.py ./takes -o ./takes_tuned --key "A minor"
```

`./takes/verse_take2.wav` → `./takes_tuned/verse_take2.wav`, etc. If `--output` resolves
to the *same* folder as the input (or, in single-file mode, the same exact path), the
tool refuses and tells you to either pick a different output location or pass
`--overwrite` to confirm you really want to replace the originals.

### Web UI (`app.py`)

```bash
python app.py
```

Starts a local Gradio server and prints its URL (http://127.0.0.1:7860 by default) —
open that in a browser. The interface has two tabs:

- **Single File** — upload a vocal, pick a key manually or auto-detect it from a
  reference song, and audition the corrected result inline before downloading.
- **Batch Folder** — select a whole folder (or multi-select files) of takes; every file
  is corrected and packaged into a `.zip` where each entry keeps its original filename.

## Tuning parameters for your voice

- **`fmin` / `fmax`** — set these to comfortably bracket your vocal range (defaults:
  80–1000 Hz) so the pitch tracker doesn't lock onto breath noise or harmonics.
- **`strength`** — 1.0 = full correction (hard-tune / "T-Pain" effect at low
  `retune-ms`); lower values (0.3–0.7) leave some of your natural pitch bend for a
  more subtle, "just nudge it in tune" effect.
- **`retune-ms`** — lower (5–15 ms) = fast/snappy/obvious correction; higher
  (50–150 ms) = slower, smoother, more natural-sounding glide into pitch.
- **`humanize_vibrato`** — keeps a portion of your natural vibrato/expression riding
  on top of the corrected pitch rather than flattening it out completely.

## Files

- `scale_utils.py` — key/scale parsing, note snapping math, key auto-detection.
- `autotune.py` — pitch detection, correction-curve generation, PSOLA resynthesis
  (importable as a library: `autotune_file(...)` / `autotune_array(...)`).
- `cli.py` — command-line front-end.
- `app.py` — Gradio web front-end.

## Notes / limitations

- Works on **mono** vocals (stereo input is auto-downmixed to mono).
- Best results come from a clean, dry vocal take (minimal reverb/effects) — heavy
  effects can confuse the pitch tracker.
- This corrects *pitch only*, not timing/rhythm.
- For an even more natural feel on sustained notes, try `retune-ms` around 40–80 with
  `strength` around 0.7–0.85 as a starting point, then adjust to taste.

## License

[MIT](LICENSE) © 2026 John Rahaman
