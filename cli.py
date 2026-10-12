#!/usr/bin/env python3
"""
cli.py - command-line front-end for key-aware pitch correction.

Supports a single file or an entire folder (batch mode). In batch mode, every
audio file in the input folder is corrected and written to the output folder
under its ORIGINAL filename -- the source folder itself is never touched
unless you explicitly pass --overwrite.

Examples
--------
# Single file, manually specified key
python cli.py vocal.wav -o vocal_tuned.wav --key "A minor"

# Single file, auto-detect the key from the instrumental/song
python cli.py vocal.wav -o vocal_tuned.wav --key-from song.wav

# Looser, more natural-sounding correction
python cli.py vocal.wav -o vocal_tuned.wav --key "C major" --strength 0.6 --retune-ms 60

# Batch: process every audio file in a folder, same filenames, new folder
python cli.py ./takes -o ./takes_tuned --key "A minor"

# Batch: output filenames same as input, written into a NEW folder next to it
python cli.py ./takes -o ./takes --key "A minor" --overwrite
#   (only needed if --output resolves to the SAME folder as the input)
"""

import argparse
import os
import sys

import librosa

from scale_utils import parse_key, detect_key_from_audio
from autotune import autotune_file

AUDIO_EXTENSIONS = (".wav", ".flac", ".aiff", ".aif", ".mp3", ".ogg", ".m4a")


def build_parser():
    p = argparse.ArgumentParser(
        description="Pitch-correct a mono vocal (or a whole folder of them) to a musical key.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", help="Vocal audio file OR a folder of vocal files (batch mode)")
    p.add_argument("-o", "--output", required=True,
                    help="Output file path (single mode) or output folder (batch mode, "
                         "or single mode if you want to keep the original filename)")

    key_group = p.add_mutually_exclusive_group(required=True)
    key_group.add_argument(
        "--key", help='Target key, e.g. "C major", "A minor", "F# dorian"'
    )
    key_group.add_argument(
        "--key-from", metavar="SONG_FILE",
        help="Auto-detect the key from a reference song/instrumental file"
    )

    p.add_argument("--strength", type=float, default=1.0,
                    help="0.0-1.0, how strongly to snap to the scale (default 1.0 = full correction)")
    p.add_argument("--retune-ms", type=float, default=20.0,
                    help="Correction speed in ms; higher = slower/more natural, lower = snappier/robotic (default 20)")
    p.add_argument("--fmin", type=float, default=80.0, help="Min vocal pitch to detect, Hz (default 80)")
    p.add_argument("--fmax", type=float, default=1000.0, help="Max vocal pitch to detect, Hz (default 1000)")
    p.add_argument("--no-humanize", action="store_true",
                    help="Disable re-injection of natural vibrato/micro-pitch variation")
    p.add_argument("--overwrite", action="store_true",
                    help="Allow writing over original file(s) when input and output resolve "
                         "to the same path/folder. Without this flag, that case is refused.")
    return p


def resolve_key(args):
    """Resolve --key or --key-from into a KeySpec, printing what was chosen."""
    if args.key_from:
        print(f"Detecting key from '{args.key_from}' ...")
        y, sr = librosa.load(args.key_from, sr=None, mono=True)
        result = detect_key_from_audio(y, sr)
        key = result["key"]
        print(f"Detected key: {key.name}  "
              f"(confidence {result['confidence']:.2f}, margin {result['margin']:.2f})")
        print("Top candidates:", ", ".join(f"{name} ({score:.2f})" for name, score in result["ranked"]))
        return key
    key = parse_key(args.key)
    print(f"Target key: {key.name}")
    return key


def find_audio_files(folder):
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(AUDIO_EXTENSIONS) and os.path.isfile(os.path.join(folder, f))
    )


def guard_against_overwrite(src_path, dst_path, overwrite_flag):
    """Refuse to write dst_path over src_path unless --overwrite was passed."""
    if os.path.abspath(src_path) == os.path.abspath(dst_path) and not overwrite_flag:
        sys.exit(
            f"Refusing to overwrite original file:\n  {src_path}\n"
            f"Pass --overwrite to allow this, or choose a different --output path/folder."
        )


def run_single(args, key):
    input_path = args.input
    output_arg = args.output

    # If --output is (or looks like) a directory, keep the original filename inside it.
    if os.path.isdir(output_arg) or output_arg.endswith(os.sep):
        os.makedirs(output_arg, exist_ok=True)
        output_path = os.path.join(output_arg, os.path.basename(input_path))
    else:
        output_path = output_arg
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    guard_against_overwrite(input_path, output_path, args.overwrite)

    print(f"Correcting '{input_path}' -> '{output_path}' "
          f"(strength={args.strength}, retune_ms={args.retune_ms}) ...")
    autotune_file(
        input_path, output_path, key=key,
        strength=args.strength, retune_ms=args.retune_ms,
        fmin=args.fmin, fmax=args.fmax,
        humanize_vibrato=not args.no_humanize,
    )
    print(f"Done. Wrote {output_path}")


def run_batch(args, key):
    input_dir = args.input
    output_dir = args.output

    if os.path.abspath(input_dir) == os.path.abspath(output_dir) and not args.overwrite:
        sys.exit(
            f"Output folder is the same as the input folder:\n  {input_dir}\n"
            f"This would overwrite your original files. Pass --overwrite to confirm, "
            f"or choose a different --output folder."
        )

    os.makedirs(output_dir, exist_ok=True)
    files = find_audio_files(input_dir)
    if not files:
        print(f"No audio files found in '{input_dir}' (looked for {', '.join(AUDIO_EXTENSIONS)})")
        return

    print(f"Found {len(files)} audio file(s) in '{input_dir}'. Correcting to {key.name} ...")
    ok, failed = 0, []
    for i, fname in enumerate(files, 1):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)  # same filename, new folder
        print(f"[{i}/{len(files)}] {fname} ...", end=" ", flush=True)
        try:
            autotune_file(
                in_path, out_path, key=key,
                strength=args.strength, retune_ms=args.retune_ms,
                fmin=args.fmin, fmax=args.fmax,
                humanize_vibrato=not args.no_humanize,
            )
            print("done")
            ok += 1
        except Exception as e:
            print(f"FAILED ({e})")
            failed.append(fname)

    print(f"\nBatch complete: {ok}/{len(files)} succeeded, written to '{output_dir}' "
          f"with original filenames.")
    if failed:
        print("Failed files:", ", ".join(failed))


def main(argv=None):
    args = build_parser().parse_args(argv)
    key = resolve_key(args)

    if os.path.isdir(args.input):
        run_batch(args, key)
    else:
        run_single(args, key)


if __name__ == "__main__":
    sys.exit(main())
