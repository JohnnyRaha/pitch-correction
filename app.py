"""
app.py - Gradio UI for key-aware pitch correction.

Run locally with:
    python app.py
Then open the printed local URL in a browser.

Has two tabs:
  - Single File: correct one vocal, preview it inline.
  - Batch Folder: select a whole folder of takes, correct them all, and
    download a .zip where each file keeps its ORIGINAL filename.
"""

import tempfile
import os
import zipfile

import gradio as gr
import librosa
import soundfile as sf

from scale_utils import parse_key, detect_key_from_audio, NOTE_NAMES
from autotune import AutotuneSettings, autotune_array

SCALE_CHOICES = ["major", "minor", "harmonic_minor", "melodic_minor", "dorian", "mixolydian", "chromatic"]


def detect_key_ui(song_file):
    if song_file is None:
        return "Upload a song/instrumental first.", gr.update(), gr.update()
    y, sr = librosa.load(song_file, sr=None, mono=True)
    result = detect_key_from_audio(y, sr)
    key = result["key"]
    top5 = "  |  ".join(f"{name} ({score:.2f})" for name, score in result["ranked"])
    msg = (f"Detected key: **{key.name}**  "
           f"(confidence {result['confidence']:.2f}, margin over runner-up {result['margin']:.2f})\n\n"
           f"Top candidates: {top5}")
    return msg, gr.update(value=NOTE_NAMES[key.root_pc]), gr.update(value=key.scale_name)


def run_autotune(vocal_file, root_note, scale_name, strength, retune_ms, fmin, fmax, humanize):
    if vocal_file is None:
        raise gr.Error("Please upload a mono vocal audio file to correct.")
    if fmin >= fmax:
        raise gr.Error("Min pitch must be lower than max pitch.")

    key = parse_key(f"{root_note} {scale_name}")
    y, sr = librosa.load(vocal_file, sr=None, mono=True)

    settings = AutotuneSettings(
        key=key, strength=strength, retune_ms=retune_ms,
        fmin=fmin, fmax=fmax, humanize_vibrato=humanize,
    )
    tuned = autotune_array(y, sr, settings)

    out_path = os.path.join(tempfile.gettempdir(), "pitch_corrected_output.wav")
    sf.write(out_path, tuned, sr)
    return out_path, f"Corrected to **{key.name}**."


def run_batch_ui(files, root_note, scale_name, strength, retune_ms, fmin, fmax, humanize,
                  progress=gr.Progress()):
    if not files:
        raise gr.Error("Please select a folder (or multiple files) of vocal takes to correct.")
    if fmin >= fmax:
        raise gr.Error("Min pitch must be lower than max pitch.")

    key = parse_key(f"{root_note} {scale_name}")

    out_dir = tempfile.mkdtemp(prefix="pitch_corrected_batch_")
    written, failed = [], []

    for f in progress.tqdm(files, desc="Correcting takes"):
        in_path = f if isinstance(f, str) else f.name
        fname = os.path.basename(in_path)
        try:
            y, sr = librosa.load(in_path, sr=None, mono=True)
            settings = AutotuneSettings(
                key=key, strength=strength, retune_ms=retune_ms,
                fmin=fmin, fmax=fmax, humanize_vibrato=humanize,
            )
            tuned = autotune_array(y, sr, settings)
            out_path = os.path.join(out_dir, fname)  # same filename, new folder
            sf.write(out_path, tuned, sr)
            written.append(out_path)
        except Exception as e:
            failed.append(f"{fname} ({e})")

    if not written:
        raise gr.Error("None of the files could be processed. " + "; ".join(failed))

    zip_path = os.path.join(tempfile.gettempdir(), "pitch_corrected_batch.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in written:
            zf.write(path, arcname=os.path.basename(path))  # original filenames inside zip

    msg = f"Corrected {len(written)}/{len(files)} file(s) to **{key.name}**. Filenames preserved inside the zip."
    if failed:
        msg += "\n\nFailed: " + "; ".join(failed)
    return zip_path, msg


with gr.Blocks(title="Key-Aware Pitch Correction") as demo:
    gr.Markdown(
        "# 🎤 Key-Aware Pitch Correction\n"
        "Snap vocal pitch to a musical key — auto-tune style, with adjustable strength and retune speed."
    )

    with gr.Tabs():
        with gr.Tab("Single File"):
            with gr.Row():
                with gr.Column():
                    vocal_input = gr.Audio(label="Vocal to correct", type="filepath")

                    with gr.Accordion("Auto-detect key from a reference song", open=False):
                        song_input = gr.Audio(label="Reference song / instrumental", type="filepath")
                        detect_btn = gr.Button("Detect key")
                        detect_output = gr.Markdown()

                    with gr.Row():
                        root_dropdown = gr.Dropdown(NOTE_NAMES, value="C", label="Root note")
                        scale_dropdown = gr.Dropdown(SCALE_CHOICES, value="major", label="Scale")

                    strength_slider = gr.Slider(0.0, 1.0, value=1.0, step=0.05,
                                                 label="Correction strength (0 = off, 1 = full snap)")
                    retune_slider = gr.Slider(1, 200, value=20, step=1,
                                               label="Retune speed (ms) — lower = snappier/more robotic, higher = smoother/more natural")

                    with gr.Row():
                        fmin_slider = gr.Slider(40, 400, value=80, step=1, label="Min pitch to detect (Hz)")
                        fmax_slider = gr.Slider(400, 1500, value=1000, step=1, label="Max pitch to detect (Hz)")

                    humanize_checkbox = gr.Checkbox(value=True, label="Preserve natural vibrato/expression")

                    run_btn = gr.Button("Correct pitch", variant="primary")

                with gr.Column():
                    output_audio = gr.Audio(label="Corrected vocal", type="filepath")
                    status_md = gr.Markdown()

            detect_btn.click(
                detect_key_ui,
                inputs=[song_input],
                outputs=[detect_output, root_dropdown, scale_dropdown],
            )

            run_btn.click(
                run_autotune,
                inputs=[vocal_input, root_dropdown, scale_dropdown, strength_slider,
                        retune_slider, fmin_slider, fmax_slider, humanize_checkbox],
                outputs=[output_audio, status_md],
            )

        with gr.Tab("Batch Folder"):
            gr.Markdown(
                "Select a **folder** of vocal takes (or multi-select files). Each one is corrected "
                "and packaged into a .zip, keeping its **original filename** — nothing overwrites "
                "your source folder."
            )
            with gr.Row():
                with gr.Column():
                    batch_input = gr.File(
                        label="Folder of vocal takes", file_count="directory", type="filepath"
                    )

                    with gr.Row():
                        batch_root_dropdown = gr.Dropdown(NOTE_NAMES, value="C", label="Root note")
                        batch_scale_dropdown = gr.Dropdown(SCALE_CHOICES, value="major", label="Scale")

                    batch_strength_slider = gr.Slider(0.0, 1.0, value=1.0, step=0.05,
                                                        label="Correction strength (0 = off, 1 = full snap)")
                    batch_retune_slider = gr.Slider(1, 200, value=20, step=1,
                                                      label="Retune speed (ms)")

                    with gr.Row():
                        batch_fmin_slider = gr.Slider(40, 400, value=80, step=1, label="Min pitch to detect (Hz)")
                        batch_fmax_slider = gr.Slider(400, 1500, value=1000, step=1, label="Max pitch to detect (Hz)")

                    batch_humanize_checkbox = gr.Checkbox(value=True, label="Preserve natural vibrato/expression")

                    batch_run_btn = gr.Button("Correct all files", variant="primary")

                with gr.Column():
                    batch_output_zip = gr.File(label="Download corrected files (.zip)")
                    batch_status_md = gr.Markdown()

            batch_run_btn.click(
                run_batch_ui,
                inputs=[batch_input, batch_root_dropdown, batch_scale_dropdown, batch_strength_slider,
                        batch_retune_slider, batch_fmin_slider, batch_fmax_slider, batch_humanize_checkbox],
                outputs=[batch_output_zip, batch_status_md],
            )

if __name__ == "__main__":
    demo.launch()
