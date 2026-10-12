"""
autotune.py

Core pitch-correction engine:
  1. detect the pitch contour of a mono vocal with librosa's pYIN
  2. snap each voiced frame to the nearest note in a target key
  3. smooth the correction curve to control how "robotic" vs "natural" it sounds
  4. resynthesize with Praat-based PSOLA (via the `psola` package), which
     preserves formants/timbre far better than a naive resampling pitch-shift

Usage (see cli.py / app.py for full front-ends):

    import librosa, soundfile as sf
    from autotune import autotune_file

    autotune_file("vocal.wav", "vocal_tuned.wav", key="A minor",
                   strength=1.0, retune_ms=20, fmin=80, fmax=1000)
"""

from dataclasses import dataclass

import numpy as np
import librosa
import soundfile as sf
import psola

from scale_utils import KeySpec, parse_key, hz_to_midi, midi_to_hz, nearest_scale_midi


@dataclass
class AutotuneSettings:
    key: KeySpec
    strength: float = 1.0     # 0 = no correction, 1 = fully snapped to scale
    retune_ms: float = 20.0   # smoothing time constant; higher = slower/more natural
    fmin: float = 80.0        # Hz, pitch-detection lower bound (vocal range)
    fmax: float = 1000.0      # Hz, pitch-detection upper bound
    frame_length: int = 2048
    hop_length: int = 256
    humanize_vibrato: bool = True  # keep natural vibrato riding on top of the snap


def _smooth_midi_curve(target_midi: np.ndarray, hop_length: int, sr: int,
                        retune_ms: float) -> np.ndarray:
    """Exponentially smooth the *target* pitch curve so correction eases in
    rather than snapping instantly (mimics a hardware auto-tune's
    "retune speed" control). NaNs (unvoiced) are passed through unchanged."""
    if retune_ms <= 0:
        return target_midi

    frame_time_ms = 1000.0 * hop_length / sr
    alpha = 1.0 - np.exp(-frame_time_ms / retune_ms)

    smoothed = np.copy(target_midi)
    last_valid = None
    for i in range(len(smoothed)):
        val = smoothed[i]
        if np.isnan(val):
            last_valid = None
            continue
        if last_valid is None:
            last_valid = val
        else:
            last_valid = last_valid + alpha * (val - last_valid)
        smoothed[i] = last_valid
    return smoothed


def correct_pitch_curve(f0: np.ndarray, settings: AutotuneSettings,
                         sr: int) -> np.ndarray:
    """Given a raw f0 contour (Hz, NaN = unvoiced), return the corrected
    target-frequency contour to feed into psola."""
    key = settings.key
    original_midi = hz_to_midi(f0)

    snapped_midi = np.array(
        [nearest_scale_midi(m, key.allowed_pcs) for m in original_midi],
        dtype=float,
    )

    # Blend original <-> fully-snapped by `strength` (in semitone space)
    with np.errstate(invalid="ignore"):
        blended_midi = original_midi + settings.strength * (snapped_midi - original_midi)

    # Ease the correction in over time instead of snapping instantly
    blended_midi = _smooth_midi_curve(blended_midi, settings.hop_length, sr,
                                       settings.retune_ms)

    if settings.humanize_vibrato:
        # Re-add natural micro-variation (vibrato/expression) that pure
        # semitone-smoothing can flatten out, by keeping a fraction of the
        # original curve's short-term deviation from its own smoothed trend.
        orig_smooth = _smooth_midi_curve(np.copy(original_midi), settings.hop_length,
                                          sr, settings.retune_ms)
        with np.errstate(invalid="ignore"):
            micro_variation = original_midi - orig_smooth
            blended_midi = blended_midi + 0.5 * micro_variation

    return midi_to_hz(blended_midi)


def autotune_array(y: np.ndarray, sr: int, settings: AutotuneSettings) -> np.ndarray:
    """Run the full pitch-correction pipeline on an in-memory mono signal."""
    if y.ndim > 1:
        y = librosa.to_mono(y.T if y.shape[0] < y.shape[1] else y)

    y = y.astype(np.float64)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=settings.fmin,
        fmax=settings.fmax,
        sr=sr,
        frame_length=settings.frame_length,
        hop_length=settings.hop_length,
    )

    target_f0 = correct_pitch_curve(f0, settings, sr)

    tuned = psola.vocode(
        y,
        sample_rate=int(sr),
        target_pitch=target_f0,
        fmin=settings.fmin,
        fmax=settings.fmax,
    )

    # psola can return audio a sample or two off in length; pad/trim to match
    if len(tuned) < len(y):
        tuned = np.pad(tuned, (0, len(y) - len(tuned)))
    else:
        tuned = tuned[: len(y)]

    return tuned.astype(np.float32)


def autotune_file(input_path: str, output_path: str, key,
                   strength: float = 1.0, retune_ms: float = 20.0,
                   fmin: float = 80.0, fmax: float = 1000.0,
                   humanize_vibrato: bool = True) -> str:
    """Load a mono (or stereo, auto-downmixed) vocal file, pitch-correct it
    to `key`, and write the result to `output_path`. Returns output_path."""
    key_spec = key if isinstance(key, KeySpec) else parse_key(key)

    y, sr = librosa.load(input_path, sr=None, mono=True)

    settings = AutotuneSettings(
        key=key_spec, strength=strength, retune_ms=retune_ms,
        fmin=fmin, fmax=fmax, humanize_vibrato=humanize_vibrato,
    )
    tuned = autotune_array(y, sr, settings)

    sf.write(output_path, tuned, sr)
    return output_path
