"""
scale_utils.py

Music-theory helpers for key-aware pitch correction:
  - parsing key names like "C major" / "F# minor" / "Bb"
  - snapping a frequency (Hz) to the nearest note in a given key/scale
  - estimating the musical key of a reference audio file (Krumhansl-Schmuckler)
"""

import re
import numpy as np
import librosa

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Enharmonic spellings a user might type -> index into NOTE_NAMES
_NOTE_ALIASES = {
    "C": 0, "B#": 0,
    "C#": 1, "DB": 1,
    "D": 2,
    "D#": 3, "EB": 3,
    "E": 4, "FB": 4,
    "F": 5, "E#": 5,
    "F#": 6, "GB": 6,
    "G": 7,
    "G#": 8, "AB": 8,
    "A": 9,
    "A#": 10, "BB": 10,
    "B": 11, "CB": 11,
}

# Semitone offsets from the root for each supported scale/mode
SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],          # natural minor (aeolian)
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],  # ascending form
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "chromatic": list(range(12)),             # no correction, all notes allowed
}

_SCALE_ALIASES = {
    "maj": "major", "major": "major", "ionian": "major",
    "min": "minor", "minor": "minor", "aeolian": "minor", "natural minor": "minor",
    "harmonic minor": "harmonic_minor", "harmonicminor": "harmonic_minor",
    "melodic minor": "melodic_minor", "melodicminor": "melodic_minor",
    "dorian": "dorian",
    "mixolydian": "mixolydian",
    "chromatic": "chromatic", "none": "chromatic",
}

A4_HZ = 440.0
A4_MIDI = 69


class KeySpec:
    """A resolved musical key: root pitch class (0-11) + scale intervals."""

    def __init__(self, root_pc: int, scale_name: str):
        self.root_pc = root_pc
        self.scale_name = scale_name
        self.intervals = SCALE_INTERVALS[scale_name]
        # Absolute pitch classes (0-11) that are "in key"
        self.allowed_pcs = sorted({(root_pc + i) % 12 for i in self.intervals})

    @property
    def name(self):
        return f"{NOTE_NAMES[self.root_pc]} {self.scale_name.replace('_', ' ')}"

    def __repr__(self):
        return f"<KeySpec {self.name}>"


def parse_key(key_str: str) -> KeySpec:
    """Parse strings like 'C major', 'A minor', 'F# dorian', 'Bb', 'G#m'."""
    s = key_str.strip()
    if not s:
        raise ValueError("Empty key string")

    # Split into note part and (optional) scale part, e.g. "F#m" -> "F#", "m"
    match = re.match(r"^\s*([A-Ga-g][#b]?)\s*(.*)$", s)
    if not match:
        raise ValueError(f"Could not parse key: '{key_str}'")

    note_part, rest = match.group(1).upper(), match.group(2).strip().lower()
    if note_part not in _NOTE_ALIASES:
        raise ValueError(f"Unrecognized note name: '{note_part}'")
    root_pc = _NOTE_ALIASES[note_part]

    if rest in ("", "maj", "major"):
        scale_name = "major"
    elif rest in ("m", "min", "minor"):
        scale_name = "minor"
    elif rest in _SCALE_ALIASES:
        scale_name = _SCALE_ALIASES[rest]
    else:
        raise ValueError(
            f"Unrecognized scale/mode: '{rest}'. "
            f"Supported: major, minor, harmonic_minor, melodic_minor, dorian, "
            f"mixolydian, chromatic"
        )

    return KeySpec(root_pc, scale_name)


def hz_to_midi(freq):
    freq = np.asarray(freq, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = A4_MIDI + 12 * np.log2(freq / A4_HZ)
    return midi


def midi_to_hz(midi):
    midi = np.asarray(midi, dtype=float)
    return A4_HZ * (2.0 ** ((midi - A4_MIDI) / 12.0))


def nearest_scale_midi(midi_val: float, allowed_pcs) -> float:
    """Return the MIDI note number, among those whose pitch class is in
    `allowed_pcs`, that is closest to the continuous value `midi_val`.

    Searches a window of nearby integers and picks the true closest by
    distance to the (fractional) input value -- not just the first in-scale
    neighbor encountered -- so e.g. a note 0.53 semitones below E and 1.47
    semitones above D correctly snaps to E, not D.
    """
    if np.isnan(midi_val):
        return np.nan
    base = int(np.round(midi_val))
    best_candidate = None
    best_dist = None
    for candidate in range(base - 7, base + 8):
        if candidate % 12 in allowed_pcs:
            dist = abs(candidate - midi_val)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_candidate = candidate
    if best_candidate is None:
        return float(base)  # should never hit (chromatic covers everything)
    return float(best_candidate)


def snap_frequencies(f0: np.ndarray, key: KeySpec) -> np.ndarray:
    """Snap an array of frequencies (NaN = unvoiced) to the nearest note
    in `key`. Returns an array of target frequencies (Hz), NaN preserved."""
    midi = hz_to_midi(f0)
    target_midi = np.array(
        [nearest_scale_midi(m, key.allowed_pcs) for m in midi], dtype=float
    )
    return midi_to_hz(target_midi)


# ---------------------------------------------------------------------------
# Key detection from a reference audio file (Krumhansl-Schmuckler algorithm)
# ---------------------------------------------------------------------------

# Krumhansl & Kessler (1982) tonal hierarchy profiles
_KS_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KS_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def detect_key_from_audio(y: np.ndarray, sr: int) -> dict:
    """Estimate the musical key of an audio signal using chroma features
    correlated against the Krumhansl-Kessler major/minor key profiles.

    Returns a dict with the best key, its KeySpec, mode, and a confidence
    score plus the full ranked list of candidates.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    profile = chroma.mean(axis=1)
    profile = profile / (profile.sum() + 1e-9)

    def corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    candidates = []
    for root in range(12):
        maj = np.roll(_KS_MAJOR_PROFILE, root)
        minr = np.roll(_KS_MINOR_PROFILE, root)
        candidates.append((corr(profile, maj), root, "major"))
        candidates.append((corr(profile, minr), root, "minor"))

    candidates.sort(key=lambda c: c[0], reverse=True)
    best_score, best_root, best_mode = candidates[0]
    second_score = candidates[1][0]

    return {
        "key": KeySpec(best_root, best_mode),
        "root_pc": best_root,
        "mode": best_mode,
        "confidence": best_score,
        "margin": best_score - second_score,
        "ranked": [
            (KeySpec(r, m).name, score) for score, r, m in candidates[:5]
        ],
    }
