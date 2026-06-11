#!/usr/bin/env python3
"""
verify_bu_pipeline.py
═════════════════════════════════════════════════════════════════════════════
Verification script for annotation_pipeline_bu.ipynb output.

Reads raw BU Radio News Corpus files (.ala, .brk, .ton) and a Text2ToBI
batch JSON file, then re-derives the expected labels from scratch and
compares them token-by-token against the JSON.

Usage
─────
    python verify_bu_pipeline.py \\
        --corpus-root /path/to/bu_corpus \\
        --labels-dir  /path/to/labels/bu \\
        [--brk-window 0.03] \\
        [--ton-window 0.20] \\
        [--verbose]

Returns exit code 0 if all samples pass, 1 if any discrepancy is found.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                       ARCHITECTURE REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEXT2TOBI MODEL OVERVIEW
─────────────────────────
Text2ToBI is a DistilBERT-based multi-task token classifier that predicts
ToBI prosodic boundary structure from TEXT ALONE — no audio at inference
time.  It has three prediction heads:

  b  (boundary detection)    — binary: is this word a phrase boundary?
  i  (intonation type)       — 4-class: none / rising / falling / level
  x  (break index)           — string label: "" / "3" / "4"

Each head is supervised by a different label source during training.
The BU corpus is the highest-quality supervision source in the pipeline
because its labels are GOLD STANDARD — human-annotated ToBI, not
model-derived silver labels.


WHAT IS ToBI?
─────────────
ToBI (Tones and Break Indices) is a transcription system for English
prosody developed by Beckman & Ayers (1994).  It captures:

  • Pitch accents   (e.g. H*, L+H*, !H*)  — nuclear and prenuclear tones
  • Phrase accents  (e.g. H-, L-)          — intermediate phrase edge tones
  • Boundary tones  (e.g. H%, L%)          — intonational phrase edges
  • Break indices   (0–4)                  — degree of juncture between words

This pipeline uses break indices and boundary tones only; pitch accents
are not currently modelled by Text2ToBI.


WHAT IS A BREAK INDEX?
──────────────────────
A break index encodes the STRENGTH OF PROSODIC JUNCTURE between two words:

  0  — clitic attachment (e.g. "gonna", "wanna"); no perceived boundary
  1  — normal word boundary; minimal disjuncture
  2  — stronger-than-normal boundary; possible lengthening/pause, but
        NOT a full prosodic phrase boundary (sometimes called a
        "prosodic word" boundary in the literature)
  3  — intermediate phrase (ip) boundary; marked edge tone (H- or L-)
        but NOT a full intonational phrase
  4  — intonational phrase (IP) boundary; full boundary tone (H% or L%),
        often with final lengthening and/or pause

In the BU corpus, break indices are stored in .brk files.
Each entry's TIMESTAMP marks the ONSET of the word that FOLLOWS the break
(verified empirically: 75/75 entries in f2bs02p3 match word-start timing
within 30 ms).  This is a BU corpus convention, not a universal standard.

TEXT2TOBI MAPPING — BREAK INDEX → b AND x LABELS
──────────────────────────────────────────────────
  break index 0  →  b = 0,  x = ""    (no boundary; clitic)
  break index 1  →  b = 0,  x = ""    (no boundary; normal word juncture)
  break index 2  →  b = 0,  x = ""    (no boundary; prosodic word edge)
  break index 3  →  b = 1,  x = "3"   (intermediate phrase boundary)
  break index 4  →  b = 1,  x = "4"   (intonational phrase boundary)

Rationale: indices 0–2 do not constitute phrase boundaries in standard
ToBI theory.  Only 3 and 4 trigger a phrasal reset that is detectable
from text-level features.  Index 2 is a prosodic-word boundary, which is
linguistically interesting but not what the model is trained to predict.

The x head provides finer supervision than b: it distinguishes intermediate
from intonational phrases.  This is useful because IP boundaries
systematically co-occur with full boundary tones (H% / L%) while ip
boundaries end in phrase accents (H- / L-) without a boundary tone.


WHAT IS AN INTONATION UNIT (IU) — AND WHY BU DOESN'T USE THEM
───────────────────────────────────────────────────────────────
The SBCSAE pipeline is segmented into Intonation Units (IUs) — the
discourse-phonological chunks described by Chafe (1994) and used by
Du Bois in the Santa Barbara Corpus annotation scheme.  IUs typically
contain one new or focused information unit and are bounded by pitch
resets, pauses, or boundary tones.

The BU Radio News Corpus is NOT segmented into IUs.  It consists of
individual radio news sentences read aloud by professional speakers —
a planned, scripted register.  Each utterance file (.ala/.brk/.ton
triplet) corresponds to one read sentence.

Because IU segmentation is absent and forcing it would require unsupported
assumptions (e.g. treating IP boundaries as IU surrogates, then windowing
across utterances that cross story or speaker lines), the BU pipeline
uses ONE UTTERANCE = ONE SAMPLE.  No windowing is applied.

This means BU samples are longer on average than SBCSAE samples
(which were windowed at 30 IUs), but the label quality is higher
because every boundary comes from hand-annotation, not model consensus.


WHAT THE .ala FILE CONTAINS
────────────────────────────
.ala is a phone-level alignment file (forced-alignment output).
Format:

    H#    0   20          ← silence symbol, start (cs), duration (cs)
    >endsil               ← word marker: CLOSES the phones above it
    AE   20   12          ← phones for the next word ("and")
    N    32    9
    D    43    1
    >and                  ← word marker: CLOSES the phones above it ("and")
    V    44   11          ← phones for the next word ("Virginia")
    ...

CRITICAL: the '>' marker names the word whose phones PRECEDE it.
It appears AFTER its phones, not before.

Timing units are CENTISECONDS (1/100 s).
  e.g. start=20 cs → 0.200 s,  duration=12 cs → end=0.320 s

Silence/pause words ('sil', 'endsil', 'sp', 'pau') are EXCLUDED from
the token sequence — they are not words and have no break index labels.

DISFLUENCY MARKERS: curly braces mark false starts / self-corrections.
  '{and'   → opening brace: start of a disfluent region
  '}didn't → closing brace: end of the disfluent region (repair)

These are REAL SPOKEN WORDS with valid .brk entries.  The braces are
STRIPPED ('{and' → 'and') but the tokens are KEPT.  Removing disfluent
words would shift the token sequence relative to the .brk timing entries,
breaking alignment.

Rationale for keeping disfluencies: Text2ToBI is designed to run on raw
transcripts, not cleaned text.  Disfluency robustness is a design goal.


WHAT THE .brk FILE CONTAINS
────────────────────────────
.brk stores break indices.  Format (after a header terminated by '#'):

    0.430000   76   2          ← time_s  color_code  break_index
    0.960000   76   1          ← (color_code is a display artifact, ignored)
    1.330000   76   1
    ...
    4.500000   76   4  ; 6     ← optional '; N' word-count comment, ignored
    23.500000  76  3-          ← trailing '-' is a variant notation, stripped

Inline '; …' comments are stripped.
Variant break index notations like '3-' are normalised to '3'.


WHAT THE .ton FILE CONTAINS
────────────────────────────
.ton stores ToBI tone events.  Format (after a header terminated by '#'):

    0.778s   color_code   L+H*        ← pitch accent (no boundary tone)
    1.640s   color_code   L-          ← phrase accent (intermediate phrase edge)
    2.430s   color_code   L-L%        ← full boundary tone (IP edge: falling)
    13.635s  color_code   L-H%        ← full boundary tone (IP edge: rising)

Tone label anatomy (e.g. 'L-L%'):
  L-   phrase accent    — marks intermediate phrase edge
  L%   boundary tone   — marks intonational phrase edge

We extract ONLY the boundary tone component (the rightmost '-'-delimited
component that ends in '%').  Non-boundary events (pitch accents H*, L+H*,
!H*; phrase accents L-, H-; HiF0 markers) are ignored.

TONE LABEL → INTONATION MAPPING
──────────────────────────────────
  H%   → rising  (1)   — e.g. L-H%  (low phrase accent + high boundary)
  L%   → falling (2)   — e.g. L-L%  (!H-L%)
  %    → level   (3)   — standalone % only; rare in BU Radio News Corpus
  (no match)  → none (0)   — boundary with no .ton entry within window

Leading accent modifiers ('!', '+') are stripped before matching:
  '!H%' → 'H%' → rising
  '+H%' → 'H%' → rising

Note on '!H' (downstepped H): downstep affects pitch height but not the
boundary tone category — a downstepped high boundary is still rising
relative to the utterance-final pitch.


ALIGNMENT BETWEEN .brk AND .ala
────────────────────────────────
.brk timestamps mark word ONSET (start of the following word).
.ala gives us word onset via the first phone's start timestamp.

For each .brk entry, we find the word whose start_s is within
BRK_ALIGN_WINDOW_S seconds (default: 30 ms).

30 ms was chosen based on empirical verification on f2bs02p3:
  - 75/75 .brk entries matched a word onset within 30 ms
  - 0/75 needed more than 30 ms (max observed diff: ~10 ms)
The window is small to avoid false assignments across nearby words.

If no word onset falls within BRK_ALIGN_WINDOW_S of a .brk timestamp,
the entry is skipped (typically happens for leading silence entries
that precede the first lexical word).


ALIGNMENT BETWEEN .ton AND BOUNDARY WORDS
──────────────────────────────────────────
.ton timestamps are placed independently of .brk timestamps by the
human annotator.  They mark the acoustic location of the tone event,
which may differ from the word onset by up to several hundred ms.

For each boundary word (b=1), we search .ton entries within
TON_ALIGN_WINDOW_S seconds (default: 200 ms) of the word's onset,
considering only entries that carry a boundary tone component.
The nearest match wins.

200 ms was chosen as a generous window that captures annotator timing
variance without crossing into the next boundary.  If a boundary word
has no .ton entry within 200 ms, i_label = 0 (none) — this is normal
and expected for intermediate phrase (brk=3) boundaries, which have
phrase accents (H-, L-) rather than boundary tones (H%, L%).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Words in .ala that represent silence/pause and should be excluded from
# the token sequence.  These have no lexical content and no .brk entries.
SILENCE_WORDS = {"sil", "endsil", "sp", "pau"}

# Break indices that constitute PHRASE boundaries in ToBI.
# 0–2 are not phrase boundaries; only 3 (ip) and 4 (IP) are.
BOUNDARY_BREAK_INDICES = {3, 4}

# Intonation integer labels
INTON_NONE    = 0   # no boundary tone found within window
INTON_RISING  = 1   # H% boundary tone
INTON_FALLING = 2   # L% boundary tone
INTON_LEVEL   = 3   # standalone % (rare)

INTON_NAMES = {
    INTON_NONE:    "none",
    INTON_RISING:  "rising",
    INTON_FALLING: "falling",
    INTON_LEVEL:   "level",
}

# Boundary tone string → intonation label
TONE_MAP = {
    "H%": INTON_RISING,
    "L%": INTON_FALLING,
    "%":  INTON_LEVEL,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA STRUCTURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Word:
    token:   str
    start_s: float
    end_s:   float

@dataclass
class BrkEntry:
    time_s: float
    index:  int

@dataclass
class TonEntry:
    time_s:        float
    label:         str
    boundary_tone: str   # extracted %-bearing component, or ""

@dataclass
class Discrepancy:
    """One token-level mismatch between expected and actual labels."""
    utt_id:       str
    token_index:  int
    token:        str

    # b head
    b_expected:   int
    b_actual:     int

    # i head
    i_expected:   int
    i_actual:     int
    # Alignment trace for i (populated when i differs)
    i_word_onset_s:       Optional[float] = None
    i_nearest_ton_time_s: Optional[float] = None
    i_nearest_ton_label:  Optional[str]   = None
    i_nearest_ton_dist_s: Optional[float] = None
    i_nearest_within_window: Optional[bool] = None

    # x head
    x_expected:   str = ""
    x_actual:     str = ""

    def is_real(self) -> bool:
        return (self.b_expected != self.b_actual or
                self.i_expected != self.i_actual or
                self.x_expected != self.x_actual)

    def summary_line(self) -> str:
        parts = []
        if self.b_expected != self.b_actual:
            parts.append(
                f"b: expected={self.b_expected} actual={self.b_actual}"
            )
        if self.i_expected != self.i_actual:
            ie = INTON_NAMES[self.i_expected]
            ia = INTON_NAMES[self.i_actual]
            trace = ""
            if self.i_word_onset_s is not None:
                trace = (
                    f" | word_onset={self.i_word_onset_s:.3f}s"
                )
                if self.i_nearest_ton_time_s is not None:
                    w = "✓" if self.i_nearest_within_window else "✗"
                    trace += (
                        f" nearest_ton={self.i_nearest_ton_label!r}"
                        f"@{self.i_nearest_ton_time_s:.3f}s"
                        f" dist={self.i_nearest_ton_dist_s*1000:.0f}ms"
                        f" in_window={w}"
                    )
                else:
                    trace += " no_ton_entry_with_boundary_tone"
            parts.append(
                f"i: expected={ie}({self.i_expected})"
                f" actual={ia}({self.i_actual}){trace}"
            )
        if self.x_expected != self.x_actual:
            parts.append(
                f"x: expected={self.x_expected!r} actual={self.x_actual!r}"
            )
        return "; ".join(parts)

    def input_string(self) -> str:
        """Human-readable representation of the gold-standard input."""
        return (
            f'utt="{self.utt_id}"  '
            f'token[{self.token_index}]="{self.token}"'
        )

    def output_string(self) -> str:
        """Human-readable representation of the actual pipeline output."""
        return self.summary_line()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PARSERS  (mirror annotation_pipeline_bu.ipynb exactly)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_ala(path: str) -> List[Word]:
    """
    Parse a .ala phone-alignment file into a list of Word objects.

    Only '>' word-marker lines produce tokens; phoneme lines supply
    timing only.  The '>' marker closes the phonemes ABOVE it, naming
    the word those phones belong to.

    Timing units: centiseconds (divide by 100 to get seconds).
    Silence words (sil, endsil, sp, pau) are excluded.
    Disfluency braces {} are stripped; the word token is kept.
    """
    words: List[Word] = []
    current_word:  Optional[str]   = None
    current_start: Optional[int]   = None   # centiseconds
    current_end:   Optional[int]   = None

    def _flush():
        nonlocal current_word, current_start, current_end
        if (current_word is not None
                and current_word.lower() not in SILENCE_WORDS
                and current_start is not None):
            token = re.sub(r"[{}]", "", current_word)
            if token:
                words.append(Word(
                    token   = token,
                    start_s = current_start / 100.0,
                    end_s   = current_end   / 100.0,
                ))
        current_word  = None
        current_start = None
        current_end   = None

    with open(path, encoding="ascii", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                current_word = line[1:]
            else:
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    start = int(parts[1])
                    dur   = int(parts[2])
                except ValueError:
                    continue
                if current_start is None:
                    current_start = start
                current_end = start + dur

    _flush()   # last word (no trailing '>' in some files)
    return words


def parse_brk(path: str) -> List[BrkEntry]:
    """
    Parse a .brk break-index file.

    Lines before the '#' sentinel are header metadata and are skipped.
    Inline '; ...' comments are stripped.
    Variant notations like '3-' are normalised by stripping non-digits.
    """
    entries: List[BrkEntry] = []
    in_data = False

    with open(path, encoding="ascii", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line == "#":
                in_data = True
                continue
            if not in_data:
                continue
            main = line.split(";")[0].strip()
            parts = main.split()
            if len(parts) < 3:
                continue
            try:
                time_s = float(parts[0])
            except ValueError:
                continue
            idx_str = re.sub(r"[^0-9]", "", parts[2])
            if not idx_str:
                continue
            entries.append(BrkEntry(time_s=time_s, index=int(idx_str)))

    return entries


def _extract_boundary_tone(label: str) -> str:
    """
    Extract the boundary tone component from a compound ToBI tone label.

    Split on '-' (phrase boundary marker) and return the rightmost
    component that ends in '%', stripping leading accent modifiers
    ('!' downstep, '+' upstep).

    Examples:
        'L-L%'   → 'L%'   (falling IP boundary)
        'L-H%'   → 'H%'   (rising  IP boundary)
        '!H-L%'  → 'L%'   (downstepped falling)
        'H*'     → ''     (pitch accent, no boundary tone)
        'L-'     → ''     (phrase accent only, no boundary tone)
        'HiF0'   → ''     (high F0 marker, not a boundary)
    """
    components = re.split(r"-", label)
    for comp in reversed(components):
        if comp.endswith("%"):
            return re.sub(r"^[!+]*", "", comp)
    return ""


def parse_ton(path: str) -> List[TonEntry]:
    """
    Parse a .ton intonation-event file.

    Lines before the '#' sentinel are header metadata and are skipped.
    Only entries with a non-empty boundary_tone are useful for the i head,
    but we return all entries so the verifier can compute nearest-neighbor
    distances over the full set.
    """
    entries: List[TonEntry] = []
    in_data = False

    with open(path, encoding="ascii", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line == "#":
                in_data = True
                continue
            if not in_data:
                continue
            main = line.split(";")[0].strip()
            parts = main.split()
            if len(parts) < 3:
                continue
            try:
                time_s = float(parts[0])
            except ValueError:
                continue
            label = parts[2]
            entries.append(TonEntry(
                time_s        = time_s,
                label         = label,
                boundary_tone = _extract_boundary_tone(label),
            ))

    return entries


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ALIGNMENT  (mirror annotation_pipeline_bu.ipynb exactly)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def derive_expected_labels(
    words:       List[Word],
    brk_entries: List[BrkEntry],
    ton_entries: List[TonEntry],
    brk_window:  float,
    ton_window:  float,
) -> Tuple[List[int], List[int], List[str]]:
    """
    Re-derive b, i, x label vectors from raw parsed data.

    This mirrors the align_labels() function in annotation_pipeline_bu.ipynb
    exactly.  If this function produces output that matches the JSON, the
    pipeline is correct.

    Returns
    -------
    b_labels : list[int]   — 0 or 1
    i_labels : list[int]   — 0/1/2/3
    x_labels : list[str]  — "3" / "4" / ""
    """
    n = len(words)
    b_labels: List[int] = [0] * n
    i_labels: List[int] = [0] * n
    x_labels: List[str] = [""] * n

    if not brk_entries:
        return b_labels, i_labels, x_labels

    # ── Step 1: assign break index to words ──────────────────────────
    for entry in brk_entries:
        t   = entry.time_s
        idx = entry.index
        best_i = min(range(n), key=lambda i: abs(words[i].start_s - t))
        dist   = abs(words[best_i].start_s - t)
        if dist > brk_window:
            continue
        if idx in BOUNDARY_BREAK_INDICES:
            b_labels[best_i] = 1
            x_labels[best_i] = str(idx)

    # ── Step 2: assign intonation to boundary words ───────────────────
    if ton_entries:
        for i, w in enumerate(words):
            if b_labels[i] == 0:
                continue
            t = w.start_s
            candidates = [
                e for e in ton_entries
                if abs(e.time_s - t) <= ton_window and e.boundary_tone
            ]
            if not candidates:
                continue
            nearest = min(candidates, key=lambda e: abs(e.time_s - t))
            clean   = re.sub(r"^[!+]*", "", nearest.boundary_tone)
            i_labels[i] = TONE_MAP.get(clean, INTON_NONE)

    return b_labels, i_labels, x_labels


def derive_alignment_trace(
    word:        Word,
    ton_entries: List[TonEntry],
    ton_window:  float,
) -> dict:
    """
    Produce a detailed trace of the .ton nearest-neighbor search for
    one boundary word, to include in discrepancy reports.
    """
    t = word.start_s
    # All entries with a boundary tone, regardless of window
    bt_entries = [e for e in ton_entries if e.boundary_tone]
    if not bt_entries:
        return {
            "word_onset_s":        t,
            "nearest_time_s":      None,
            "nearest_label":       None,
            "nearest_dist_s":      None,
            "nearest_within_window": None,
        }
    nearest = min(bt_entries, key=lambda e: abs(e.time_s - t))
    dist    = abs(nearest.time_s - t)
    return {
        "word_onset_s":        t,
        "nearest_time_s":      nearest.time_s,
        "nearest_label":       nearest.label,
        "nearest_dist_s":      dist,
        "nearest_within_window": dist <= ton_window,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VERIFICATION CORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def verify_utterance(
    utt_id:      str,
    stem:        str,
    actual:      dict,
    brk_window:  float,
    ton_window:  float,
    verbose:     bool,
) -> List[Discrepancy]:
    """
    Verify one utterance sample.

    Parameters
    ----------
    utt_id   : sample key in the batch JSON (e.g. 'f2bs02p3')
    stem     : filesystem path stem (no extension) to the corpus files
    actual   : the JSON sample dict {"b": ..., "i": ..., "x": ...}
    ...

    Returns
    -------
    List of Discrepancy objects (empty = perfect match).
    """
    discrepancies: List[Discrepancy] = []

    # ── Parse gold-standard files ──────────────────────────────────
    ala_path = stem + ".ala"
    brk_path = stem + ".brk"
    ton_path = stem + ".ton"

    if not os.path.exists(ala_path):
        print(f"  [SKIP] {utt_id}: .ala not found at {ala_path}")
        return discrepancies
    if not os.path.exists(brk_path):
        print(f"  [SKIP] {utt_id}: .brk not found at {brk_path}")
        return discrepancies

    words       = parse_ala(ala_path)
    brk_entries = parse_brk(brk_path)
    ton_entries = parse_ton(ton_path) if os.path.exists(ton_path) else []

    # ── Derive expected labels ────────────────────────────────────
    b_exp, i_exp, x_exp = derive_expected_labels(
        words, brk_entries, ton_entries, brk_window, ton_window
    )

    # ── Extract actual labels from JSON ───────────────────────────
    act_tokens    = actual["b"]["tokens"]
    b_act         = actual["b"]["consensus"]
    i_act         = actual["i"]["labels"]
    x_act         = actual["x"]["labels"]

    # ── Structural checks ─────────────────────────────────────────
    exp_tokens = [w.token for w in words]

    if len(exp_tokens) != len(act_tokens):
        print(
            f"  [ERROR] {utt_id}: token count mismatch "
            f"(expected {len(exp_tokens)}, actual {len(act_tokens)})"
        )
        print(f"    expected: {exp_tokens[:10]}...")
        print(f"    actual:   {act_tokens[:10]}...")
        # Return a single structural discrepancy
        discrepancies.append(Discrepancy(
            utt_id=utt_id, token_index=-1,
            token="[STRUCTURAL]",
            b_expected=-1, b_actual=-1,
            i_expected=-1, i_actual=-1,
            x_expected=f"len={len(exp_tokens)}", x_actual=f"len={len(act_tokens)}",
        ))
        return discrepancies

    for j, (tok_exp, tok_act) in enumerate(zip(exp_tokens, act_tokens)):
        if tok_exp != tok_act:
            discrepancies.append(Discrepancy(
                utt_id=utt_id, token_index=j,
                token=f"[TOKEN MISMATCH: exp={tok_exp!r} act={tok_act!r}]",
                b_expected=-1, b_actual=-1,
                i_expected=-1, i_actual=-1,
                x_expected="", x_actual="",
            ))

    if any(d.token_index >= 0 and d.b_expected == -1 for d in discrepancies):
        return discrepancies   # token sequence wrong; label comparison meaningless

    # ── Label-by-label comparison ─────────────────────────────────
    for j in range(len(words)):
        be = b_exp[j]; ba = b_act[j]
        ie = i_exp[j]; ia = i_act[j]
        xe = x_exp[j]; xa = x_act[j]

        # Convert actual x from int to str if the JSON stored it as int
        if isinstance(xa, int):
            xa = str(xa) if xa else ""
        if isinstance(xe, int):
            xe = str(xe) if xe else ""

        if be == ba and ie == ia and xe == xa:
            continue   # all three heads match

        d = Discrepancy(
            utt_id=utt_id, token_index=j,
            token=words[j].token,
            b_expected=be, b_actual=ba,
            i_expected=ie, i_actual=ia,
            x_expected=xe, x_actual=xa,
        )

        # Enrich with alignment trace if i differs
        if ie != ia and ton_entries:
            trace = derive_alignment_trace(words[j], ton_entries, ton_window)
            d.i_word_onset_s       = trace["word_onset_s"]
            d.i_nearest_ton_time_s = trace["nearest_time_s"]
            d.i_nearest_ton_label  = trace["nearest_label"]
            d.i_nearest_ton_dist_s = trace["nearest_dist_s"]
            d.i_nearest_within_window = trace["nearest_within_window"]

        discrepancies.append(d)

    if verbose and not discrepancies:
        print(f"  [OK]  {utt_id}: {len(words)} tokens, {sum(b_exp)} boundaries")

    return discrepancies


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEM LOOKUP  —  map utterance ID to corpus file stem
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_stem_index(corpus_root: str) -> dict:
    """
    Walk the corpus tree and build a dict mapping utterance_id → stem path.
    e.g.  "f2bs02p3" → "/path/to/bu_corpus/data/f2b/s/radio/f2bs02p3"
    """
    index = {}
    for ala_path in glob.glob(
        os.path.join(corpus_root, "**", "*.ala"), recursive=True
    ):
        stem   = os.path.splitext(ala_path)[0]
        utt_id = os.path.basename(stem)
        index[utt_id] = stem
    return index


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Verify annotation_pipeline_bu.ipynb output against BU corpus gold standard."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus-root", required=True,
        help="Root of the BU Radio News Corpus (CD-ROM layout: contains data/f1a/...)",
    )
    parser.add_argument(
        "--labels-dir", required=True,
        help="Directory containing batch_NNNN.json files from the pipeline",
    )
    parser.add_argument(
        "--brk-window", type=float, default=0.03,
        help="Max distance (s) for .brk→word-onset alignment (default: 0.03)",
    )
    parser.add_argument(
        "--ton-window", type=float, default=0.20,
        help="Max distance (s) for .ton→boundary-word alignment (default: 0.20)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print a confirmation line for every passing utterance",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Text2ToBI BU Pipeline Verifier")
    print("=" * 70)
    print(f"  corpus_root : {args.corpus_root}")
    print(f"  labels_dir  : {args.labels_dir}")
    print(f"  brk_window  : {args.brk_window*1000:.0f} ms")
    print(f"  ton_window  : {args.ton_window*1000:.0f} ms")
    print()

    # ── Load all batch JSON files ─────────────────────────────────
    batch_files = sorted(
        f for f in os.listdir(args.labels_dir)
        if f.startswith("batch_") and f.endswith(".json")
    )
    if not batch_files:
        print(f"ERROR: no batch_*.json files found in {args.labels_dir}")
        sys.exit(1)

    all_samples: dict = {}
    for fname in batch_files:
        path = os.path.join(args.labels_dir, fname)
        with open(path) as fh:
            all_samples.update(json.load(fh))

    print(f"  Loaded {len(all_samples)} samples from {len(batch_files)} batch file(s).")

    # ── Build corpus stem index ───────────────────────────────────
    stem_index = build_stem_index(args.corpus_root)
    print(f"  Found {len(stem_index)} .ala files in corpus.")

    # ── Verify each sample ────────────────────────────────────────
    all_discrepancies: List[Discrepancy] = []
    n_checked  = 0
    n_skipped  = 0
    n_passing  = 0

    for utt_id, sample in sorted(all_samples.items()):
        if utt_id not in stem_index:
            print(f"  [SKIP] {utt_id}: not found in corpus (check corpus_root layout)")
            n_skipped += 1
            continue

        stem = stem_index[utt_id]
        discs = verify_utterance(
            utt_id     = utt_id,
            stem       = stem,
            actual     = sample,
            brk_window = args.brk_window,
            ton_window = args.ton_window,
            verbose    = args.verbose,
        )
        n_checked += 1
        if discs:
            all_discrepancies.extend(discs)
        else:
            n_passing += 1

    # ── Summary ───────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"  Samples checked  : {n_checked}")
    print(f"  Samples skipped  : {n_skipped}  (not found in corpus)")
    print(f"  Samples passing  : {n_passing}")
    print(f"  Samples failing  : {n_checked - n_passing}")
    print(f"  Total discrepancies: {len(all_discrepancies)}")

    if not all_discrepancies:
        print()
        print("  RESULT: TRUE — all samples match the gold standard exactly.")
        print("  The pipeline is verified.")
        print()
        return True, []

    # ── Discrepancy breakdown ─────────────────────────────────────
    b_disc  = [d for d in all_discrepancies if d.b_expected != d.b_actual]
    i_disc  = [d for d in all_discrepancies if d.i_expected != d.i_actual]
    x_disc  = [d for d in all_discrepancies if d.x_expected != d.x_actual]

    print()
    print(f"  Discrepancies by head:")
    print(f"    b (boundary)  : {len(b_disc)}")
    print(f"    i (intonation): {len(i_disc)}")
    print(f"    x (break idx) : {len(x_disc)}")

    # ── i discrepancy window analysis ────────────────────────────
    if i_disc:
        print()
        print("  Intonation discrepancy alignment trace:")
        # Bucket by whether the nearest .ton entry was inside vs outside window
        inside_wrong  = [d for d in i_disc
                         if d.i_nearest_within_window is True]
        outside_win   = [d for d in i_disc
                         if d.i_nearest_within_window is False]
        no_ton        = [d for d in i_disc
                         if d.i_nearest_ton_time_s is None]
        print(f"    nearest .ton INSIDE  window but still wrong : {len(inside_wrong)}")
        print(f"    nearest .ton OUTSIDE window (tone missed)    : {len(outside_win)}")
        print(f"    no boundary-tone .ton entry at all           : {len(no_ton)}")

        if outside_win:
            dists = [d.i_nearest_ton_dist_s * 1000 for d in outside_win
                     if d.i_nearest_ton_dist_s is not None]
            if dists:
                print(f"    distance stats for outside-window cases (ms):")
                print(f"      min={min(dists):.0f}  max={max(dists):.0f}"
                      f"  mean={sum(dists)/len(dists):.0f}")
                print(f"    → consider increasing --ton-window if max distance"
                      f" is systematically above {args.ton_window*1000:.0f} ms")

    # ── Detailed discrepancy listing ─────────────────────────────
    print()
    print("  DISCREPANCY DETAILS  (input string → output string):")
    print("  " + "-" * 66)
    for d in all_discrepancies:
        if not d.is_real():
            continue
        print(f"  INPUT : {d.input_string()}")
        print(f"  OUTPUT: {d.output_string()}")
        print()

    print()
    print("  RESULT: FALSE — discrepancies found (see above).")
    print()

    message_lines = [
        f"[{d.input_string()}] → [{d.output_string()}]"
        for d in all_discrepancies
        if d.is_real()
    ]
    return False, message_lines


if __name__ == "__main__":
    result, messages = main()
    sys.exit(0 if result else 1)
