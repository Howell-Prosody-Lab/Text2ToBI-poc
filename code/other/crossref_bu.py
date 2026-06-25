#!/usr/bin/env python3
"""
crossref_bu.py
──────────────
Prints a single merged table for one BU utterance, showing for each word:
  - its timing from .ala
  - the .brk entry that matched it (timestamp, raw index, distance)
  - the .ton entry that matched it (timestamp, raw label, distance)
  - the derived b / i / x labels from the JSON

Every source is visible on the same row, so discrepancies are immediately
obvious without any mental joining of separate files.

Usage
─────
    python crossref_bu.py <utterance_id> <corpus_root> <labels_dir>

Example
───────
    python crossref_bu.py f2bs02p3 /path/to/bu_corpus /path/to/labels/bu

Columns
───────
  #          token index
  TOKEN      orthographic word (disfluency braces stripped)
  ALA        word onset – offset in seconds (from .ala phoneme timing)
  BRK_T      timestamp of the .brk entry that matched this word ('—' if none)
  BRK_IDX    raw break index value ('—' if none)
  BRK_DIST   distance in ms between .brk timestamp and word onset
  TON_T      timestamp of the nearest boundary-tone .ton entry ('—' if none)
  TON_LABEL  full tone label (e.g. L-L%, L-H%)
  TON_DIST   distance in ms between .ton timestamp and word onset
  IN_WIN     whether the .ton entry was within the 200ms tolerance window
  b          derived boundary label (0/1)
  i          derived intonation label (none/rising/falling/level)
  x          derived break index string ("3"/"4"/"")
"""

import glob, json, os, re, sys

# ── Configuration (must match annotation_pipeline_bu.ipynb Cell 1) ──
BRK_ALIGN_WINDOW_S = 0.03
TON_ALIGN_WINDOW_S = 0.20
BOUNDARY_BREAK_INDICES = {3, 4}
SILENCE_WORDS = {"sil", "endsil", "sp", "pau"}
INTON_NAMES = {0: "none", 1: "rising", 2: "falling", 3: "level"}
TONE_MAP = {"H%": 1, "L%": 2, "%": 3}

# ── Parsers ──────────────────────────────────────────────────────────

def parse_ala(path):
    words = []; cw = None; cs = None; ce = None
    with open(path, encoding="ascii", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line: continue
            if line.startswith(">"):
                if cw and cw.lower() not in SILENCE_WORDS and cs is not None:
                    tok = re.sub(r"[{}]", "", cw)
                    if tok: words.append({"token": tok, "start_s": cs/100.0, "end_s": ce/100.0})
                cw = line[1:]; cs = None; ce = None
            else:
                parts = line.split()
                if len(parts) < 3: continue
                try:
                    s = int(parts[1]); d = int(parts[2])
                    if cs is None: cs = s
                    ce = s + d
                except ValueError: continue
    if cw and cw.lower() not in SILENCE_WORDS and cs is not None:
        tok = re.sub(r"[{}]", "", cw)
        if tok: words.append({"token": tok, "start_s": cs/100.0, "end_s": ce/100.0})
    return words

def parse_brk(path):
    entries = []; in_data = False
    with open(path, encoding="ascii", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line: continue
            if line == "#": in_data = True; continue
            if not in_data: continue
            main = line.split(";")[0].strip(); parts = main.split()
            if len(parts) < 3: continue
            try: t = float(parts[0])
            except ValueError: continue
            idx = re.sub(r"[^0-9]", "", parts[2])
            if idx: entries.append({"time_s": t, "index": int(idx)})
    return entries

def parse_ton(path):
    entries = []; in_data = False
    with open(path, encoding="ascii", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line: continue
            if line == "#": in_data = True; continue
            if not in_data: continue
            main = line.split(";")[0].strip(); parts = main.split()
            if len(parts) < 3: continue
            try: t = float(parts[0])
            except ValueError: continue
            label = parts[2]
            comps = re.split(r"-", label)
            bt = ""
            for c in reversed(comps):
                if c.endswith("%"): bt = re.sub(r"^[!+]*", "", c); break
            entries.append({"time_s": t, "label": label, "boundary_tone": bt})
    return entries

# ── Alignment (mirrors pipeline exactly) ────────────────────────────

def align(words, brk_entries, ton_entries):
    """
    Returns one row dict per word with all source values + derived labels.
    """
    n = len(words)
    rows = []

    # Pre-assign each .brk entry to its best word
    brk_matched = {}   # word_index → brk entry
    for e in brk_entries:
        best = min(range(n), key=lambda j: abs(words[j]["start_s"] - e["time_s"]))
        dist = abs(words[best]["start_s"] - e["time_s"])
        if dist <= BRK_ALIGN_WINDOW_S:
            brk_matched[best] = {"entry": e, "dist_s": dist}

    for j, w in enumerate(words):
        # ── brk ──────────────────────────────────────────────────
        brk_hit = brk_matched.get(j)
        if brk_hit:
            brk_idx = brk_hit["entry"]["index"]
            brk_t   = brk_hit["entry"]["time_s"]
            brk_d   = brk_hit["dist_s"]
            is_boundary = brk_idx in BOUNDARY_BREAK_INDICES
        else:
            brk_idx = None; brk_t = None; brk_d = None
            is_boundary = False

        b_label = 1 if is_boundary else 0
        x_label = str(brk_idx) if is_boundary else ""

        # ── ton ───────────────────────────────────────────────────
        # Show nearest boundary-tone entry for ALL words (not just boundaries)
        # so you can spot misalignments at non-boundary positions too
        ton_t = ton_label = ton_bt = ton_d = None
        ton_in_win = None
        i_label = 0

        bt_entries = [e for e in ton_entries if e["boundary_tone"]]
        if bt_entries:
            nearest = min(bt_entries, key=lambda e: abs(e["time_s"] - w["start_s"]))
            ton_t     = nearest["time_s"]
            ton_label = nearest["label"]
            ton_bt    = nearest["boundary_tone"]
            ton_d     = abs(nearest["time_s"] - w["start_s"])
            ton_in_win = ton_d <= TON_ALIGN_WINDOW_S

            if is_boundary and ton_in_win:
                clean = re.sub(r"^[!+]*", "", ton_bt)
                i_label = TONE_MAP.get(clean, 0)

        rows.append({
            "token":      w["token"],
            "ala_start":  w["start_s"],
            "ala_end":    w["end_s"],
            "brk_t":      brk_t,
            "brk_idx":    brk_idx,
            "brk_d_ms":   brk_d * 1000 if brk_d is not None else None,
            "ton_t":      ton_t,
            "ton_label":  ton_label,
            "ton_d_ms":   ton_d * 1000 if ton_d is not None else None,
            "ton_in_win": ton_in_win,
            "b":          b_label,
            "i":          i_label,
            "x":          x_label,
        })

    return rows

# ── Helpers ──────────────────────────────────────────────────────────

def find_stem(corpus_root, utt_id):
    matches = glob.glob(os.path.join(corpus_root, "**", utt_id + ".ala"), recursive=True)
    if not matches:
        sys.exit(f"ERROR: could not find {utt_id}.ala under {corpus_root}")
    return os.path.splitext(matches[0])[0]

def find_sample(labels_dir, utt_id):
    for fname in sorted(f for f in os.listdir(labels_dir)
                        if f.startswith("batch_") and f.endswith(".json")):
        with open(os.path.join(labels_dir, fname)) as fh:
            batch = json.load(fh)
        if utt_id in batch:
            return batch[utt_id], fname
    sys.exit(f"ERROR: {utt_id} not found in any batch file in {labels_dir}")

def fmt(val, fmt_str, dash_if_none=True):
    if val is None:
        return "—" if dash_if_none else ""
    return format(val, fmt_str)

# ── Main ─────────────────────────────────────────────────────────────

if len(sys.argv) != 4:
    sys.exit("Usage: crossref_bu.py <utterance_id> <corpus_root> <labels_dir>")

utt_id      = sys.argv[1]
corpus_root = sys.argv[2]
labels_dir  = sys.argv[3]

stem           = find_stem(corpus_root, utt_id)
sample, bfname = find_sample(labels_dir, utt_id)

words       = parse_ala(stem + ".ala")
brk_entries = parse_brk(stem + ".brk")
ton_entries = parse_ton(stem + ".ton") if os.path.exists(stem + ".ton") else []
rows        = align(words, brk_entries, ton_entries)

# ── Cross-check derived labels against JSON ───────────────────────
json_tokens = sample["b"]["tokens"]
json_b      = sample["b"]["consensus"]
json_i      = sample["i"]["labels"]
json_x      = sample["x"]["labels"]

mismatches = []
if len(rows) != len(json_tokens):
    print(f"WARNING: token count mismatch — derived {len(rows)}, JSON has {len(json_tokens)}")
else:
    for j, (row, jt, jb, ji, jx) in enumerate(zip(rows, json_tokens, json_b, json_i, json_x)):
        if row["token"] != jt or row["b"] != jb or row["i"] != ji or row["x"] != str(jx) if jx else row["x"] != "":
            mismatches.append(j)

# ── Print table ───────────────────────────────────────────────────
W_TOK = max(len(r["token"]) for r in rows) + 1
W_TOK = max(W_TOK, 8)

HDR = (f"{'#':>3}  {'TOKEN':<{W_TOK}}  "
       f"{'ALA_START':>9}  {'ALA_END':>7}  "
       f"{'BRK_T':>7}  {'IDX':>3}  {'DIST':>6}  "
       f"{'TON_T':>7}  {'TON_LABEL':<12}  {'DIST':>6}  {'WIN':>3}  "
       f"{'b':>2}  {'i':<9}  {'x':>4}")
SEP = "─" * len(HDR)

print(f"\n  Utterance : {utt_id}")
print(f"  Source    : {stem}")
print(f"  JSON      : {bfname}")
print(f"  Tokens    : {len(rows)}  |  Boundaries: {sum(r['b'] for r in rows)}")
print(f"  BRK window: {BRK_ALIGN_WINDOW_S*1000:.0f}ms  "
      f"TON window: {TON_ALIGN_WINDOW_S*1000:.0f}ms")
if mismatches:
    print(f"  ⚠  JSON MISMATCH at token indices: {mismatches}")
else:
    print(f"  ✓  All derived labels match JSON exactly")
print()
print(HDR)
print(SEP)

for j, row in enumerate(rows):
    is_boundary = row["b"] == 1
    mismatch    = j in mismatches

    # Only show ton columns when the entry is within window (or it's a boundary)
    # For non-boundary words, show ton only if it's suspiciously close (< window)
    show_ton = is_boundary or (row["ton_in_win"] is True)

    brk_t_s   = fmt(row["brk_t"],   ".3f")
    brk_idx_s = fmt(row["brk_idx"], "d")
    brk_d_s   = fmt(row["brk_d_ms"], ".0f") + "ms" if row["brk_d_ms"] is not None else "—"

    if show_ton:
        ton_t_s   = fmt(row["ton_t"],   ".3f")
        ton_lbl_s = row["ton_label"] or "—"
        ton_d_s   = fmt(row["ton_d_ms"], ".0f") + "ms" if row["ton_d_ms"] is not None else "—"
        win_s     = "✓" if row["ton_in_win"] else "✗"
    else:
        ton_t_s = ton_lbl_s = ton_d_s = win_s = "—"

    flag = " ← BOUNDARY" if is_boundary else ""
    flag = flag + " ⚠ MISMATCH" if mismatch else flag

    line = (f"  {j:>3}  {row['token']:<{W_TOK}}  "
            f"  {row['ala_start']:>7.3f}s  {row['ala_end']:>5.3f}s  "
            f"  {brk_t_s:>7}  {brk_idx_s:>3}  {brk_d_s:>6}  "
            f"  {ton_t_s:>7}  {ton_lbl_s:<12}  {ton_d_s:>6}  {win_s:>3}  "
            f"  {row['b']:>2}  {INTON_NAMES[row['i']]:<9}  {row['x']!r:>4}"
            f"{flag}")
    print(line)

print(SEP)
print(f"  {len(rows)} tokens  |  rising={sum(1 for r in rows if r['i']==1)}  "
      f"falling={sum(1 for r in rows if r['i']==2)}  "
      f"level={sum(1 for r in rows if r['i']==3)}  "
      f"none@boundary={sum(1 for r in rows if r['b']==1 and r['i']==0)}")
print()
