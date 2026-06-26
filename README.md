# text2tobi-poc

A research prototype for predicting ToBI prosodic structure from text alone — no audio required at inference time. This repository contains the full annotation pipeline, training notebooks, and experimental results for the capstone project underlying the [`text2tobi`](https://github.com/Howell-Prosody-Lab/Text2ToBI) package.

**Best result:** boundary F1 = **0.84** (Model #36: LibriTTS + People's Speech + SBCSAE, BLW=2.0) on the SBC001–005 gold held-out test set, against a GPT-Neo text-only baseline of 0.77. Intonation F1 = 0.58; break index F1 = 0.60 (0.65 at BLW=5.0) on the BU Radio News Corpus.

---

## Background

Prosody — the rhythm, melody, and phrasing of speech — is not arbitrary. It is partly determined by syntactic structure and semantic content, which means a model reading text has real signal to work with, even without audio. This project asks how well that signal can be exploited: given a sentence, can a model predict where a speaker would place a prosodic phrase boundary, and whether their pitch would rise or fall at that boundary?

The practical motivation is twofold. First, text-to-speech systems that predict prosody explicitly produce more interpretable and controllable output than systems that learn it implicitly end-to-end. Second, explicit symbolic prediction is tractable on CPU with a small model, making it applicable in low-resource and edge settings where large acoustic models are not viable.

The annotation framework used here is **ToBI** (Tones and Break Indices), the standard linguistic system for transcribing English prosody. ToBI encodes two things at each prosodic boundary: the **break index** (how strong the prosodic break is, on a scale of 0–4) and the **boundary tone** (the pitch movement at the boundary, e.g. H% for high/rising, L% for low/falling). This project predicts three quantities per word token:

- **Boundary detection** — is there a prosodic phrase boundary after this word? (binary)
- **Intonation type** — if so, is the boundary tone rising, falling, or level? (3-class)
- **Break index** — is this a minor (index 3) or major (index 4) prosodic break? (2-class)

---

## Model Architecture

The model is a multi-task token classifier built on top of `distilbert-base-uncased` (66M parameters). DistilBERT reads the full sentence bidirectionally — it sees context on both sides of each position simultaneously — which is a meaningful advantage over unidirectional language models for prosody prediction, since boundaries are often signaled by what comes *after* a word (e.g. a subject–verb boundary) as much as what precedes it.

Three classification heads are attached to DistilBERT's final hidden states:

```
DistilBERT encoder  [768-dim hidden states]
    [+ optional POS embedding addition, post-transformer]
    └─► dropout
         ├─► boundary_head     Linear(768 → 2)   all positions
         ├─► intonation_head   Linear(768 → 3)   rising / falling / level
         └─► break_idx_head    Linear(768 → 2)   index-3 / index-4
```

### Subword tokenization and label alignment

DistilBERT tokenizes text into subword pieces (e.g. *running* → [`run`, `##ning`]), but prosodic labels are inherently word-level. The alignment rule applied throughout is: the **first subword** of each word receives the word's label; all continuation subwords and special tokens (`[CLS]`, `[SEP]`, `[PAD]`) are assigned -100 and masked from the loss. This is standard practice for token classification tasks and ensures that label boundaries match word boundaries despite subword tokenization.

### POS embedding injection

An optional part-of-speech feature can be injected into the model. POS tags are generated with spaCy (`en_core_web_sm`) at the word level before tokenization. A small embedding layer (`nn.Embedding(19, 64)`) maps each UPOS tag to a 64-dimensional vector, which is then projected to 768 dimensions via a linear layer and **added to DistilBERT's last hidden state after the transformer**.

Post-transformer injection was chosen deliberately: adding POS information to the input embeddings would disturb DistilBERT's pretrained representations, which were never trained with POS tokens in the input. Adding it after leaves the pretrained weights fully intact and makes the POS contribution independently ablatable. POS tags follow the same word-level alignment as labels — one tag per word, applied at the first subword position only.

### Label masking

Not all positions carry supervision for all heads:

- **Boundary:** all positions supervised (0 = no boundary, 1 = boundary)
- **Intonation:** positions with `none` or `unclear` labels are masked to -100; only rising (1), falling (2), and level (3) labels are trained on
- **Break index:** empty string labels (present in SBCSAE, where Wav2ToBI was not applied) are masked to -100; only `"3"` and `"4"` receive real supervision. Empty strings represent absence of annotation, not an affirmative "no break" — treating them as class 0 would fabricate supervision.

### Loss

Two loss strategies were explored:

- **Standard loss (stl):** unweighted cross-entropy across all three heads
- **Weighted loss:** upweights the minority boundary class to address class imbalance (~15% of tokens are boundaries); controlled by the `boundary_loss_weight` hyperparameter

Weighted loss produces a different precision/recall operating point (higher recall, lower precision) rather than uniformly better results. The effect is non-monotonic: boundary F1 peaks at BLW=2.0 on the full corpus, while break index F1 peaks at BLW=5.0.

---

## Data and Annotation Pipeline

Four corpora are used, combining silver-standard automatic annotation with gold-standard human annotation.

### LibriTTS (silver standard)

LibriTTS is a large collection of audiobook recordings — read speech from thousands of speakers across clean studio conditions. Transcripts are exact. The silver-standard annotation pipeline processes LibriTTS using two independent automatic annotators and retains only positions where both agree.

Approximately 2.28 million training words across clean-100 and clean-360 subsets. Used for training only; not part of the held-out test set. Train/validation split is 89/11 word-proportional within each subset (seed 42).

LibriTTS provides Wav2ToBI break index labels (`"3"` / `"4"`) for boundary positions via the silver-standard pipeline, making it a primary supervision source for the break index head.

### People's Speech (silver standard)

The People's Speech Corpus is a large-scale dataset spanning a wide range of domains — interviews, speeches, podcasts, YouTube videos, and other Internet Archive sources — mixing single-speaker and multi-speaker data. The clean, CC-BY-licensed subset (MLCommons) is used here.

Approximately 4.37 million training words. Train/validation split is 89/11 word-proportional (seed 42). People's Speech contributes register diversity that LibriTTS alone does not provide, and is the largest single training data source in the pipeline.

### SBCSAE — Santa Barbara Corpus of Spoken American English (gold standard)

SBCSAE contains 60 recordings of naturally occurring spoken American English across a wide range of registers and participant configurations. Labels are derived directly from the Du Bois transcript markup — IU line boundaries become boundary labels; terminal contour markers (`.` falling, `,` level, `?` rising) become intonation labels. No automatic annotation tools are applied.

Approximately 383,000 windowed training words (30-IU sliding windows, stride 15; unique word count ~217,600). **SBC001–005 are held out as the fixed test set** for all runs that include SBCSAE, providing the only apples-to-apples comparison with the GPT-Neo baseline.

Note: SBCSAE transcripts do not include ToBI break indices. All break index positions for this corpus are masked to -100 and receive no supervision.

### BU Radio News Corpus (gold standard, evaluation only)

The Boston University Radio News Corpus contains ~426 annotated files of professionally read broadcast news speech. BU provides explicit human-annotated ToBI labels including break indices, encoded in `.brk` files, making it the gold-standard evaluation set for the break index head.

The BU annotation pipeline (`annotation_pipeline_bu.ipynb`) parses `.ala` (word alignment), `.brk` (break index), and `.ton` (boundary tone) files. Key decisions:

- One utterance = one sample (no windowing)
- `.ton`→`.brk` alignment uses a 200ms tolerance window
- Only `>word` lines from `.ala` are used
- Boundary tone label mapping: `H%`→1 (rising), `L%`/`!H%`→2 (falling), `%`→3 (level), no match→0

**Note on deployment:** BU is used for evaluation only and is excluded from the deployed `text2tobi` model weights due to licensing constraints.

### Annotation pipeline (silver standard)

Since LibriTTS and People's Speech have no human prosody annotations, two independent automatic tools label the audio. A consensus filter then retains only positions where both agree.

**Annotator 1 — PSST**
Built on Whisper, fine-tuned to detect Intonation Unit boundaries. PSST inserts boundary markers directly into its transcription output, already word-aligned. Trained on SBCSAE; F1 = 0.87.

PSST never marks utterance-final words as boundaries — a known behaviour arising from its training data. Because utterance-final positions are near-certain prosodic boundaries, a forced consensus fix is applied: any utterance-final word missing a PSST boundary label is assigned one unconditionally, regardless of Wav2ToBI agreement. This is noted explicitly in the annotation metadata.

**Annotator 2 — Wav2ToBI**
Built on Wav2Vec2 with a bidirectional LSTM and raw F0 appended as an additional feature. Wav2ToBI outputs timestamps in seconds rather than word identities, so a CTC forced alignment model (Wav2Vec2 fine-tuned for ASR) is used to compute word-level timestamps. Trained on BU Radio News; F1 = 0.86.

**Consensus filter**
Only positions where both annotators agree (within a one-word tolerance window) are retained as boundary labels. A stricter metric restricted to positions where at least one system detected a boundary yields 74% symmetric inter-annotator agreement on LibriTTS and 68% on People's Speech. Under the TSP-style metric (which includes non-boundary positions in the denominator), agreement is 94% and 91% respectively.

---

## Results

All test F1 figures reported here are evaluated on **SBC001–005** for boundary and intonation, and on the **BU Radio News Corpus** for break index, unless otherwise noted.

### Corpus combination sweep

All runs: punctuation stripped, no POS injection, best BLW per combination.

| Model | Corpora | BLW | B-F1 | B-Prec | B-Rec | I-F1 | X-F1 |
|-------|---------|-----|------|--------|-------|------|------|
| GPT-Neo 1.2B (baseline) | SBCSAE | — | 0.770 | — | — | — | — |
| #36 | LibriTTS + People's Speech + SBCSAE | 2.0 | **0.835** | 0.816 | 0.856 | 0.577 | 0.602 |
| #44 | People's Speech + SBCSAE | 2.0 | 0.833 | 0.830 | 0.835 | 0.537 | 0.590 |
| #24 | LibriTTS + SBCSAE | 2.0 | 0.824 | 0.787 | 0.865 | 0.557 | 0.604 |
| #40 | SBCSAE only | 2.0 | 0.823 | 0.796 | 0.852 | 0.554 | 0.375† |
| #47 | LibriTTS + People's Speech | 5.0 | 0.798 | 0.758 | 0.843 | 0.219 | 0.623 |
| #51 | People's Speech only | 7.0 | 0.794 | 0.751 | 0.842 | 0.221 | 0.560 |
| #45 | LibriTTS only | 7.0 | 0.718 | 0.689 | 0.749 | 0.223 | 0.612 |

†Break index head untrained (SBCSAE contains no break index labels).

B-F1 = boundary F1 (SBCSAE test); I-F1 = intonation F1 (SBCSAE test); X-F1 = break index F1 (BU Radio News).

### BLW sweep on full corpus

Full corpus (LibriTTS + People's Speech + SBCSAE), no POS, no punctuation.

| Model | BLW | B-F1 | B-Prec | B-Rec | I-F1 | X-F1 |
|-------|-----|------|--------|-------|------|------|
| #1 | stl | 0.824 | 0.863 | 0.789 | 0.559 | 0.582 |
| #38 | 1.5 | 0.833 | 0.834 | 0.832 | 0.563 | 0.576 |
| #36 | **2.0** | **0.835** | 0.816 | 0.856 | 0.577 | 0.602 |
| #26 | 3.0 | 0.824 | 0.766 | 0.891 | 0.556 | 0.559 |
| #9 | 5.0 | 0.812 | 0.729 | 0.915 | 0.537 | **0.647** |

Precision decreases and recall increases monotonically as BLW rises. Boundary F1 peaks at BLW=2.0; break index F1 peaks at BLW=5.0.

### POS ablation

A POS-only run — where the model receives a stream of part-of-speech tags with no lexical input whatsoever — achieves a boundary F1 of **0.70**, approaching the GPT-Neo baseline of 0.77. This is a positive scientific finding: syntactic structure alone, encoded as a continuous POS sequence with no access to words, is a genuine predictor of prosodic phrasing. This result provides direct empirical support for the syntax-phonology interface literature.

However, adding POS injection as a supplement to the full text-only model produced a near-null result: text+POS configurations did not consistently improve over text-only across any metric. DistilBERT's pretraining appears to have already encoded syntactic category information implicitly through lexical context, rendering the explicit POS signal redundant.

### Training dynamics

Across all multi-epoch runs, a consistent pattern emerged:

- **Boundary F1** on the validation set typically peaks at **epoch 2–3** and plateaus or slightly degrades thereafter
- **Intonation F1** on the validation set continues improving through **epoch 5–6**

The divergence between boundary and intonation head dynamics reflects the relative difficulty of the two tasks. Boundary detection converges quickly on syntactic cues; intonation classification requires more exposure to learn subtler distributional patterns. Checkpoint selection should treat the two heads independently.

---

## Evaluation Validity

- **Boundary F1 on SBC001–005** is the only metric directly comparable to the GPT-Neo baseline. All headline comparisons use this figure.
- **Break index F1** is evaluated against BU Radio News gold `.brk` annotations. SBCSAE contains no break index labels.
- **Intonation F1** stands alone — no prior published text-only baseline exists for this task.
- The original best result of 0.8408 was inflated by a speaker-change token (`/`) data leakage mechanism. The corrected headline is 0.8352.

---

## Repository Structure

```
text2tobi-poc/
├── cache/                                   # empty on GitHub; populated locally during annotation runs
├── code/
│   ├── annotations/
│   │   ├── annotation_pipeline_libritts_silver.ipynb
│   │   ├── annotation_pipeline_peoples_speech_silver.ipynb
│   │   ├── annotation_pipeline_sbcsae.ipynb
│   │   └── annotation_pipeline_bu.ipynb
│   ├── model/
│   │   ├── distilBERT_multitrain.ipynb   # main training notebook
│   │   └── run_summary.ipynb             # iterates models/ and creates report
│   └── other/                            # misc code (tons actually, most excluded)
│       └── ...
├── labels/
│   ├── bu/                                  # BU Radio News gold labels
│   ├── clean-100/                           # LibriTTS clean-100 silver labels
│   ├── clean-360/                           # LibriTTS clean-360 silver labels
│   ├── ps/                                  # People's Speech silver labels
│   └── sbcsae/                              # SBCSAE gold labels
├── models/
│   ├── full/                                # one folder per completed run
│   │   └── {run_id}/
│   │       ├── {run_id}_hparams.json
│   │       ├── {run_id}_curves.png
│   │       └── {run_id}_confusion_matrix.png
│   │
│   ├── partial/                             # interrupted or exploratory runs
│   │   └── {run_id}/
│   ├── model_registry.json
│   └── runs_summary.xlsx
└── README.md
```

**Model weights** are not included in this repository due to file size. The best-performing model (Model #36) is publicly available on HuggingFace under the Apache 2.0 license: [`lemmatix/text2tobi`](https://huggingface.co/lemmatix/text2tobi).

### Environment

All notebooks are designed for **Google Colab with a T4 GPU**. Data and label files live on Google Drive and are mounted at the start of each session. No local GPU is required.

Dependencies are installed inline by each notebook's setup cell:

```
transformers datasets scikit-learn matplotlib spacy evaluate
```

### Wav2ToBI patches (required every session)

Wav2ToBI (`ReginaZ/Wav2ToBI-PB-Fuzzy`) requires two manual patches after installation. These must be applied each Colab session before running any annotation notebook that calls Wav2ToBI. The patches are **not** applied automatically — the installed package files must be edited directly.

**Patch 1 — `model.py`: fix LSTM hidden size**

Locate the Wav2ToBI `model.py` file (typically at `/usr/local/lib/python3.x/dist-packages/wav2tobi/model.py` or equivalent) and change:

```python
# Before
lstm_hidden_size = 512

# After
lstm_hidden_size = 256
```

This is required for the PB checkpoint specifically. Without it the model will fail to load with a size mismatch error.

**Patch 2 — `train.py`: fix deprecated evaluate import**

Locate `train.py` in the same package directory and change:

```python
# Before
from datasets import load_metric

# After
from evaluate import load as load_metric
```

The `load_metric` function was moved from `datasets` to the `evaluate` package in a breaking API change. Without this patch, importing Wav2ToBI raises an `ImportError`.

Both patches survive for the duration of the Colab session but are lost on runtime restart.

### Running the pipeline

1. Apply the Wav2ToBI patches above.
2. Run the relevant annotation notebook(s) for your target corpus. Label files are written to Google Drive under `labels/{corpus}/`.
3. Open `distilBERT_multitrain_v2.ipynb`. Edit the configuration cell — set corpus paths, loss strategy, BLW, and run notes.
4. Run all cells top to bottom.
5. Results are saved to `models/{run_id}/` on Drive, including `hparams.json`, `curves.png`, and `confusion_matrix.png`.

---

## Limitations

- **Domain coverage** is limited to audiobook (LibriTTS), conversational (SBCSAE), broadcast news (BU), and mixed Internet audio (People's Speech). Generalization to other registers — telephony, noisy speech, stylized registers — is not tested.
- **Silver-standard noise:** automatic labels retain some annotation error despite the consensus filter. PSST and Wav2ToBI disagree on at least 26% of positions in LibriTTS and 32% in People's Speech where at least one model detects a boundary. Consensus filtering discards these positions entirely, which may bias the training distribution toward high-confidence boundaries.
- **POS contribution** is unobservable: the POS embedding is never supervised independently, making it difficult to determine whether it contributes signal or is simply ignored. The near-null text+POS result suggests DistilBERT's pretraining already encodes syntactic structure implicitly.

---

## Future Directions

- **Checkpoint selection per head:** per-epoch analysis consistently shows boundary and intonation heads peaking at different epochs. Selecting checkpoints independently per head would likely yield gains without additional training.
- **Extended intonation head:** intonation direction is currently predicted only at boundary positions. A full SSML-compatible transcription would require pitch accent and word-level prominence as additional heads.
- **Targeted corpus expansion:** People's Speech domain composition is heterogeneous and undocumented. Higher-quality or domain-filtered subsets, or additional conversational corpora matching the SBCSAE register, may further close the SBC/non-SBC gap.
- **Python package deployment:** model weights are available on HuggingFace; packaging as a pip-installable library would lower the barrier to adoption for TTS front-ends and corpus annotation pipelines.

---

## Citation and Licensing

This repository is released under Apache 2.0. Model weights are separately licensed under Apache 2.0 at [`lemmatix/text2tobi`](https://huggingface.co/lemmatix/text2tobi).

Code generation assistance provided by Claude Sonnet 4.6 (Anthropic, 2026). Prompts, design decisions, and verification by the author.

**Models used:**
- `distilbert-base-uncased` — Sanh et al. (2019)
- `NathanRoll/psst-medium-en` — Roll et al. (2023)
- `ReginaZ/Wav2ToBI-PB-Fuzzy` — Zhai & Hasegawa-Johnson (2023)
- `en_core_web_sm` — spaCy
