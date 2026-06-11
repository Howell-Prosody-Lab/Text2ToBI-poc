# text2tobi-poc

A research prototype for predicting ToBI prosodic structure from text alone — no audio required at inference time. This repository contains the full annotation pipeline, training notebooks, and experimental results for the capstone project underlying the [`text2tobi`](https://github.com/your-handle/text2tobi) package.

**Best result:** boundary F1 = **0.8419** (`sbc_stl`) and **0.8408** (`libri+sbc_pos_stl`) on the SBC001–005 gold held-out test set, against a GPT-Neo text-only baseline of 0.77.

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
- **Break index:** empty string labels (present in LibriTTS silver data and SBCSAE, where Wav2ToBI was not applied) are masked to -100; only `"3"` and `"4"` receive real supervision. Empty strings represent absence of annotation, not an affirmative "no break" — treating them as class 0 would fabricate supervision.

### Loss

Two loss strategies were explored:

- **Standard loss (stl):** unweighted cross-entropy across all three heads
- **Weighted loss:** upweights the minority boundary class to address class imbalance (~15% of tokens are boundaries)

Weighted loss produces a different precision/recall operating point (higher recall, lower precision) rather than uniformly better results. Both are reported.

---

## Data and Annotation Pipeline

Three corpora are used, combining silver-standard automatic annotation with gold-standard human annotation.

### LibriTTS (silver standard)

LibriTTS is a large collection of audiobook recordings — read speech from thousands of speakers across clean studio conditions. Transcripts are exact. The silver-standard annotation pipeline processes LibriTTS using two independent automatic annotators and retains only positions where both agree.

**~145,000 samples** (utterance-level). Used for training only; not part of the held-out test set. Random 80/10/10 train/val/test split when used without SBCSAE.

LibriTTS provides Wav2ToBI break index labels (`"3"` / `"4"`) for boundary positions, making it the primary supervision source for the break index head.

### SBCSAE — Santa Barbara Corpus of Spoken American English (gold standard)

SBCSAE contains 60 recordings of naturally occurring spoken American English across a wide range of registers and participant configurations. Labels are derived directly from the Du Bois transcript markup — IU line boundaries become boundary labels; terminal contour markers (`.` falling, `,` level, `?` rising) become intonation labels. No automatic annotation tools are applied.

**~4,140 samples** (30-IU sliding windows, 50% overlap). **SBC001–005 are held out as the fixed test set** for all runs that include SBCSAE, providing the only apples-to-apples comparison with the GPT-Neo baseline.

Note: SBCSAE transcripts do not include ToBI break indices. The `x` field is null for all SBCSAE samples and is substituted with empty strings downstream, masking all break index positions for this corpus.

### BU Radio News Corpus (gold standard)

The Boston University Radio News Corpus contains ~426 annotated files of professionally read broadcast news speech. Unlike LibriTTS and SBCSAE, BU provides explicit human-annotated ToBI labels including break indices, making it the most direct ToBI supervision source in the pipeline.

The BU annotation pipeline (`annotation_pipeline_bu.ipynb`) parses `.ala` (word alignment), `.brk` (break index), and `.ton` (boundary tone) files. Key decisions:

- One utterance = one sample (no windowing; BU files are discrete annotated utterances, not continuous conversation)
- `.ton`→`.brk` alignment uses a 200ms tolerance window
- Only `>word` lines from `.ala` are used
- Boundary tone label mapping: `H%`→1 (rising), `L%`/`!H%`→2 (falling), `%`→3 (level), no match→0

**Note on deployment:** BU is used for training and evaluation only and is excluded from the deployed `text2tobi` package model due to fair use restrictions.

### Annotation pipeline (LibriTTS silver standard)

Since LibriTTS has no human prosody annotations, two independent automatic tools label the audio. A consensus filter then retains only positions where both agree.

**Annotator 1 — PSST**
Built on Whisper, fine-tuned to detect Intonation Unit boundaries. PSST inserts boundary markers directly into its transcription output (`!!!!!`), already word-aligned. Trained on SBCSAE; F1 = 0.87.

PSST never marks utterance-final words as boundaries — a known behaviour arising from its training data. Because utterance-final positions are near-certain prosodic boundaries, a forced consensus fix is applied: any utterance-final word missing a PSST boundary label is assigned one unconditionally, regardless of Wav2ToBI agreement. This is noted explicitly in the annotation metadata.

**Annotator 2 — Wav2ToBI**
Built on Wav2Vec2 with a bidirectional LSTM and raw F0 appended as an additional feature. Wav2ToBI outputs timestamps in seconds rather than word identities, so a CTC forced alignment model (Wav2Vec2 fine-tuned for ASR) is used to compute word-level timestamps and bridge the gap: if Wav2ToBI places a boundary at 0.54s and forced alignment places `moon` ending at 0.52s, the boundary is assigned to `moon`. Trained on BU Radio News; F1 = 0.86.

**Consensus filter**
Only positions where both annotators agree (within a one-word tolerance window) are retained as boundary labels. The cross-validation of PSST against Wav2ToBI has not been published previously; the agreement rate is itself a reportable finding. **87.3% of utterance-final words received Wav2ToBI corroboration within ±1 word**, validating the silver-standard methodology.

---

## Results

All test F1 figures reported here are evaluated on **SBC001–005** unless otherwise noted. This is the only test configuration directly comparable to the GPT-Neo text-only baseline of F1 = 0.77 (reported in the PSST paper).

LibriTTS random-split results (F1 ~0.87–0.89 on boundary) are not reported as baseline comparisons — the test set overlaps with the training distribution and cannot be meaningfully benchmarked against GPT-Neo.

### Boundary detection

| Run | Corpus | Loss | POS | Boundary F1 | Precision | Recall |
|-----|--------|------|-----|-------------|-----------|--------|
| GPT-Neo (baseline) | SBCSAE | — | — | 0.77 | — | — |
| `sbc_stl` | SBCSAE | standard | — | **0.8419** | 0.8827 | 0.8048 |
| `libri+sbc_pos_stl` | LibriTTS + SBCSAE | standard | ✓ | **0.8408** | 0.8883 | 0.7982 |
| `libri+sbc_stl` | LibriTTS + SBCSAE | standard | — | 0.8225 | 0.8467 | 0.7996 |
| `libri+sbc_weighted` | LibriTTS + SBCSAE | weighted | — | 0.8149 | 0.7479 | 0.8951 |

Both headline configurations beat the baseline. `sbc_stl` trains on SBCSAE alone and achieves the highest boundary F1. `libri+sbc_pos_stl` adds LibriTTS and POS injection; the marginal drop in boundary F1 is accompanied by improved intonation performance, reflecting the benefit of additional data for the harder task.

Weighted loss consistently shifts the precision/recall tradeoff toward higher recall at the cost of precision — a different operating point rather than a weaker model. It may be preferable in applications where missed boundaries are more costly than false positives.

### Intonation type

Intonation F1 is macro-averaged across three classes (rising, falling, level) and evaluated on SBC001–005. There is no prior published text-only baseline for this task.

| Run | Intonation F1 |
|-----|---------------|
| `sbc_stl` | 0.5942 |
| `libri+sbc_pos_stl` | 0.5847 |
| `libri+sbc_weighted` | 0.5708 |

Intonation is a substantially harder task than boundary detection and these figures reflect that. The three-way distinction between rising, falling, and level pitch movements at boundaries carries less syntactic signal than boundary placement itself.

### Break index

Break index F1 is evaluated against LibriTTS silver labels in the validation split. **The SBC001–005 test set contains no break index labels** (Du Bois transcripts do not include ToBI break indices; all positions are masked). A rigorous gold-standard evaluation of break index using BU Radio News corpus annotations is planned before final submission — see Future Directions.

| Run | Break Index F1 |
|-----|----------------|
| `sbc_bu_stl` | 0.7125* |
| `libri+sbc_pos_stl` | — |

*Evaluated against LibriTTS silver val split, not gold test set. Treat with caution.

### POS ablation

A POS-only run (`libri+sbc_posonly_stl`) — where DistilBERT receives POS abbreviation strings as input text rather than actual words — collapsed to near-majority-class performance (boundary F1 ≈ 0.04). This is a positive scientific finding: POS tags alone carry insufficient information to predict prosodic boundaries, confirming that lexical and contextual information encoded in the actual word sequence is essential. POS features are useful as a supplementary signal (as in `libri+sbc_pos_stl`) but not as a standalone input.

### Training dynamics

Across all multi-epoch runs, a consistent pattern emerged:

- **Boundary F1** on the validation set typically peaks at **epoch 2–3** and plateaus or slightly degrades thereafter
- **Intonation F1** on the validation set continues improving through **epoch 5–6**
- Both metrics continue rising on the **training set** indefinitely, indicating overfitting rather than genuine generalization

The divergence between boundary and intonation head dynamics reflects the relative difficulty of the two tasks. Boundary detection converges quickly on the available signal; intonation classification requires more exposure to learn the subtler distributional patterns. The val F1 / val loss divergence observed in some runs is a calibration artifact — val loss continues falling while val F1 has plateaued — and does not indicate generalization failure.

Overfitting is the primary motivation for expanding the training corpus to include BU Radio News and People's Speech.

### Performance trend across corpora

A slight downward trend in boundary F1 is observable as additional corpora are added (LibriTTS alone: ~0.88; LibriTTS + SBCSAE: ~0.82–0.84; LibriTTS + SBCSAE + BU: ~0.84). Two hypotheses for this pattern are under investigation: increased lexical diversity across domains, and domain mismatch between broadcast news, audiobook, and conversational speech. Results with People's Speech are pending.

![Training curves — sbc_stl](results/sbc_stl_curves.png)
![Training curves — libri+sbc_pos_stl](results/libri+sbc_pos_stl_curves.png)

---

## Evaluation Validity

The following distinctions are important for interpreting reported numbers:

- **Boundary F1 on SBC001–005** is the only metric directly comparable to the GPT-Neo baseline. All headline comparisons use this figure.
- **LibriTTS random-split F1** (~0.87–0.89) reflects performance on held-out audiobook data and cannot be compared to the GPT-Neo baseline, which was evaluated on conversational speech.
- **Break index F1** is currently evaluated against LibriTTS silver labels in the val split. A gold-standard evaluation against BU annotations is in progress.
- **Intonation F1** stands alone — no prior published text-only baseline exists for this task.

---

## Repository Structure

```
text2tobi-poc/
├── code/
│   ├── annotations/
│   │   ├── annotation_pipeline_libritts_silver.ipynb
│   │   ├── annotation_pipeline_sbcsae.ipynb
│   │   ├── annotation_pipeline_bu.ipynb
│   │   ├── annotation_pipeline_peoples_speech_silver.ipynb
│   │   ├── parameter_tuning_w2t.ipynb
│   │   ├── crossref_bu.py
│   │   └── verify_bu_pipeline.py
│   └── model/
│       ├── distilBERT_pos.ipynb          # main training notebook (current)
│       ├── distilBERT_multitrain_v2.ipynb
│       ├── distilBERT.ipynb              # early single-task baseline
│       └── run_summary.ipynb
├── runs/                                 # complete experimental record
│   ├── sbc/
│   ├── libri/
│   ├── libri+sbc/
│   ├── libri+sbc+bu/
│   └── sbc+bu/
│       └── {run_id}/
│           ├── {run_id}_curves.png
│           ├── {run_id}_confusion_matrix.png
│           └── {run_id}_hparams.json
│           # checkpoint/ excluded — see HuggingFace Hub note below
├── results/                              # curated highlights
│   ├── sbc_stl_curves.png
│   ├── sbc_stl_confusion_matrix.png
│   ├── libri+sbc_pos_stl_curves.png
│   ├── libri+sbc_pos_stl_confusion_matrix.png
│   ├── libri+sbc_posonly_stl_curves.png  # POS-only collapse
│   └── runs_summary.xlsx
└── README.md
```

**Model weights** are not included in this repository due to file size. Checkpoints for `sbc_stl` and `libri+sbc_pos_stl` will be uploaded to HuggingFace Hub — link to follow.

### Environment

All notebooks are designed for **Google Colab with a T4 GPU**. Data and label files live on Google Drive and are mounted at the start of each session. No local GPU is required.

Dependencies are installed inline by each notebook's Cell 2:

```
transformers datasets scikit-learn matplotlib spacy
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

Both patches survive for the duration of the Colab session but are lost on runtime restart. If you are running annotation notebooks in sequence within a single session, the patches only need to be applied once.

### Running the pipeline

1. Apply the Wav2ToBI patches above.
2. Run the relevant annotation notebook(s) for your target corpus. Label files are written to Google Drive under `labels/{corpus}/`.
3. Open `distilBERT_pos.ipynb`. Edit **Cell 1** only — set corpus paths, POS flags, loss strategy, and run notes.
4. Run all cells top to bottom. The multi-run harness in Cell 14 accepts a `RUNS` list for batched experiments.
5. Results and checkpoints are saved to `runs/{run_id}/` on Drive, including `hparams.json`, `test_predictions.json`, `curves.png`, and `confusion_matrix.png`.

### Drive folder structure

```
/MyDrive/Capstone/project/
├── labels/
│   ├── clean-100/       # LibriTTS clean-100 silver labels
│   ├── clean-360/       # LibriTTS clean-360 silver labels
│   ├── sbcsae/          # SBCSAE gold labels
│   └── bu/              # BU Radio News gold labels
└── runs/
    └── {run_id}/
        ├── checkpoint/          # HuggingFace model weights + tokenizer
        ├── {run_id}_hparams.json
        ├── {run_id}_test_predictions.json
        ├── {run_id}_curves.png
        └── {run_id}_confusion_matrix.png
```

Label files are excluded from this repository and must be regenerated by running the annotation notebooks. They can also be obtained from HuggingFace Datasets / Zenodo (see Citation and Licensing).

---

## Limitations

- **Break index evaluation** lacks a gold-standard test set. Current figures are against LibriTTS silver labels in the val split. This is being addressed before final submission.
- **Domain coverage** is limited to audiobook (LibriTTS), spontaneous conversational (SBCSAE), and broadcast news (BU) speech. Generalization to other registers is not yet tested.
- **Silver-standard noise:** LibriTTS labels are automatic and retain some annotation error despite the consensus filter. The 87.3% Wav2ToBI corroboration rate is high but not perfect.
- **POS contribution** is not yet fully quantified. The difference between `libri+sbc_stl` and `libri+sbc_pos_stl` is modest and warrants further controlled ablation.
- **BU corpus excluded from deployment** due to fair use constraints. The deployable model in `text2tobi` is trained on LibriTTS and SBCSAE only.

---

## Future Directions

- **Gold-standard break index evaluation:** use BU Radio News `.brk` annotations as a proper val/test set for the break index head, mirroring the SBC001–005 methodology for boundary F1
- **People's Speech expansion:** preliminary results on a partial People's Speech clean subset are in progress; the corpus is large enough to substantially increase training data and may reduce overfitting
- **POS contribution quantification:** controlled ablation isolating the effect of POS injection at different corpus scales
- **Continued BU integration:** full training runs including BU for boundary and intonation, pending fair use resolution for the deployed model
- **Deployment:** a pip-installable CLI and Python API are in development in the [`text2tobi`](https://github.com/your-handle/text2tobi) repository

---

## Citation and Licensing

This repository is released under **CC BY-NC 4.0**. Label files are excluded from the repository; dataset artifacts are available via HuggingFace Datasets and Zenodo.

Code generation assistance provided by Claude Sonnet 3.5/4.6 (Anthropic, 2026). Prompts, design decisions, and verification by the author.

**Models used:**
- `distilbert-base-uncased` — Sanh et al. (2019)
- `NathanRoll/psst-medium-en` — PSST
- `ReginaZ/Wav2ToBI-PB-Fuzzy` — Wav2ToBI
- `en_core_web_sm` — spaCy
