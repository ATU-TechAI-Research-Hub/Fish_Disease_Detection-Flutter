# AquaScan — UEMCON 2026 manuscript notes

Files in this folder:

| File | What it is |
|---|---|
| `aquascan_uemcon2026.tex` | The manuscript. Standard `IEEEtran` conference class. |
| `aquascan_uemcon2026_preview.pdf` | Local preview build (see "Compiling" below). |
| `figures/` | All figures (`.pdf` vectors + the four Android screenshots). |

Three extra figures are built but not currently placed in the manuscript, in
case you want them back: `fig_leakage.pdf` (dataset-duplication diagram — the
same content as Table I), `fig_perclass.pdf` (per-class bars with CI whiskers —
same content as Table IV), and `fig_classes.pdf` (one exemplar image per class).
Each is a drop-in `\includegraphics`.
| `stats.json` | Every derived statistic used in the paper. |
| `aquascan.tex`, `references.bib` | The previous draft, left untouched. |

## Compiling

The `.tex` targets the official `IEEEtran.cls` and cites references with
`\cite{bN}`, with the bibliography generated from `references.bib` via
`\bibliographystyle{IEEEtran}` + `\bibliography{references}` at the bottom of
the file. Upload both `aquascan_uemcon2026.tex` **and** `references.bib`
(plus `figures/`) to Overleaf, or locally:

```
pdflatex aquascan_uemcon2026.tex
bibtex aquascan_uemcon2026
pdflatex aquascan_uemcon2026.tex
pdflatex aquascan_uemcon2026.tex
```

All four passes are needed — `bibtex` reads the `.aux` file's `\citation`
commands and `references.bib`, writes a `.bbl`, and the following two
`pdflatex` passes pull that in and resolve cross-references. Forgetting
`references.bib` when uploading, or skipping the `bibtex` step, is the most
common way this fails to compile cleanly elsewhere.

Author block is active (Rachit Bhusal, Robin Ghosh — Arkansas Tech University),
not commented out for blind review.

`aquascan_uemcon2026_preview.pdf` was built with a local stand-in class and a
stand-in `.bst` (order-of-citation numbering, but author names are not
abbreviated to initials the way real `IEEEtran.bst` does) because neither was
installed in the build sandbox. Layout is close but not identical —
**treat the preview as a proof, not as the camera-ready; the real IEEEtran.bst
on Overleaf will format author initials correctly**. Current build is 10
pages, zero LaTeX errors, zero undefined references/citations.

## Length

The preview runs 11 pages. If the CFP limit is 6–8 pages, cut in this order —
each item is self-contained and removing it breaks nothing else:

1. §III-G *Baseline architectures* and Table VI — move to a sentence in §VI. (~0.7 p)
2. Table IV per-class results — the same information is summarised in the text. (~0.4 p)
3. §V-C paragraph on decoding hyper-parameters and low-memory mode. (~0.3 p)
4. Table VII client components — compress to prose. (~0.4 p)
5. §VI last item *Data collection and deployment* — shorten to three sentences. (~0.4 p)
6. Fig. 2 (CNN layer table) — the text already states the key numbers. (~0.3 p)

That path gets to ~8 pages without touching the four contributions.

## The main finding, and why the headline number changed

I ran the project's own `backend/train/audit_dataset.py` against the dataset.
Report written to `backend/outputs/dataset_audit.json`. Result:

- **All 700 images in `Test/` are byte-identical (SHA-256) copies of images in
  `Train/`.** 698 exact duplicate groups, every one of them cross-split.
- Collapsing duplicates leaves **1,747 distinct photographs** in the 2,450-file
  corpus.
- 208 further cross-split pairs match at difference-hash distance ≤ 4.
- One exact group and five near-duplicate pairs carry **conflicting labels**
  (the same photo filed under both *bacterial gill disease* and *parasitic
  disease*).
- Duplication *within* `Train/` is negligible (3 groups, 6 images), so the
  internal 15% validation split is essentially clean.

This is why the paper reports 81.29% as **in-sample** and 63.71% (internal
validation) as the generalisation estimate. It also explains the 17-point gap
between the two, which would otherwise look like a bug. The framing is neutral:
it is a property of the published dataset, not a mistake by anyone who used it.

## Provenance of every number

| Claim | Source |
|---|---|
| 2,450 files / 1,750 / 700 / imbalance 1.00 / 0 unreadable | `backend/outputs/dataset_audit.json` |
| 698 groups, 700/700, 1,747 distinct, 208 near-dups, 181 images, 1+5 conflicts | same |
| Layer shapes, 2,202,087 / 2,201,639 / 448 params, L2 1e-4 | read back from `model/model.h5` (Keras 3.14 model config) |
| Batch 32, 60 epochs, LR 8e-4, AdamW, label smoothing 0.05, split 0.15 seed 42, class weights, callbacks | `backend/outputs/training_summary.json` |
| Val acc 63.71% @ epoch 51, val loss 1.3010 @ epoch 56, four LR drops | `backend/outputs/training_history.json` |
| 81.29 / 95.14 / macro P,R,F1 / log loss 0.7830 / per-class | `backend/outputs/evaluation_summary.json` |
| Confusion matrix, 131 errors, 40 into parasitic, 24 into healthy | `backend/outputs/confusion_matrix.csv` |
| Balanced acc, MCC 0.7834, κ 0.7817, all bootstrap + Wilson CIs | computed here → `paper/stats.json` (10,000 resamples, seed 20260809) |
| Earlier checkpoint 69.43% / 0.6984 | `backend/outputs/evaluation_summary_v1.json` |
| McNemar χ² ≥ 19.97, p ≤ 7.9e-6 | derived bound (see below) |
| Gate: 16 ImageNet indices, τ_g = 0.05, 224² input, fail-open | `backend/app/core/fish_detector.py` |
| Rejection τ_c = 0.20, τ_H = 1.90, tiers 0.70 / 0.45, top-3 | `backend/app/services/prediction_service.py` |
| 15 MB cap, MIME allowlist, timing header | `backend/app/main.py` |
| Artefact sizes 25.3 MB / 8.4 MB / 1.5 kB / 4.7 kB | file sizes on disk |
| Client behaviour: 15 s poll, 30 s timeout, 200 px / 3:1 / 15 kB checks | `lib/services/*.dart` |
| RAG: MiniLM-L6-v2, 384-d, IndexFlatIP, 36 chunks, 900/140, k = 4, 54 kB index | `AI chatbot/aquaculture_assistant/*`, `vector_db/manifest.json` |
| GGUF models 4.1–4.7 GiB, n_ctx 4096, temp 0.35, top-p 0.90, top-k 40 | `llm/local_llm.py`, `config.py`, file sizes |
| Prof.'s 99.71%, loss 0.0119, ResNet50 70.83% | Tamut et al. 2025 |
| Baseline parameter counts (3.4 / 5.3 / 5.7 / 25.6 M) | published architecture properties, labelled as such in Table VI |

### The McNemar bound

An exact paired test needs the per-image agreement table, and per-image
predictions were not saved for the earlier checkpoint. But the marginals pin it
down anyway: with b − c = 83, b ≤ 214 and c ≤ 131, the discordant total is at
most 345, and χ² = (b−c)²/(b+c) is decreasing in (b+c). So χ² ≥ 83²/345 =
19.97 for *every* table consistent with the data, giving p ≤ 7.9×10⁻⁶. That is
a real result, not an estimate — it just happens to be a bound rather than a
point value.

## What is reported as NOT done (deliberately)

Nothing in these categories is estimated, interpolated, or filled in:

- **Baselines.** ResNet50, EfficientNetB0, MobileNetV2, ViT-Tiny have no
  evaluation artefacts in the project. The MobileNetV2 transfer path exists in
  `train.py` but was never evaluated, so Table VI has empty accuracy columns and
  the row says "implemented, not evaluated".
- **Fish-presence gate.** No labelled fish/non-fish set exists, so there is no
  precision, recall, FPR, or OOD number anywhere in the paper — only the
  protocol.
- **RAG quality.** No retrieval or grounding evaluation has been run, so no
  Recall@k, grounding rate, citation correctness, or hallucination figure
  appears — only the benchmark design.
- **Calibration.** ECE and Brier score need saved probability vectors, which
  the evaluator did not persist. Listed in Future Work.

## Two things to fix in the codebase

1. **The citation in your code is wrong.** `README.md`, `labels.json`,
   `preprocessing.py`, `train.py`, and `main.py` all cite the paper as
   *"Tamut, J., Mangang, Y.A., & Chingakham, C. — Image Classification of
   Freshwater Fish Diseases in South Asian Aquaculture Using Convolutional
   Neural Network."* The actual paper at that DOI is:

   > H. Tamut, R. Ghosh, K. Gosh, M. A. S. Siddique, "Enhancing Disease
   > Detection in the Aquaculture Sector Using Convolutional Neural Networks
   > Analysis," *Aquaculture Journal* 5(1), 6, 2025.

   The manuscript uses the correct form. Worth fixing in the repo too.

2. **Persist per-image predictions.** Adding a `predictions.npz` (probability
   vectors + true labels) to `train/evaluate.py` unlocks exact McNemar tests,
   ECE/Brier, reliability diagrams, confidence-stratified error analysis, and a
   leakage-corrected accuracy on the deduplicated subset — four of the five
   things currently sitting in Future Work.

## Suggested next experiment (highest value per hour)

Rebuild the split from the 1,747 distinct images with group-disjoint
partitioning, retrain the current architecture, and re-evaluate. That single
run converts the paper's central limitation into a result and makes every
baseline comparison meaningful.
