# Accuracy improvement protocol

This document separates **implemented improvements** from **unverified accuracy
claims**. A new architecture is not considered better until it wins on the
untouched `Test/` split after duplicate/leakage review.

## Research basis

The original AquaScan CNN follows Tamut et al. (2025) and remains available as
`--architecture paper_cnn`. The recommended trainer now uses ImageNet-pretrained
MobileNetV2 followed by staged fine-tuning. This direction is supported by:

- Ahmed et al. (2024), *Enhancing Fish Disease Classification in Bangladeshi
  Aquaculture through Transfer Learning and LIME*, compared InceptionV3,
  ResNet-50, DenseNet-121 and EfficientNetB3. Their best **test** accuracy was
  82.51% despite 95.51% validation accuracy, demonstrating both the benefit of
  transfer learning and the danger of reporting validation accuracy alone.
  DOI: [10.1109/ICSES63445.2024.10763350](https://doi.org/10.1109/ICSES63445.2024.10763350)
- Biswas et al. (2024), *Empirical Evaluation of Deep Learning Techniques for
  Fish Disease Detection in Aquaculture Systems*, evaluates transfer-learning
  and fusion approaches on freshwater fish disease data.
  DOI: [10.1109/ACCESS.2024.3504283](https://doi.org/10.1109/ACCESS.2024.3504283)
- Cui et al. (CVPR 2019), *Class-Balanced Loss Based on Effective Number of
  Samples*, motivates the less aggressive effective-number class weights now
  used by default.
  DOI: [10.1109/CVPR.2019.00949](https://doi.org/10.1109/CVPR.2019.00949)
- Guo et al. (ICML 2017), *On Calibration of Modern Neural Networks*, shows
  that accuracy and confidence reliability are different. Evaluation now
  reports expected calibration error (ECE) and multiclass Brier score.
  [Paper](https://proceedings.mlr.press/v70/guo17a.html)

Recent ensemble and ViT papers sometimes report 95%+ accuracy, but those
numbers are not directly transferable to AquaScan. Dataset size, class
definitions, duplicate handling and split strategy differ. AquaScan therefore
uses its own untouched test split as the only deployment comparison.

## What changed

1. **ImageNet transfer learning**
   - MobileNetV2 backbone, frozen for an initial warm-up phase.
   - Final backbone layers then fine-tuned at a 100× lower learning rate.
   - Batch-normalisation layers stay frozen during fine-tuning to avoid
     destabilising pretrained statistics on the small dataset.
   - Global average pooling replaces the parameter-heavy flattened feature map.
   - The model includes its own `[0,1] → [-1,1]` MobileNet conversion, so the
     existing backend preprocessing and API remain compatible.
2. **Training/serving preprocessing parity**
   - Training now calls the same `preprocess_batch` function as evaluation and
     API inference: EXIF transpose, RGB conversion, LANCZOS 150×150 resize and
     `/255` normalization.
   - A deterministic per-class split is saved as `validation_split.json`.
   - Augmentation runs only after this shared preprocessing.
3. **Class imbalance and regularization**
   - Effective-number weighting is the default.
   - Raw inverse-frequency weighting and no weighting remain available for
     controlled ablation tests.
   - AdamW and kernel L2 are no longer applied simultaneously.
4. **Leakage audit**
   - `train.audit_dataset` checks unreadable images, class balance, exact
     duplicates and perceptually similar Train/Test pairs.
5. **More honest evaluation**
   - Top-1 and top-3 accuracy.
   - Macro/weighted precision, recall and F1.
   - Balanced accuracy, Matthews correlation coefficient and Cohen's kappa.
   - Log loss, multiclass Brier score and ECE.

## Reproducible experiment

Run from `backend/` with the virtual environment active:

```powershell
# 1. Audit before training. Manually review cross-split matches.
python -m train.audit_dataset

# 2. Baseline: reproduce the custom CNN.
python -m train.train `
  --architecture paper_cnn `
  --output ../model/paper_cnn_baseline.h5 `
  --outputs-dir outputs/paper_cnn
python -m train.evaluate `
  --model-h5 ../model/paper_cnn_baseline.h5 `
  --outputs-dir outputs/paper_cnn

# 3. Recommended transfer-learning candidate.
python -m train.train `
  --architecture mobilenet_v2 `
  --epochs 50 `
  --warmup-epochs 8 `
  --fine-tune-layers 30 `
  --output ../model/mobilenet_v2_candidate.h5 `
  --outputs-dir outputs/mobilenet_v2
python -m train.evaluate `
  --model-h5 ../model/mobilenet_v2_candidate.h5 `
  --outputs-dir outputs/mobilenet_v2
```

Promote the candidate to `model/model.h5` only if:

- test accuracy and macro-F1 improve;
- no disease class suffers an unacceptable recall regression;
- MCC and balanced accuracy improve (important under imbalance);
- ECE/Brier do not indicate substantially worse confidence reliability;
- repeated runs with at least three fixed seeds show the gain is stable; and
- the fish/non-fish gate is evaluated separately on real phone photos.

## Highest-value future data work

Model changes cannot compensate for missing visual diversity. The next
research cycle should collect expert-verified photographs across fish species,
farms, phones, lighting, water clarity and disease stages. Keep all photos of
the same fish or capture session in one split (grouped splitting), and include
a dedicated non-fish/unknown dataset for evaluating rejection performance.
