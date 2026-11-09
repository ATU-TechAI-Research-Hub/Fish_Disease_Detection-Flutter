# Interpreting AquaScan fish-disease predictions

AquaScan classifies a photograph into seven trained categories: Bacterial Red
Disease, Bacterial Aeromoniasis, Bacterial Gill Disease, Fungal
Saprolegniasis, Healthy Fish, Parasitic Disease, and Viral White Tail Disease.
It also uses a separate fish-presence gate and may return No Fish Detected.

## What the score means

The displayed percentage is the classifier's softmax probability estimate for
the selected class. It is not the probability that a clinical diagnosis is
correct, and it does not replace microscopy, culture, PCR, necropsy, or
professional examination. Scores may be poorly calibrated on photographs that
differ from training data.

The top alternatives are useful when two diseases have overlapping visual
signs. A small difference between the first and second score indicates
ambiguity even if the first score looks substantial.

## Why a prediction may be wrong

- Several bacterial diseases produce similar redness, ulcers, or fin damage.
- Fungal growth may be secondary to trauma or another infection.
- Water-quality injury can resemble infectious gill or skin disease.
- Lighting, blur, reflections, nets, hands, background, and partial fish views
  can dominate a small 150 by 150 model input.
- The model does not know farm history, water chemistry, internal lesions,
  species susceptibility, or laboratory results.
- A fish species or disease appearance absent from training data is
  out-of-distribution.

## Explaining "why"

Unless a saliency or attribution map was generated for the particular image,
the system cannot truthfully identify the exact pixels that caused the CNN
decision. It can compare the predicted class with documented symptoms and
state which visible signs are commonly associated with that class. That is a
plausibility explanation, not proof of the model's internal reasoning.

## Recommended next steps

Use a sharp, well-lit side view where the fish fills the frame. Take additional
views of skin, fins, tail, and gills when safe. Check dissolved oxygen,
temperature, pH, ammonia, and nitrite. Compare several affected fish, review
recent management changes, and obtain professional or laboratory confirmation
before treatment when losses are important, signs are severe, or confidence is
low.
