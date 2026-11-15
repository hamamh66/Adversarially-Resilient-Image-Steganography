# Cover–Stego Separability as a Predictor of Steganalysis Error

Code for a source-disjoint evaluation of image steganalysis on BOSSbase 1.01, with payload whitening used as a negative and a positive control.

## Questions

1. Do separability measures computed on training covers (Fisher ratio, mean shift, deflection, maximum mean discrepancy, Jensen–Shannon divergence) rank embedding conditions in the same order as the error of trained detectors on unseen covers?
2. How do detection estimates change between a single-cover design (P0), a row-level split (P1), a source-disjoint split (P2) and a shifted cover source (P3)?
3. Whitening a uniform message cannot change the stego distribution. How often does each protocol nevertheless report a whitening effect, and does it detect the real effect that exists for a biased message?
4. How does the uncertainty of an AUC estimate depend on the number of independent test covers?

## Contents

| Path | Description |
|---|---|
| `Cover_Stego_Separability_Pattern_Recognition.ipynb` | Complete pipeline: cover preparation, embedding, features, detectors, protocols, controls, tables and figures |

## Experimental design

- **Covers**: BOSSbase 1.01, 10,000 greyscale images downsampled to 256×256; 4,000 / 1,000 / 5,000 training / validation / test covers, split by source image. Central crops of the test images form a shifted test source.
- **Embedding**: LSB matching with uniform and with activity-weighted positions; HILL and S-UNIWARD simulated at the payload–distortion bound; payloads 0.05–0.40 bpp.
- **Detectors**: logistic regression and random forest on 13 summary features; an ensemble of Fisher discriminants on SPAM-686 features; SRNet.
- **Uncertainty**: percentile bootstrap over test covers, with a cover and its stego image always resampled together.

The embedding simulator is an independent implementation. It is checked at every run by self-tests (simulated change rate, S-UNIWARD cost against a brute-force evaluation of its definition, LSB-matching change rate) and is not bit-exact with the reference implementations of the original authors.

## Running

1. Place the BOSSbase 1.01 `.pgm` files in `DATA_ROOT/BOSSbase_1.01` (the notebook also accepts `BOSSbase_1.01.zip` and otherwise tries to download the archive).
2. Set `RUN_MODE = "quick"` in the first code cell for a short pipeline test on sample images, then `RUN_MODE = "full"`.
3. Run all cells. Every stage is cached; an interrupted run resumes where it stopped, including SRNet training.

Requirements: Python 3.10+, NumPy, SciPy, pandas, scikit-learn, scikit-image, Pillow, PyWavelets, joblib, matplotlib, PyTorch. A GPU is needed for SRNet in full mode.

On a cluster, `CSS_DATA_ROOT`, `CSS_WORK_DIR` and `CSS_OUT_DIR` set the dataset, cache and output locations.

## Outputs

Written to `OUT_DIR`: `Tables/*.tex`, `Figures/*.pdf`, CSV files with all per-condition results, `results.json` and `outputs_summary.txt`.

## Archived version

https://doi.org/10.5281/zenodo.24230323
