# cifar10-LT, imbalance 100

## 1. Test accuracy (%), mean ± sample std over seeds (bold = best mean per column)

| Family | Method | n | Overall | Many | Medium | Few |
|---|---|---|---|---|---|---|
| Baseline | ERM | 3 | 69.39 ± 0.74 | 92.59 ± 0.62 | 71.23 ± 0.94 | 43.74 ± 2.21 |
| Reweighting | Reweighted CE | 3 | 69.39 ± 1.48 | 86.41 ± 1.02 | 70.71 ± 1.38 | 50.62 ± 2.84 |
| Reweighting | Focal | 3 | 68.82 ± 0.63 | 92.42 ± 1.08 | 70.55 ± 1.24 | 42.91 ± 1.49 |
| Reweighting | Class-balanced | 3 | 69.27 ± 1.88 | 87.16 ± 1.29 | 69.74 ± 2.29 | 50.76 ± 2.26 |
| Logit adj. (loss) | Balanced softmax | 3 | 77.97 ± 0.85 | 90.43 ± 0.20 | 75.77 ± 1.07 | 68.43 ± 1.36 |
| Post-hoc | Post-hoc LA (tau on val) | 3 | **78.83 ± 0.30** | 83.42 ± 5.13 | 75.88 ± 1.35 | **78.16 ± 5.98** |
| Post-hoc | Post-hoc LA (tau=1) | 3 | 76.43 ± 1.00 | 90.68 ± 0.65 | 75.33 ± 0.73 | 63.64 ± 2.87 |
| Decoupled | tau-norm | 3 | 77.86 ± 0.43 | 84.84 ± 4.04 | **76.06 ± 1.62** | 73.28 ± 4.24 |
| Decoupled | cRT | 3 | 73.95 ± 0.98 | 91.68 ± 0.96 | 74.30 ± 1.24 | 55.74 ± 1.89 |
| Cosine head | Cosine + CE | 3 | 68.79 ± 0.18 | 92.63 ± 0.43 | 70.92 ± 0.12 | 42.11 ± 0.23 |
| Cosine head | Cosine + LDAM | 3 | 72.04 ± 0.59 | **92.92 ± 0.63** | 73.25 ± 0.93 | 49.54 ± 0.62 |
| Cosine head | Cosine + LDAM-DRW | 3 | 76.67 ± 0.76 | 91.43 ± 0.47 | 75.60 ± 0.86 | 63.34 ± 1.85 |

## 2. Paired difference vs ERM, percentage points (same seed)

| Method | n | Δ overall | Δ many | Δ medium | Δ few | beats ERM (overall) |
|---|---|---|---|---|---|---|
| Reweighted CE | 3 | -0.00 ± 1.54 | -6.18 ± 0.70 | -0.52 ± 2.15 | +6.88 ± 3.35 | 1/3 |
| Focal | 3 | -0.57 ± 0.78 | -0.17 ± 1.43 | -0.68 ± 0.60 | -0.83 ± 2.48 | 1/3 |
| Class-balanced | 3 | -0.12 ± 1.81 | -5.43 ± 1.53 | -1.49 ± 1.67 | +7.01 ± 3.92 | 2/3 |
| Balanced softmax | 3 | +8.57 ± 0.25 | -2.16 ± 0.43 | +4.53 ± 1.08 | +24.69 ± 1.41 | 3/3 |
| Post-hoc LA (tau on val) | 3 | +9.43 ± 0.97 | -9.17 ± 4.76 | +4.65 ± 0.43 | +34.41 ± 7.33 | 3/3 |
| Post-hoc LA (tau=1) | 3 | +7.04 ± 0.34 | -1.91 ± 0.04 | +4.10 ± 0.30 | +19.90 ± 0.69 | 3/3 |
| tau-norm | 3 | +8.47 ± 1.02 | -7.74 ± 3.69 | +4.83 ± 0.83 | +29.53 ± 5.56 | 3/3 |
| cRT | 3 | +4.55 ± 0.56 | -0.91 ± 0.35 | +3.07 ± 0.32 | +12.00 ± 1.66 | 3/3 |
| Cosine + CE | 3 | -0.60 ± 0.92 | +0.04 ± 0.89 | -0.32 ± 0.83 | -1.63 ± 2.43 | 1/3 |
| Cosine + LDAM | 3 | +2.65 ± 0.36 | +0.33 ± 1.05 | +2.02 ± 1.07 | +5.80 ± 1.60 | 3/3 |
| Cosine + LDAM-DRW | 3 | +7.28 ± 0.47 | -1.16 ± 0.95 | +4.37 ± 0.84 | +19.60 ± 2.12 | 3/3 |

## 3. Per seed (overall / many / medium / few)

- ERM seed 0: 70.23 / 93.17 / 71.40 / 45.73
- ERM seed 1: 68.82 / 91.93 / 72.08 / 41.37
- ERM seed 2: 69.13 / 92.67 / 70.23 / 44.13
- Reweighted CE seed 0: 69.32 / 86.63 / 71.62 / 48.93
- Reweighted CE seed 1: 67.95 / 85.30 / 69.12 / 49.03
- Reweighted CE seed 2: 70.91 / 87.30 / 71.38 / 53.90
- Focal seed 0: 69.22 / 91.37 / 71.40 / 44.17
- Focal seed 1: 69.15 / 92.37 / 71.12 / 43.30
- Focal seed 2: 68.09 / 93.53 / 69.12 / 41.27
- Class-balanced seed 0: 70.33 / 87.80 / 71.70 / 51.03
- Class-balanced seed 1: 70.38 / 88.00 / 70.30 / 52.87
- Class-balanced seed 2: 67.10 / 85.67 / 67.22 / 48.37
- Balanced softmax seed 0: 78.95 / 90.57 / 76.95 / 70.00
- Balanced softmax seed 1: 77.54 / 90.20 / 75.48 / 67.63
- Balanced softmax seed 2: 77.41 / 90.53 / 74.88 / 67.67
- Post-hoc LA (tau on val) seed 0: 78.65 / 89.17 / 75.95 / 71.73 (tau=1.3)
- Post-hoc LA (tau on val) seed 1: 79.17 / 81.80 / 77.20 / 79.17 (tau=2.2)
- Post-hoc LA (tau on val) seed 2: 78.66 / 79.30 / 74.50 / 83.57 (tau=2.3)
- Post-hoc LA (tau=1) seed 0: 77.47 / 91.30 / 75.67 / 66.03 (tau=1.0)
- Post-hoc LA (tau=1) seed 1: 75.47 / 90.00 / 75.83 / 60.47 (tau=1.0)
- Post-hoc LA (tau=1) seed 2: 76.35 / 90.73 / 74.50 / 64.43 (tau=1.0)
- tau-norm seed 0: 77.71 / 89.33 / 75.63 / 68.87 (tau=1.5)
- tau-norm seed 1: 78.34 / 83.70 / 77.85 / 73.63 (tau=2.5)
- tau-norm seed 2: 77.53 / 81.50 / 74.70 / 77.33 (tau=2.5)
- cRT seed 0: 74.99 / 92.53 / 74.65 / 57.90
- cRT seed 1: 73.80 / 90.63 / 75.33 / 54.93
- cRT seed 2: 73.05 / 91.87 / 72.93 / 54.40
- Cosine + CE seed 0: 68.59 / 92.20 / 70.88 / 41.93
- Cosine + CE seed 1: 68.92 / 92.63 / 71.05 / 42.37
- Cosine + CE seed 2: 68.86 / 93.07 / 70.83 / 42.03
- Cosine + LDAM seed 0: 72.69 / 93.07 / 74.30 / 50.17
- Cosine + LDAM seed 1: 71.88 / 93.47 / 72.90 / 48.93
- Cosine + LDAM seed 2: 71.55 / 92.23 / 72.55 / 49.53
- Cosine + LDAM-DRW seed 0: 77.47 / 90.93 / 76.50 / 65.30
- Cosine + LDAM-DRW seed 1: 76.59 / 91.50 / 75.52 / 63.10
- Cosine + LDAM-DRW seed 2: 75.96 / 91.87 / 74.78 / 61.63

## 4. Tail robustness

| Method | worst-decile class accuracy (%) | classes below 50% accuracy |
|---|---|---|
| ERM | 36.23 ± 3.50 | 2.0 ± 0.0 |
| Reweighted CE | 44.87 ± 4.97 | 0.7 ± 0.6 |
| Focal | 37.60 ± 2.52 | 2.0 ± 0.0 |
| Class-balanced | 44.10 ± 7.40 | 1.0 ± 0.0 |
| Balanced softmax | 66.27 ± 1.69 | 0.0 ± 0.0 |
| Post-hoc LA (tau on val) | 67.63 ± 2.08 | 0.0 ± 0.0 |
| Post-hoc LA (tau=1) | 59.80 ± 3.55 | 0.0 ± 0.0 |
| tau-norm | 66.83 ± 3.43 | 0.0 ± 0.0 |
| cRT | 50.00 ± 4.27 | 0.7 ± 0.6 |
| Cosine + CE | 32.40 ± 1.95 | 2.0 ± 0.0 |
| Cosine + LDAM | 41.13 ± 2.91 | 2.0 ± 0.0 |
| Cosine + LDAM-DRW | 56.20 ± 3.12 | 0.0 ± 0.0 |

## 7. Tuned tau (seeds 0, 1, 2)

| Method | tau per seed | mean |
|---|---|---|
| Post-hoc LA (tau on val) | 1.3, 2.2, 2.3 | 1.93 |
| Post-hoc LA (tau=1) | 1.0, 1.0, 1.0 | 1.00 |
| tau-norm | 1.5, 2.5, 2.5 | 2.17 |

## Notes for the report
- Test set is balanced, so overall accuracy equals balanced accuracy (mean per-class accuracy).
- Seeds change only initialisation and batch order. The LT split is fixed, so std does not include split variability.
- cRT, tau-norm and post-hoc LA reuse the ERM backbone of the same seed. Their seeds are not independent of ERM's, so use the paired table (2) for comparisons with ERM.
- With n=3 a sample std is rough. Call differences under about 1 to 2 points noise, and do not claim significance.
- Tuned-tau methods choose tau on a small validation set. CIFAR-100 uses 990 held-out training images (99 classes).
- Never mix CIFAR-10 (split: val) and CIFAR-100 (split: lt_full) rows in one table.