# cifar100-LT, imbalance 100

## 1. Test accuracy (%), mean ± sample std over seeds (bold = best mean per column)

| Family | Method | n | Overall | Many | Medium | Few |
|---|---|---|---|---|---|---|
| Baseline | ERM | 3 | 38.10 ± 0.36 | **64.92 ± 0.74** | 37.24 ± 0.68 | 7.82 ± 0.28 |
| Logit adj. (loss) | Balanced softmax | 3 | **42.63 ± 0.85** | 60.15 ± 0.36 | **42.56 ± 1.15** | 22.26 ± 1.15 |
| Post-hoc | Post-hoc LA (tau on val) | 3 | 42.16 ± 0.92 | 56.50 ± 3.28 | 41.86 ± 1.68 | **25.78 ± 2.49** |
| Post-hoc | Post-hoc LA (tau=1) | 3 | 41.86 ± 0.70 | 60.70 ± 0.61 | 41.98 ± 1.25 | 19.73 ± 0.98 |
| Decoupled | tau-norm | 3 | 41.57 ± 0.71 | 59.69 ± 2.41 | 40.71 ± 1.97 | 21.42 ± 2.47 |
| Decoupled | cRT | 3 | 40.52 ± 0.85 | 62.52 ± 1.04 | 40.84 ± 0.96 | 14.49 ± 0.68 |
| Cosine head | Cosine + LDAM-DRW | 3 | 42.44 ± 0.97 | 62.10 ± 1.17 | 42.32 ± 1.15 | 19.63 ± 0.64 |

## 2. Paired difference vs ERM, percentage points (same seed)

| Method | n | Δ overall | Δ many | Δ medium | Δ few | beats ERM (overall) |
|---|---|---|---|---|---|---|
| Balanced softmax | 3 | +4.52 ± 0.79 | -4.77 ± 0.70 | +5.32 ± 1.54 | +14.43 ± 1.13 | 3/3 |
| Post-hoc LA (tau on val) | 3 | +4.05 ± 0.57 | -8.43 ± 2.54 | +4.62 ± 1.13 | +17.96 ± 2.37 | 3/3 |
| Post-hoc LA (tau=1) | 3 | +3.75 ± 0.34 | -4.23 ± 0.17 | +4.74 ± 0.57 | +11.91 ± 0.93 | 3/3 |
| tau-norm | 3 | +3.46 ± 0.37 | -5.24 ± 1.67 | +3.48 ± 1.33 | +13.60 ± 2.34 | 3/3 |
| cRT | 3 | +2.42 ± 0.50 | -2.40 ± 0.32 | +3.60 ± 0.32 | +6.67 ± 0.92 | 3/3 |
| Cosine + LDAM-DRW | 3 | +4.34 ± 0.70 | -2.82 ± 0.51 | +5.09 ± 1.29 | +11.81 ± 0.85 | 3/3 |

## 3. Per seed (overall / many / medium / few)

- ERM seed 0: 38.29 / 65.57 / 37.03 / 7.93
- ERM seed 1: 38.33 / 65.09 / 38.00 / 7.50
- ERM seed 2: 37.69 / 64.11 / 36.69 / 8.03
- Balanced softmax seed 0: 43.61 / 60.51 / 43.89 / 23.57
- Balanced softmax seed 1: 42.08 / 59.80 / 41.77 / 21.77
- Balanced softmax seed 2: 42.19 / 60.14 / 42.03 / 21.43
- Post-hoc LA (tau on val) seed 0: 42.72 / 59.29 / 42.31 / 23.87 (tau=1.2)
- Post-hoc LA (tau on val) seed 1: 42.66 / 57.31 / 43.26 / 24.87 (tau=1.4)
- Post-hoc LA (tau on val) seed 2: 41.09 / 52.89 / 40.00 / 28.60 (tau=1.7)
- Post-hoc LA (tau=1) seed 0: 42.26 / 61.14 / 41.71 / 20.87 (tau=1.0)
- Post-hoc LA (tau=1) seed 1: 42.26 / 60.94 / 43.34 / 19.20 (tau=1.0)
- Post-hoc LA (tau=1) seed 2: 41.05 / 60.00 / 40.89 / 19.13 (tau=1.0)
- tau-norm seed 0: 41.76 / 61.74 / 40.77 / 19.60 (tau=0.7)
- tau-norm seed 1: 42.16 / 60.29 / 42.66 / 20.43 (tau=0.8)
- tau-norm seed 2: 40.78 / 57.03 / 38.71 / 24.23 (tau=1.0)
- cRT seed 0: 40.84 / 63.31 / 40.77 / 14.70
- cRT seed 1: 41.17 / 62.91 / 41.83 / 15.03
- cRT seed 2: 39.56 / 61.34 / 39.91 / 13.73
- Cosine + LDAM-DRW seed 0: 43.42 / 63.34 / 43.57 / 20.00
- Cosine + LDAM-DRW seed 1: 42.41 / 61.94 / 42.09 / 20.00
- Cosine + LDAM-DRW seed 2: 41.49 / 61.03 / 41.31 / 18.90

## 4. Tail robustness

| Method | worst-decile class accuracy (%) | classes below 10% accuracy |
|---|---|---|
| ERM | 1.13 ± 0.15 | 24.7 ± 0.6 |
| Balanced softmax | 8.40 ± 0.66 | 5.7 ± 0.6 |
| Post-hoc LA (tau on val) | 9.63 ± 1.20 | 4.7 ± 0.6 |
| Post-hoc LA (tau=1) | 6.53 ± 0.49 | 8.7 ± 1.5 |
| tau-norm | 6.10 ± 0.72 | 9.0 ± 1.0 |
| cRT | 3.60 ± 0.10 | 14.7 ± 0.6 |
| Cosine + LDAM-DRW | 5.60 ± 1.01 | 10.3 ± 2.9 |

## 5. Accuracy by class bin (head -> tail), mean over seeds

| Method | 0-9 | 10-19 | 20-29 | 30-39 | 40-49 | 50-59 | 60-69 | 70-79 | 80-89 | 90-99 |
|---|---|---|---|---|---|---|---|---|---|---|
| ERM | 75.8 | 65.6 | 60.1 | 48.4 | 43.9 | 35.4 | 28.4 | 13.5 | 7.4 | 2.6 |
| Balanced softmax | 67.9 | 59.2 | 58.5 | 48.2 | 46.2 | 42.3 | 37.2 | 26.8 | 25.3 | 14.7 |
| Post-hoc LA (tau on val) | 63.7 | 56.2 | 55.3 | 45.2 | 45.8 | 40.8 | 37.2 | 28.6 | 28.6 | 20.2 |
| Post-hoc LA (tau=1) | 69.6 | 60.6 | 58.2 | 47.4 | 46.9 | 40.6 | 36.0 | 24.3 | 22.2 | 12.7 |
| tau-norm | 69.2 | 60.4 | 56.0 | 46.4 | 44.3 | 40.1 | 35.0 | 26.0 | 22.7 | 15.6 |
| cRT | 71.4 | 62.7 | 59.3 | 48.8 | 46.3 | 39.8 | 33.5 | 21.2 | 15.3 | 6.9 |
| Cosine + LDAM-DRW | 69.5 | 60.3 | 61.0 | 49.8 | 46.8 | 44.0 | 34.1 | 23.8 | 22.7 | 12.4 |

## 6. Gain over ERM by class bin, points

| Method | 0-9 | 10-19 | 20-29 | 30-39 | 40-49 | 50-59 | 60-69 | 70-79 | 80-89 | 90-99 |
|---|---|---|---|---|---|---|---|---|---|---|
| Balanced softmax | -8.0 | -6.3 | -1.6 | -0.2 | +2.3 | +6.9 | +8.8 | +13.3 | +17.9 | +12.1 |
| Post-hoc LA (tau on val) | -12.1 | -9.4 | -4.8 | -3.2 | +1.9 | +5.4 | +8.8 | +15.1 | +21.2 | +17.6 |
| Post-hoc LA (tau=1) | -6.2 | -5.0 | -1.9 | -1.0 | +3.0 | +5.2 | +7.7 | +10.9 | +14.8 | +10.1 |
| tau-norm | -6.7 | -5.2 | -4.1 | -2.0 | +0.4 | +4.8 | +6.7 | +12.5 | +15.3 | +13.0 |
| cRT | -4.5 | -2.9 | -0.8 | +0.4 | +2.4 | +4.4 | +5.1 | +7.7 | +8.0 | +4.3 |
| Cosine + LDAM-DRW | -6.3 | -5.3 | +0.9 | +1.4 | +2.9 | +8.7 | +5.8 | +10.3 | +15.4 | +9.8 |

## 7. Tuned tau (seeds 0, 1, 2)

| Method | tau per seed | mean |
|---|---|---|
| Post-hoc LA (tau on val) | 1.2, 1.4, 1.7 | 1.43 |
| Post-hoc LA (tau=1) | 1.0, 1.0, 1.0 | 1.00 |
| tau-norm | 0.7, 0.8, 1.0 | 0.83 |

## Notes for the report
- Test set is balanced, so overall accuracy equals balanced accuracy (mean per-class accuracy).
- Seeds change only initialisation and batch order. The LT split is fixed, so std does not include split variability.
- cRT, tau-norm and post-hoc LA reuse the ERM backbone of the same seed. Their seeds are not independent of ERM's, so use the paired table (2) for comparisons with ERM.
- With n=3 a sample std is rough. Call differences under about 1 to 2 points noise, and do not claim significance.
- Tuned-tau methods choose tau on a small validation set. CIFAR-100 uses 990 held-out training images (99 classes).
- Never mix CIFAR-10 (split: val) and CIFAR-100 (split: lt_full) rows in one table.