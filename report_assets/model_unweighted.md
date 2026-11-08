| Model                        | Weighting                         |   ROC-AUC |   PR-AUC |   Brier Score | Calibration Error Reduction vs Weighted   |
|:-----------------------------|:----------------------------------|----------:|---------:|--------------:|:------------------------------------------|
| Prevalence Constant Baseline | N/A (predicts constant p=0.0876)  |    0.5    |   0.0876 |       0.07993 | N/A                                       |
| Model B (Subscription Only)  | Weighted (scale_pos_weight=10.42) |    0.8816 |   0.5964 |       0.12291 | Baseline                                  |
| Model B (Subscription Only)  | Unweighted (Natural Prevalence)   |    0.8808 |   0.5971 |       0.05136 | 58.2% error reduction                     |
| Model C (Combined Features)  | Weighted (scale_pos_weight=10.42) |    0.8843 |   0.6125 |       0.11856 | Baseline                                  |
| Model C (Combined Features)  | Unweighted (Natural Prevalence)   |    0.8846 |   0.6139 |       0.05028 | 57.6% error reduction                     |
