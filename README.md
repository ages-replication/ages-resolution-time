# sprint-overrun-prediction

Replication package for sprint-overrun classification study (random vs. chronological split, threshold calibration strategy) 
How Evaluation Protocols Shape Reported Performance: An Empirical Study of Machine Learning for Classifying Long-Resolution Software Issues

This repository contains the code, processed data, and results supporting the paper: **How Evaluation Protocols Shape Reported Performance: An Empirical Study of Machine Learning for Classifying Long-Resolution Software Issues**.

Seven ML models (Random Forest, XGBoost, Gradient Boosting, AdaBoost, Decision Tree, Logistic Regression, MLP) are evaluated under three protocols of increasing rigour:

  - Protocol A (permissive): random split, threshold calibrated on the test set
  - Protocol B (intermediate): random split, threshold calibrated via out-of-fold (OOF) predictions on the training set
  - Protocol C (rigorous): chronological split, OOF-calibrated threshold

This study uses the raw AgES file (Raw_AgES_Dataset.zip) from https://github.com/shankasp/AgES, retrieved on 14 April 2026. The file is not redistributed here. To reproduce the results, download and unzip it, and place Raw_AgES_Dataset.csv in the ages-resolution-time folder, next to the pipeline folder. The data preparation notebook (01) rebuilds the processed dataset used by all other notebooks.

**Repository structure**

```
project_directory/
├── Datasets/
│   └── df_final.csv
├── Notebooks/
│   └── ages_common.py
│   └── 01_data_preparation.py
│   └── 02a_random_split_benchmark.py
│   └── 02b_chronological_split_benchmark.py
│   └── 03a_random_optimisation.py
│   └── 03b_chronological_optimisation.py
│   └── 04_protocols_a_b_c.py
│   └── 05_statistical_testing.py
│   └── 06_protocol_comparison_b_vs_c.py
│   └── 07_cross_project_and_within_repository.py
│   └── 08_horizon_sensitivity_7_14_21_28.py
│   └── 09_issue_category.py
│   └── 10_rolling_origin_cut_off.py
│   └── 11_creation_time_features.py
├── Results/
│   └── Tables/
│       └── A_feature_importance.csv
│       └── A_test_results.csv
│       └── B_feature_importance.csv
│       └── B_test_results.csv
│       └── B_vs_C_bootstrap.csv
│       └── C_feature_importance.csv
│       └── C_seen_vs_unseen_repositories.csv
│       └── C_test_results.csv
│       └── category_deltas.csv
│       └── chronological_baseline_full.csv
│       └── contextual_correlations.csv
│       └── creation_time_features.csv
│       └── feature_ranking_all_protocols.csv
│       └── horizon_delta_summary.csv
│       └── horizon_rank_stability.csv
│       └── leave_one_repository_out.csv
│       └── model_rank_stability.csv
│       └── protocol_deltas.csv
│       └── random_baseline_full.csv
│       └── repository_profile.csv
│       └── rolling_origin.csv
│       └── wilcoxon_main.csv
│       └── within_repo_replication.csv
│       └── within_repo_wilcoxon.csv
│   └── Figures/
│       └── A_confusion_matrix.png
│       └── A_feature_importance_top10.png
│       └── B_confusion_matrix.png
│       └── B_feature_importance_top10.png
│       └── C_confusion_matrix.png
│       └── C_feature_importance_top10.png
│       └── category_performance.png
│       └── feature_rank_bumpchart.png
│       └── horizon_sensitivity.png
│       └── roc_all_protocols.png
├── requirements.txt
```

**Reproducing the results**

Notebooks are numbered in the order they should be executed. Each notebook reads only from its designated input folder(s) and writes only to its own output folder, so re-running any single notebook does not affect the outputs of others.

All notebooks were originally run in Google Colab. File paths reference Google Drive by default; adjust the `PROJECT` variable at the top of each notebook if running locally.

All experiments were performed in Google Colab using Python 3.13.18.
See **requirements.txt** for exact package versions.
