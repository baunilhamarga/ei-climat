# Study 5 Presentation Figures

This folder contains polished PDF figures for presenting Study 5: cross-validation with the California Housing dataset and a decision tree regressor.

## Figures

### `01_cross_validation_concept.pdf`
Concept slide explaining why cross-validation is useful. It contrasts a single train/test split with repeated fold-based evaluation and includes the observed 20-fold cross-validation MAE summary.

### `02_california_house_value_map.pdf`
Geographic overview of the California Housing dataset. Each sampled district is plotted by longitude and latitude, colored by median house value, showing the spatial structure in the target variable.

### `03_target_and_income_distributions.pdf`
Side-by-side distributions of the prediction target, median house value, and the strongest linear predictor, median income. Use this to introduce the regression task and the scale of the values.

### `04_feature_correlation_heatmap.pdf`
Correlation matrix for all numerical features. This highlights that `MedInc` has the strongest linear relationship with house price while geographic variables also carry important information.

### `05_decision_tree_overfitting_train_vs_test.pdf`
Train versus test MAE for two train/test splits. This figure demonstrates that an unconstrained decision tree nearly memorizes the training data but has much larger test error.

### `06_single_split_mae_sensitivity.pdf`
Decision tree test MAE across 30 different `random_state` values. This shows why relying on only one train/test split can give a fragile estimate of model performance.

### `07_cross_validation_fold_errors.pdf`
Twenty-fold cross-validation results shown as both fold-by-fold bars and an error distribution. Use this as the main evidence slide for cross-validation.

### `08_error_relative_to_house_prices.pdf`
House price distribution with the cross-validation MAE overlaid. This puts the model error in context by comparing average error with the average house value.

### `09_study5_takeaway_summary.pdf`
Summary slide with the main presentation conclusions: do not evaluate on training data, a single split is sample-dependent, and cross-validation gives a more reliable performance estimate.
