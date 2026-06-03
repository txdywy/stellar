# Stellar Classification - Kaggle Playground S6E6

Kaggle competition: [Predicting Stellar Class](https://www.kaggle.com/competitions/playground-series-s6e6)

## Approach

- **Models**: XGBoost + LightGBM ensemble with diverse hyperparameters
- **Features**: Color indices (u-g, g-r, r-i, i-z, etc.), redshift interactions, categorical encodings
- **CV**: 10-fold stratified cross-validation
- **Ensemble**: Stacking with Logistic Regression meta-learner

## Key Insight

`spectral_type` and `galaxy_population` are deterministic functions of color indices:
- `spectral_type = bin(r - g)`
- `galaxy_population = bin(u - r)`

## Files

- `solution_fast.py` - Main 4-model ensemble (3 XGBoost + 1 LightGBM)
- `solution_final.py` - Full 5-model ensemble with stacking
- `solution.py` / `solution_v2-v4.py` - Iterative improvements

## Score

Local CV accuracy: ~0.9682
