"""
Kaggle Playground S6E6 - Predicting Stellar Class
Target: balanced_accuracy >= 0.9672 (top 100)

Approach: CatBoost + XGBoost + LightGBM + stacking
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load Data
# ============================================================
print("Loading data...")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sample_sub = pd.read_csv("data/sample_submission.csv")

ID = 'id'
TARGET = 'class'

y_raw = train[TARGET].copy()
le = LabelEncoder()
y = le.fit_transform(y_raw)  # GALAXY=0, QSO=1, STAR=2
classes = le.classes_
print(f"Classes: {classes}")
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

X = train.drop([ID, TARGET], axis=1)
X_test = test.drop([ID], axis=1)
train_ids = train[ID].values
test_ids = test[ID].values

# ============================================================
# 2. Feature Engineering
# ============================================================
def feature_engineering(df):
    """Create domain-informed features for stellar classification."""
    df = df.copy()

    # --- Color indices (fundamental in astronomy) ---
    df['u_g'] = df['u'] - df['g']
    df['g_r'] = df['g'] - df['r']
    df['r_i'] = df['r'] - df['i']
    df['i_z'] = df['i'] - df['z']
    df['u_r'] = df['u'] - df['r']
    df['g_i'] = df['g'] - df['i']
    df['u_z'] = df['u'] - df['z']
    df['g_z'] = df['g'] - df['z']

    # --- Redshift interactions (critical for QSO vs GALAXY) ---
    df['g_div_z'] = df['g'] / (df['redshift'] + 1e-6)
    df['i_div_z'] = df['i'] / (df['redshift'] + 1e-6)
    df['r_div_z'] = df['r'] / (df['redshift'] + 1e-6)
    df['u_div_z'] = df['u'] / (df['redshift'] + 1e-6)

    # --- Log redshift ---
    df['log_redshift'] = np.log1p(np.clip(df['redshift'], 0, None))

    # --- Coordinate features ---
    df['alpha_sin'] = np.sin(np.radians(df['alpha']))
    df['alpha_cos'] = np.cos(np.radians(df['alpha']))
    df['delta_sin'] = np.sin(np.radians(df['delta']))
    df['delta_cos'] = np.cos(np.radians(df['delta']))

    # --- Magnitude combinations ---
    df['u_g_over_redshift'] = df['u_g'] / (np.abs(df['redshift']) + 1e-6)

    # --- Encode categorical features ---
    df['spectral_type_enc'] = df['spectral_type'].astype('category').cat.codes
    df['galaxy_population_enc'] = df['galaxy_population'].astype('category').cat.codes

    # --- Interaction between cat and num ---
    df['spec_x_redshift'] = df['spectral_type_enc'] * df['redshift']
    df['pop_x_redshift'] = df['galaxy_population_enc'] * df['redshift']

    return df

print("\nEngineering features...")
X_fe = feature_engineering(X)
X_test_fe = feature_engineering(X_test)

# Store original categorical columns for CatBoost
cat_features = ['spectral_type', 'galaxy_population']
feature_cols = [c for c in X_fe.columns if c not in cat_features]
print(f"Total features: {len(X_fe.columns)} ({len(feature_cols)} numeric + {len(cat_features)} categorical)")

# ============================================================
# 3. Model Definitions
# ============================================================
N_FOLDS = 10
SEED = 42

def get_catboost():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        random_seed=SEED,
        loss_function='MultiClass',
        eval_metric='TotalF1',
        verbose=0,
        early_stopping_rounds=200,
        cat_features=cat_features,
    )

def get_xgboost():
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        objective='multi:softproba',
        eval_metric='mlogloss',
        early_stopping_rounds=200,
        verbosity=0,
    )

def get_lightgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        objective='multiclass',
        metric='multi_logloss',
        verbose=-1,
    )

# ============================================================
# 4. Cross-Validation Training
# ============================================================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

model_configs = [
    ("catboost", get_catboost, X_fe, X_test_fe, True),   # uses cat features
    ("xgboost",  get_xgboost,  X_fe[feature_cols], X_test_fe[feature_cols], False),
    ("lightgbm", get_lightgbm, X_fe[feature_cols], X_test_fe[feature_cols], False),
]

oof_preds = {}
test_preds = {}
oof_scores = {}

for model_name, model_fn, X_tr_data, X_ts_data, use_cat in model_configs:
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} | {N_FOLDS}-Fold CV")
    print(f"{'='*60}")

    oof = np.zeros((len(X_tr_data), len(classes)))
    tst = np.zeros((len(X_ts_data), len(classes)))
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_data, y), 1):
        if use_cat:
            X_tr = X_tr_data.iloc[tr_idx].copy()
            X_val = X_tr_data.iloc[val_idx].copy()
            X_tst = X_ts_data.copy()
        else:
            X_tr = X_tr_data.iloc[tr_idx]
            X_val = X_tr_data.iloc[val_idx]
            X_tst = X_ts_data

        y_tr, y_val = y[tr_idx], y[val_idx]

        model = model_fn()

        if model_name == "catboost":
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        elif model_name == "xgboost":
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:  # lightgbm
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[__import__('lightgbm').early_stopping(200, verbose=False)])

        oof[val_idx] = model.predict_proba(X_val)
        tst += model.predict_proba(X_tst) / N_FOLDS

        fold_score = balanced_accuracy_score(y_val, np.argmax(oof[val_idx], axis=1))
        fold_scores.append(fold_score)
        print(f"  Fold {fold:2d}: {fold_score:.5f}")

    overall = balanced_accuracy_score(y, np.argmax(oof, axis=1))
    print(f"\n  >> {model_name.upper()} OOF: {overall:.5f}")
    print(f"  >> Fold mean: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    oof_preds[model_name] = oof
    test_preds[model_name] = tst
    oof_scores[model_name] = overall

# ============================================================
# 5. Stacking / Blending
# ============================================================
print(f"\n{'='*60}")
print("STACKING: Logistic Regression meta-learner")
print(f"{'='*60}")

# Stack OOF predictions as features for meta-learner
stack_oof = np.hstack([oof_preds[m[0]] for m in model_configs])
stack_test = np.hstack([test_preds[m[0]] for m in model_configs])

meta_scores = []
meta_oof = np.zeros((len(y), len(classes)))
meta_test = np.zeros((len(test_ids), len(classes)))

for fold, (tr_idx, val_idx) in enumerate(skf.split(stack_oof, y), 1):
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    meta.fit(stack_oof[tr_idx], y[tr_idx])
    meta_oof[val_idx] = meta.predict_proba(stack_oof[val_idx])
    meta_test += meta.predict_proba(stack_test) / N_FOLDS

    fold_score = balanced_accuracy_score(y[val_idx], np.argmax(meta_oof[val_idx], axis=1))
    meta_scores.append(fold_score)

stacking_score = balanced_accuracy_score(y, np.argmax(meta_oof, axis=1))
print(f"  Stacking OOF: {stacking_score:.5f}")
print(f"  Fold mean: {np.mean(meta_scores):.5f} +/- {np.std(meta_scores):.5f}")

# ============================================================
# 6. Weighted Average Blend
# ============================================================
print(f"\n{'='*60}")
print("WEIGHTED AVERAGE BLEND")
print(f"{'='*60}")

# Simple average
avg_oof = sum(oof_preds.values()) / len(oof_preds)
avg_score = balanced_accuracy_score(y, np.argmax(avg_oof, axis=1))
print(f"  Simple Average OOF: {avg_score:.5f}")

# Weighted by individual scores
weights = np.array([oof_scores[m[0]] for m in model_configs])
weights = weights / weights.sum()
weighted_oof = sum(oof_preds[m[0]] * w for m, w in zip(model_configs, weights))
weighted_score = balanced_accuracy_score(y, np.argmax(weighted_oof, axis=1))
print(f"  Weighted Average OOF: {weighted_score:.5f}")

# ============================================================
# 7. Select Best & Create Submission
# ============================================================
print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
for name, score in oof_scores.items():
    print(f"  {name:12s}: {score:.5f}")
print(f"  {'stacking':12s}: {stacking_score:.5f}")
print(f"  {'avg_blend':12s}: {avg_score:.5f}")
print(f"  {'wtd_blend':12s}: {weighted_score:.5f}")

# Choose best strategy
all_scores = {
    **oof_scores,
    'stacking': stacking_score,
    'avg_blend': avg_score,
    'wtd_blend': weighted_score,
}
best_name = max(all_scores, key=all_scores.get)
best_score = all_scores[best_name]
print(f"\n  >> Best: {best_name} = {best_score:.5f}")

# Generate submission for best model
if best_name in oof_preds:
    best_test = test_preds[best_name]
elif best_name == 'stacking':
    best_test = meta_test
elif best_name == 'avg_blend':
    best_test = sum(test_preds.values()) / len(test_preds)
elif best_name == 'wtd_blend':
    best_test = sum(test_preds[m[0]] * w for m, w in zip(model_configs, weights))

best_preds = le.inverse_transform(np.argmax(best_test, axis=1))

# Also generate stacking submission (often better on LB)
stacking_preds = le.inverse_transform(np.argmax(meta_test, axis=1))

# Save both
sub_best = pd.DataFrame({ID: test_ids, TARGET: best_preds})
sub_best.to_csv(f"submissions/submission_{best_name}_{best_score:.5f}.csv", index=False)

sub_stack = pd.DataFrame({ID: test_ids, TARGET: stacking_preds})
sub_stack.to_csv(f"submissions/submission_stacking_{stacking_score:.5f}.csv", index=False)

# Save all individual model submissions for potential voting later
for name in oof_preds:
    preds = le.inverse_transform(np.argmax(test_preds[name], axis=1))
    sub = pd.DataFrame({ID: test_ids, TARGET: preds})
    sub.to_csv(f"submissions/submission_{name}_{oof_scores[name]:.5f}.csv", index=False)

# Save OOF for analysis
oof_df = pd.DataFrame({ID: train_ids, TARGET: y_raw})
for name in oof_preds:
    for i, cls in enumerate(classes):
        oof_df[f'{name}_{cls}'] = oof_preds[name][:, i]
oof_df.to_csv("submissions/oof_predictions.csv", index=False)

print(f"\nSubmissions saved to submissions/")
print(f"\nReady to submit when you give the go-ahead!")
