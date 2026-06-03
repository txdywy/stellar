"""
Kaggle Playground S6E6 - Predicting Stellar Class
Target: balanced_accuracy >= 0.9672 (top 100)

Key insights:
- spectral_type = binned(r-g), galaxy_population = binned(u-r)
- SDSS17 original dataset can augment training (100K extra samples)
- Metric: balanced_accuracy_score
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# 1. Load Data
# ============================================================
print("Loading data...")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sdss = pd.read_csv("data/star_classification.csv")

ID = 'id'
TARGET = 'class'

# Prepare SDSS17 (original dataset) for augmentation
sdss_common = sdss[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift', 'class']].copy()

# Add derived features to SDSS to match train format
def add_derived_features(df):
    df = df.copy()
    df['spectral_type'] = pd.cut(df['r'] - df['g'], [-np.inf, -1, -0.5, 0, np.inf],
                                  labels=['M', 'G/K', 'A/F', 'O/B']).astype(str)
    df['galaxy_population'] = pd.cut(df['u'] - df['r'], [-np.inf, 2.2, np.inf],
                                      labels=['Blue_Cloud', 'Red_Sequence']).astype(str)
    return df

sdss_common = add_derived_features(sdss_common)

# Augment training data
train_aug = pd.concat([train, sdss_common], ignore_index=True)
print(f"Train original: {len(train)}, SDSS: {len(sdss_common)}, Augmented: {len(train_aug)}")

y_raw = train_aug[TARGET].copy()
le = LabelEncoder()
y = le.fit_transform(y_raw)
classes = le.classes_
print(f"Classes: {classes}")

X = train_aug.drop([TARGET], axis=1)
test_ids = test[ID].values
X_test = test.copy()

# ============================================================
# 2. Feature Engineering
# ============================================================
def feature_engineering(df, is_train=True):
    """Create domain-informed features for stellar classification."""
    df = df.copy()

    # --- Critical color indices (from formulae discovery) ---
    # r-g determines spectral_type, u-r determines galaxy_population
    df['r_minus_g'] = df['r'] - df['g']
    df['u_minus_r'] = df['u'] - df['r']

    # --- All pairwise color indices ---
    bands = ['u', 'g', 'r', 'i', 'z']
    for i in range(len(bands)):
        for j in range(i+1, len(bands)):
            df[f'{bands[i]}_{bands[j]}'] = df[bands[i]] - df[bands[j]]

    # --- Redshift interactions (critical for QSO vs GALAXY) ---
    for band in bands:
        df[f'{band}_div_z'] = df[band] / (df['redshift'] + 1e-6)
        df[f'{band}_mul_z'] = df[band] * df['redshift']

    # --- Log transforms ---
    df['log_redshift'] = np.log1p(np.clip(df['redshift'], 0, None))
    df['log_redshift_neg'] = np.where(df['redshift'] < 0,
                                       -np.log1p(-df['redshift']),
                                       df['log_redshift'])

    # --- Coordinate features ---
    df['alpha_rad'] = np.radians(df['alpha'])
    df['delta_rad'] = np.radians(df['delta'])
    df['alpha_sin'] = np.sin(df['alpha_rad'])
    df['alpha_cos'] = np.cos(df['alpha_rad'])
    df['delta_sin'] = np.sin(df['delta_rad'])
    df['delta_cos'] = np.cos(df['delta_rad'])

    # --- Binned spectral features (redundant but useful for tree models) ---
    df['r_g_bin'] = pd.cut(df['r_minus_g'], [-np.inf, -1, -0.5, 0, np.inf], labels=False)
    df['u_r_bin'] = pd.cut(df['u_minus_r'], [-np.inf, 2.2, np.inf], labels=False)

    # --- Encode categoricals ---
    df['spectral_enc'] = df['spectral_type'].map({'M': 0, 'G/K': 1, 'A/F': 2, 'O/B': 3})
    df['pop_enc'] = df['galaxy_population'].map({'Blue_Cloud': 0, 'Red_Sequence': 1})

    # --- Key interactions ---
    df['spec_x_z'] = df['spectral_enc'] * df['redshift']
    df['pop_x_z'] = df['pop_enc'] * df['redshift']
    df['r_g_x_z'] = df['r_minus_g'] * df['redshift']
    df['u_r_x_z'] = df['u_minus_r'] * df['redshift']

    # --- Magnitude ratios ---
    df['u_div_g'] = df['u'] / (df['g'] + 1e-6)
    df['g_div_r'] = df['g'] / (df['r'] + 1e-6)
    df['r_div_i'] = df['r'] / (df['i'] + 1e-6)
    df['i_div_z'] = df['i'] / (df['z'] + 1e-6)

    # Drop original string categoricals and id
    drop_cols = ['spectral_type', 'galaxy_population', 'alpha_rad', 'delta_rad']
    if ID in df.columns and not is_train:
        drop_cols.append(ID)
    if ID in df.columns and is_train:
        drop_cols.append(ID)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df

print("\nEngineering features...")
X_fe = feature_engineering(X, is_train=True)
X_test_fe = feature_engineering(X_test, is_train=False)

# Add ID back for tracking
print(f"Feature count: {X_fe.shape[1]}")
print(f"Features: {X_fe.columns.tolist()}")

# ============================================================
# 3. Model Training - 3 GBDT models + stacking
# ============================================================
N_FOLDS = 10
SEED = 42

# Remove ID if it leaked into features
if ID in X_fe.columns:
    X_fe = X_fe.drop(columns=[ID])
if ID in X_test_fe.columns:
    X_test_fe = X_test_fe.drop(columns=[ID])

print(f"\nFinal feature shape: X={X_fe.shape}, X_test={X_test_fe.shape}")

def get_catboost():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=5000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5,
        random_seed=SEED,
        loss_function='MultiClass',
        eval_metric='TotalF1',
        verbose=0,
        early_stopping_rounds=300,
    )

def get_xgboost():
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=5000,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        objective='multi:softproba',
        eval_metric='mlogloss',
        early_stopping_rounds=300,
        verbosity=0,
    )

def get_lightgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.02,
        max_depth=8,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=SEED,
        objective='multiclass',
        metric='multi_logloss',
        verbose=-1,
    )

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

model_configs = [
    ("catboost", get_catboost),
    ("xgboost",  get_xgboost),
    ("lightgbm", get_lightgbm),
]

oof_preds = {}
test_preds = {}
oof_scores = {}

for model_name, model_fn in model_configs:
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} | {N_FOLDS}-Fold CV")
    print(f"{'='*60}")

    oof = np.zeros((len(X_fe), len(classes)))
    tst = np.zeros((len(X_test_fe), len(classes)))
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_fe, y), 1):
        # Only evaluate on original training data (not SDSS augmented)
        # SDSS indices are >= len(train)
        orig_mask = val_idx < len(train)

        X_tr, y_tr = X_fe.iloc[tr_idx], y[tr_idx]
        X_val, y_val = X_fe.iloc[val_idx], y[val_idx]

        model = model_fn()

        if model_name == "catboost":
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        elif model_name == "xgboost":
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            import lightgbm
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lightgbm.early_stopping(300, verbose=False)])

        oof[val_idx] = model.predict_proba(X_val)
        tst += model.predict_proba(X_test_fe) / N_FOLDS

        # Score only on original training data
        if orig_mask.sum() > 0:
            fold_score = balanced_accuracy_score(
                y[val_idx][orig_mask],
                np.argmax(oof[val_idx][orig_mask], axis=1)
            )
        else:
            fold_score = balanced_accuracy_score(y[val_idx], np.argmax(oof[val_idx], axis=1))
        fold_scores.append(fold_score)
        print(f"  Fold {fold:2d}: {fold_score:.5f}")

    # Overall score on original training data
    orig_idx = np.arange(len(train))
    overall = balanced_accuracy_score(y[orig_idx], np.argmax(oof[orig_idx], axis=1))
    print(f"\n  >> {model_name.upper()} OOF (orig data): {overall:.5f}")
    print(f"  >> Fold mean: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    oof_preds[model_name] = oof
    test_preds[model_name] = tst
    oof_scores[model_name] = overall

# ============================================================
# 4. Stacking with Logistic Regression
# ============================================================
print(f"\n{'='*60}")
print("STACKING: Logistic Regression meta-learner")
print(f"{'='*60}")

# Stack on original training data only
orig_idx = np.arange(len(train))
stack_oof = np.hstack([oof_preds[m[0]][orig_idx] for m in model_configs])
stack_test = np.hstack([test_preds[m[0]] for m in model_configs])
y_orig = y[orig_idx]

meta_scores = []
meta_oof = np.zeros((len(y_orig), len(classes)))
meta_test = np.zeros((len(test_ids), len(classes)))

skf_meta = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for fold, (tr_idx, val_idx) in enumerate(skf_meta.split(stack_oof, y_orig), 1):
    from sklearn.linear_model import LogisticRegression
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    meta.fit(stack_oof[tr_idx], y_orig[tr_idx])
    meta_oof[val_idx] = meta.predict_proba(stack_oof[val_idx])
    meta_test += meta.predict_proba(stack_test) / N_FOLDS

    fold_score = balanced_accuracy_score(y_orig[val_idx], np.argmax(meta_oof[val_idx], axis=1))
    meta_scores.append(fold_score)

stacking_score = balanced_accuracy_score(y_orig, np.argmax(meta_oof, axis=1))
print(f"  Stacking OOF: {stacking_score:.5f}")

# ============================================================
# 5. Weighted Average
# ============================================================
print(f"\n{'='*60}")
print("BLENDING")
print(f"{'='*60}")

weights = np.array([oof_scores[m[0]] for m in model_configs])
weights = weights / weights.sum()
weighted_oof = sum(oof_preds[m[0]][orig_idx] * w for m, w in zip(model_configs, weights))
weighted_score = balanced_accuracy_score(y_orig, np.argmax(weighted_oof, axis=1))
print(f"  Weighted Avg OOF: {weighted_score:.5f}")

# Simple average
avg_oof = sum(oof_preds[m[0]][orig_idx] for m in model_configs) / len(model_configs)
avg_score = balanced_accuracy_score(y_orig, np.argmax(avg_oof, axis=1))
print(f"  Simple Avg OOF: {avg_score:.5f}")

# ============================================================
# 6. Select Best & Create Submission
# ============================================================
print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
all_scores = {**oof_scores, 'stacking': stacking_score,
              'avg_blend': avg_score, 'wtd_blend': weighted_score}
for name, score in sorted(all_scores.items(), key=lambda x: -x[1]):
    print(f"  {name:12s}: {score:.5f}")

best_name = max(all_scores, key=all_scores.get)
best_score = all_scores[best_name]
print(f"\n  >> Best: {best_name} = {best_score:.5f}")

# Generate submissions
def save_submission(preds, name, score):
    labels = le.inverse_transform(np.argmax(preds, axis=1))
    sub = pd.DataFrame({'id': test_ids, 'class': labels})
    path = f"submissions/sub_{name}_{score:.5f}.csv"
    sub.to_csv(path, index=False)
    print(f"  Saved: {path}")

# Save all submissions
for name in oof_scores:
    save_submission(test_preds[name], name, oof_scores[name])
save_submission(meta_test, 'stacking', stacking_score)

# Save best blend
if best_name == 'wtd_blend':
    best_test = sum(test_preds[m[0]] * w for m, w in zip(model_configs, weights))
elif best_name == 'avg_blend':
    best_test = sum(test_preds[m[0]] for m in model_configs) / len(model_configs)
elif best_name == 'stacking':
    best_test = meta_test
else:
    best_test = test_preds[best_name]
save_submission(best_test, 'best', best_score)

print(f"\nTarget: 0.96720 | Achieved: {best_score:.5f}")
if best_score >= 0.9672:
    print("TARGET MET! Ready to submit.")
else:
    print(f"Gap: {0.9672 - best_score:.5f} - may need further optimization.")
