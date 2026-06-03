"""
Kaggle Playground S6E6 - v4: Target Encoding + Diverse Ensemble
Key insight: target encoding of feature interactions is crucial
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

print = lambda *a, **k: (sys.stdout.write(' '.join(map(str, a)) + k.get('end', '\n')), sys.stdout.flush())
np.random.seed(42)

# ============================================================
# 1. Load Data
# ============================================================
print("Loading data...")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

ID, TARGET = 'id', 'class'
y_raw = train[TARGET].copy()
le = LabelEncoder()
y = le.fit_transform(y_raw)
classes = le.classes_
print(f"Classes: {classes}, Train: {train.shape}, Test: {test.shape}")

X = train.drop([ID, TARGET], axis=1)
X_test = test.drop([ID], axis=1)
test_ids = test[ID].values

# ============================================================
# 2. Feature Engineering
# ============================================================
def feature_engineering(df):
    df = df.copy()

    # Critical color indices
    df['r-g'] = df['r'] - df['g']  # determines spectral_type
    df['u-r'] = df['u'] - df['r']  # determines galaxy_population

    # All pairwise color indices
    bands = ['u', 'g', 'r', 'i', 'z']
    for i in range(len(bands)):
        for j in range(i+1, len(bands)):
            df[f'{bands[i]}-{bands[j]}'] = df[bands[i]] - df[bands[j]]

    # Redshift features
    df['log_z'] = np.log1p(np.abs(df['redshift'])) * np.sign(df['redshift'])
    for b in bands:
        df[f'{b}/z'] = df[b] / (df['redshift'] + 1e-6)

    # Encode categoricals
    df['spec_enc'] = df['spectral_type'].map({'M': 0, 'G/K': 1, 'A/F': 2, 'O/B': 3})
    df['pop_enc'] = df['galaxy_population'].map({'Blue_Cloud': 0, 'Red_Sequence': 1})

    # Interactions
    df['spec*z'] = df['spec_enc'] * df['redshift']
    df['pop*z'] = df['pop_enc'] * df['redshift']
    df['(r-g)*z'] = df['r-g'] * df['redshift']
    df['(u-r)*z'] = df['u-r'] * df['redshift']

    # Drop string categoricals
    df = df.drop(columns=['spectral_type', 'galaxy_population'])
    return df

print("\nEngineering features...")
X_fe = feature_engineering(X)
X_test_fe = feature_engineering(X_test)
print(f"Features: {X_fe.shape[1]}")

# ============================================================
# 3. Target Encoding with CV (no leakage)
# ============================================================
N_FOLDS = 10
SEED = 42

# Columns to target-encode (interactions of categoricals with key features)
te_candidates = ['spec_enc', 'pop_enc', 'r-g_bin', 'u-r_bin']

# Create binned versions for target encoding
for df in [X_fe, X_test_fe]:
    df['r-g_bin'] = pd.cut(df['r-g'], bins=20, labels=False)
    df['u-r_bin'] = pd.cut(df['u-r'], bins=20, labels=False)
    df['z_bin'] = pd.cut(df['redshift'], bins=20, labels=False)
    df['spec*z_bin'] = (df['spec_enc'] * 20 + df['z_bin']).astype(int)
    df['pop*z_bin'] = (df['pop_enc'] * 20 + df['z_bin']).astype(int)

te_cols = ['spec_enc', 'pop_enc', 'r-g_bin', 'u-r_bin', 'z_bin', 'spec*z_bin', 'pop*z_bin']

# ============================================================
# 4. Train Models with Target Encoding per Fold
# ============================================================
def get_catboost(seed):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=5000, learning_rate=0.02, depth=8,
        l2_leaf_reg=5, random_seed=seed,
        loss_function='MultiClass', eval_metric='MultiClass',
        verbose=0, early_stopping_rounds=300,
    )

def get_xgboost(seed):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=5000, learning_rate=0.02, max_depth=8,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
        random_state=seed, objective='multi:softproba',
        eval_metric='mlogloss', early_stopping_rounds=300,
        verbosity=0,
    )

def get_lightgbm(seed):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=5000, learning_rate=0.02, max_depth=8,
        num_leaves=127, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, objective='multiclass',
        metric='multi_logloss', verbose=-1,
    )

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Multiple model configs with different seeds and params
model_configs = [
    ("cb_42",    lambda: get_catboost(42)),
    ("cb_123",   lambda: get_catboost(123)),
    ("cb_456",   lambda: get_catboost(456)),
    ("xgb_42",   lambda: get_xgboost(42)),
    ("xgb_123",  lambda: get_xgboost(123)),
    ("lgb_42",   lambda: get_lightgbm(42)),
    ("lgb_123",  lambda: get_lightgbm(123)),
]

oof_preds = {}
test_preds = {}
oof_scores = {}

for model_name, model_fn in model_configs:
    print(f"\n{'='*50}")
    print(f"Training {model_name} | {N_FOLDS}-Fold CV")
    print(f"{'='*50}")

    oof = np.zeros((len(X_fe), len(classes)))
    tst = np.zeros((len(X_test_fe), len(classes)))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_fe, y), 1):
        X_tr = X_fe.iloc[tr_idx].copy()
        X_val = X_fe.iloc[val_idx].copy()
        X_tst = X_test_fe.copy()
        y_tr, y_val = y[tr_idx], y[val_idx]

        # Target encoding per fold (no leakage)
        te = TargetEncoder(cv=5, smooth='auto', shuffle=True, random_state=SEED)
        te_tr = te.fit_transform(X_tr[te_cols], y_tr)
        te_val = te.transform(X_val[te_cols])
        te_tst = te.transform(X_tst[te_cols])

        # Handle output shape
        if te_tr.ndim == 1:
            te_tr = te_tr.reshape(-1, 1)
            te_val = te_val.reshape(-1, 1)
            te_tst = te_tst.reshape(-1, 1)

        n_out = te_tr.shape[1]
        if n_out == len(te_cols):
            te_names = [f'TE_{c}' for c in te_cols]
        else:
            cls_per = n_out // len(te_cols)
            te_names = [f'TE_{c}_{i}' for c in te_cols for i in range(cls_per)]

        for i, name in enumerate(te_names):
            X_tr.loc[:, name] = te_tr[:, i]
            X_val.loc[:, name] = te_val[:, i]
            X_tst.loc[:, name] = te_tst[:, i]

        model = model_fn()
        is_cb = model_name.startswith('cb')

        if is_cb:
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        elif model_name.startswith('xgb'):
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            import lightgbm
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lightgbm.early_stopping(300, verbose=False)])

        oof[val_idx] = model.predict_proba(X_val)
        tst += model.predict_proba(X_tst) / N_FOLDS

        fold_score = balanced_accuracy_score(y_val, np.argmax(oof[val_idx], axis=1))
        print(f"  Fold {fold:2d}: {fold_score:.5f}")

    overall = balanced_accuracy_score(y, np.argmax(oof, axis=1))
    print(f"  >> {model_name} OOF: {overall:.5f}")

    oof_preds[model_name] = oof
    test_preds[model_name] = tst
    oof_scores[model_name] = overall

# ============================================================
# 5. Stacking
# ============================================================
print(f"\n{'='*50}")
print("STACKING")
print(f"{'='*50}")

stack_oof = np.hstack([oof_preds[m[0]] for m in model_configs])
stack_test = np.hstack([test_preds[m[0]] for m in model_configs])

meta_oof = np.zeros((len(y), len(classes)))
meta_test = np.zeros((len(test_ids), len(classes)))

for fold, (tr_idx, val_idx) in enumerate(skf.split(stack_oof, y), 1):
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    meta.fit(stack_oof[tr_idx], y[tr_idx])
    meta_oof[val_idx] = meta.predict_proba(stack_oof[val_idx])
    meta_test += meta.predict_proba(stack_test) / N_FOLDS

stacking_score = balanced_accuracy_score(y, np.argmax(meta_oof, axis=1))
print(f"  Stacking OOF: {stacking_score:.5f}")

# ============================================================
# 6. Weighted Average
# ============================================================
weights = np.array([oof_scores[m[0]] for m in model_configs])
weights = weights / weights.sum()
weighted_oof = sum(oof_preds[m[0]] * w for m, w in zip(model_configs, weights))
weighted_score = balanced_accuracy_score(y, np.argmax(weighted_oof, axis=1))

# ============================================================
# 7. Results & Submissions
# ============================================================
print(f"\n{'='*50}")
print("RESULTS")
print(f"{'='*50}")
all_scores = {**oof_scores, 'stacking': stacking_score, 'wtd_blend': weighted_score}
for name, score in sorted(all_scores.items(), key=lambda x: -x[1]):
    print(f"  {name:12s}: {score:.5f}")

best_name = max(all_scores, key=all_scores.get)
best_score = all_scores[best_name]
print(f"\n  >> Best: {best_name} = {best_score:.5f}")

import os
os.makedirs("submissions", exist_ok=True)

# Save best submission
if best_name == 'stacking':
    best_test = meta_test
elif best_name == 'wtd_blend':
    best_test = sum(test_preds[m[0]] * w for m, w in zip(model_configs, weights))
else:
    best_test = test_preds[best_name]

labels = le.inverse_transform(np.argmax(best_test, axis=1))
pd.DataFrame({'id': test_ids, 'class': labels}).to_csv(
    f"submissions/sub_best_{best_score:.5f}.csv", index=False)

# Save stacking submission too
labels = le.inverse_transform(np.argmax(meta_test, axis=1))
pd.DataFrame({'id': test_ids, 'class': labels}).to_csv(
    f"submissions/sub_stacking_{stacking_score:.5f}.csv", index=False)

print(f"\nTarget: 0.96720 | Achieved: {best_score:.5f}")
if best_score >= 0.9672:
    print("TARGET MET! Ready to submit.")
else:
    print(f"Gap: {0.9672 - best_score:.5f} - need more optimization.")
