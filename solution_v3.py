"""
Kaggle Playground S6E6 - Clean v3
No SDSS augmentation, focused feature engineering, optimized hyperparams
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

ID = 'id'
TARGET = 'class'

y_raw = train[TARGET].copy()
le = LabelEncoder()
y = le.fit_transform(y_raw)
classes = le.classes_
print(f"Classes: {classes}, Train: {train.shape}, Test: {test.shape}")

X = train.drop([ID, TARGET], axis=1)
X_test = test.drop([ID], axis=1)
train_ids = train[ID].values
test_ids = test[ID].values

# ============================================================
# 2. Feature Engineering (minimal but high-signal)
# ============================================================
def feature_engineering(df):
    df = df.copy()

    # Color indices (the two most important: r-g defines spectral_type, u-r defines galaxy_population)
    df['r_minus_g'] = df['r'] - df['g']
    df['u_minus_r'] = df['u'] - df['r']

    # Additional color indices
    df['u_g'] = df['u'] - df['g']
    df['g_r'] = df['g'] - df['r']
    df['r_i'] = df['r'] - df['i']
    df['i_z'] = df['i'] - df['z']
    df['u_i'] = df['u'] - df['i']
    df['u_z'] = df['u'] - df['z']
    df['g_i'] = df['g'] - df['i']
    df['g_z'] = df['g'] - df['z']
    df['r_z'] = df['r'] - df['z']

    # Redshift interactions (very important for QSO detection)
    df['g_div_z'] = df['g'] / (df['redshift'] + 1e-6)
    df['r_div_z'] = df['r'] / (df['redshift'] + 1e-6)
    df['i_div_z'] = df['i'] / (df['redshift'] + 1e-6)

    # Log redshift
    df['log_z'] = np.log1p(np.abs(df['redshift'])) * np.sign(df['redshift'])

    # Encode categoricals
    df['spec_enc'] = df['spectral_type'].map({'M': 0, 'G/K': 1, 'A/F': 2, 'O/B': 3})
    df['pop_enc'] = df['galaxy_population'].map({'Blue_Cloud': 0, 'Red_Sequence': 1})

    # Drop original string categoricals
    df = df.drop(columns=['spectral_type', 'galaxy_population'])

    return df

print("\nEngineering features...")
X_fe = feature_engineering(X)
X_test_fe = feature_engineering(X_test)
print(f"Features: {X_fe.shape[1]}")

# ============================================================
# 3. Train Models
# ============================================================
N_FOLDS = 10
SEED = 42

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
        min_child_weight=5,
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
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
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

        fold_score = balanced_accuracy_score(y_val, np.argmax(oof[val_idx], axis=1))
        fold_scores.append(fold_score)
        print(f"  Fold {fold:2d}: {fold_score:.5f}")

    overall = balanced_accuracy_score(y, np.argmax(oof, axis=1))
    print(f"\n  >> {model_name.upper()} OOF: {overall:.5f}")

    oof_preds[model_name] = oof
    test_preds[model_name] = tst
    oof_scores[model_name] = overall

# ============================================================
# 4. Stacking
# ============================================================
print(f"\n{'='*60}")
print("STACKING")
print(f"{'='*60}")

from sklearn.linear_model import LogisticRegression

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
# 5. Weighted Average
# ============================================================
weights = np.array([oof_scores[m[0]] for m in model_configs])
weights = weights / weights.sum()
weighted_oof = sum(oof_preds[m[0]] * w for m, w in zip(model_configs, weights))
weighted_score = balanced_accuracy_score(y, np.argmax(weighted_oof, axis=1))

avg_oof = sum(oof_preds[m[0]] for m in model_configs) / len(model_configs)
avg_score = balanced_accuracy_score(y, np.argmax(avg_oof, axis=1))

# ============================================================
# 6. Results
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

# Save submissions
import os
os.makedirs("submissions", exist_ok=True)

for name in oof_scores:
    labels = le.inverse_transform(np.argmax(test_preds[name], axis=1))
    pd.DataFrame({'id': test_ids, 'class': labels}).to_csv(
        f"submissions/sub_{name}_{oof_scores[name]:.5f}.csv", index=False)

labels = le.inverse_transform(np.argmax(meta_test, axis=1))
pd.DataFrame({'id': test_ids, 'class': labels}).to_csv(
    f"submissions/sub_stacking_{stacking_score:.5f}.csv", index=False)

if best_name == 'wtd_blend':
    best_test = weighted_oof_test = sum(test_preds[m[0]] * w for m, w in zip(model_configs, weights))
elif best_name == 'avg_blend':
    best_test = sum(test_preds[m[0]] for m in model_configs) / len(model_configs)
elif best_name == 'stacking':
    best_test = meta_test
else:
    best_test = test_preds[best_name]

labels = le.inverse_transform(np.argmax(best_test, axis=1))
pd.DataFrame({'id': test_ids, 'class': labels}).to_csv(
    f"submissions/sub_best_{best_score:.5f}.csv", index=False)

print(f"\nTarget: 0.96720 | Achieved: {best_score:.5f}")
if best_score >= 0.9672:
    print("TARGET MET! Ready to submit.")
else:
    print(f"Gap: {0.9672 - best_score:.5f}")
