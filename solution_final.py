"""
Kaggle Playground S6E6 - Final Solution
Target: accuracy >= 0.9672 (top 100) -> ACHIEVED
Now optimizing for even higher score
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
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

X = train.drop([ID, TARGET], axis=1)
X_test = test.drop([ID], axis=1)
test_ids = test[ID].values

# ============================================================
# 2. Feature Engineering
# ============================================================
def feature_engineering(df):
    df = df.copy()

    # Critical color indices
    df['r-g'] = df['r'] - df['g']
    df['u-r'] = df['u'] - df['r']

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

    # Drop string categoricals
    df = df.drop(columns=['spectral_type', 'galaxy_population'])
    return df

X_fe = feature_engineering(X)
X_test_fe = feature_engineering(X_test)

# ============================================================
# 3. Train Diverse Models
# ============================================================
N_FOLDS = 10
SEED = 42
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

model_configs = [
    ("xgb_a", lambda s: XGBClassifier(
        n_estimators=5000, learning_rate=0.02, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0, random_state=s,
        objective='multi:softproba', eval_metric='mlogloss',
        early_stopping_rounds=300, verbosity=0)),
    ("xgb_b", lambda s: XGBClassifier(
        n_estimators=8000, learning_rate=0.01, max_depth=8,
        subsample=0.8, colsample_bytree=0.9, min_child_weight=5,
        reg_alpha=0.05, reg_lambda=1.5, random_state=s,
        objective='multi:softproba', eval_metric='mlogloss',
        early_stopping_rounds=400, verbosity=0)),
    ("xgb_c", lambda s: XGBClassifier(
        n_estimators=5000, learning_rate=0.02, max_depth=10,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=3,
        reg_alpha=0.2, reg_lambda=0.5, random_state=s,
        objective='multi:softproba', eval_metric='mlogloss',
        early_stopping_rounds=300, verbosity=0)),
    ("lgb_a", lambda s: LGBMClassifier(
        n_estimators=5000, learning_rate=0.02, max_depth=8,
        num_leaves=127, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
        random_state=s, objective='multiclass', metric='multi_logloss',
        verbose=-1)),
    ("lgb_b", lambda s: LGBMClassifier(
        n_estimators=8000, learning_rate=0.01, max_depth=10,
        num_leaves=255, subsample=0.7, colsample_bytree=0.7,
        min_child_samples=10, reg_alpha=0.05, reg_lambda=1.5,
        random_state=s, objective='multiclass', metric='multi_logloss',
        verbose=-1)),
]

oof_preds = {}
test_preds = {}
oof_scores = {}

for model_name, model_fn in model_configs:
    print(f"\nTraining {model_name} | {N_FOLDS}-Fold CV")
    oof = np.zeros((len(X_fe), len(classes)))
    tst = np.zeros((len(X_test_fe), len(classes)))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_fe, y), 1):
        X_tr, y_tr = X_fe.iloc[tr_idx], y[tr_idx]
        X_val, y_val = X_fe.iloc[val_idx], y[val_idx]

        model = model_fn(SEED + fold)

        if model_name.startswith("xgb"):
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            import lightgbm
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lightgbm.early_stopping(300, verbose=False)])

        oof[val_idx] = model.predict_proba(X_val)
        tst += model.predict_proba(X_test_fe) / N_FOLDS

    acc = accuracy_score(y, np.argmax(oof, axis=1))
    ba = balanced_accuracy_score(y, np.argmax(oof, axis=1))
    print(f"  >> {model_name} Accuracy: {acc:.5f} | BA: {ba:.5f}")

    oof_preds[model_name] = oof
    test_preds[model_name] = tst
    oof_scores[model_name] = acc

# ============================================================
# 4. Ensemble Strategies
# ============================================================
print(f"\n{'='*50}")
print("ENSEMBLE STRATEGIES")
print(f"{'='*50}")

# Simple average
avg_oof = sum(oof_preds.values()) / len(oof_preds)
avg_acc = accuracy_score(y, np.argmax(avg_oof, axis=1))
print(f"  Simple Average: {avg_acc:.5f}")

# Weighted average (by individual accuracy)
weights = np.array([oof_scores[m[0]] for m in model_configs])
weights = weights / weights.sum()
wtd_oof = sum(oof_preds[m[0]] * w for m, w in zip(model_configs, weights))
wtd_acc = accuracy_score(y, np.argmax(wtd_oof, axis=1))
print(f"  Weighted Average: {wtd_acc:.5f}")

# Stacking
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

stack_acc = accuracy_score(y, np.argmax(meta_oof, axis=1))
print(f"  Stacking: {stack_acc:.5f}")

# ============================================================
# 5. Results & Submissions
# ============================================================
print(f"\n{'='*50}")
print("FINAL RESULTS")
print(f"{'='*50}")
all_scores = {**oof_scores, 'avg_blend': avg_acc, 'wtd_blend': wtd_acc, 'stacking': stack_acc}
for name, score in sorted(all_scores.items(), key=lambda x: -x[1]):
    print(f"  {name:12s}: {score:.5f}")

best_name = max(all_scores, key=all_scores.get)
best_score = all_scores[best_name]
print(f"\n  >> Best: {best_name} = {best_score:.5f}")

import os
os.makedirs("submissions", exist_ok=True)

# Generate submission for best strategy
if best_name == 'stacking':
    best_test = meta_test
elif best_name == 'wtd_blend':
    best_test = sum(test_preds[m[0]] * w for m, w in zip(model_configs, weights))
elif best_name == 'avg_blend':
    best_test = avg_oof_test = sum(test_preds.values()) / len(test_preds)
else:
    best_test = test_preds[best_name]

# Save best submission
labels = le.inverse_transform(np.argmax(best_test, axis=1))
sub_path = f"submissions/best_{best_name}_{best_score:.5f}.csv"
pd.DataFrame({'id': test_ids, 'class': labels}).to_csv(sub_path, index=False)
print(f"\n  Saved: {sub_path}")

# Save all individual submissions too
for name in oof_scores:
    labels = le.inverse_transform(np.argmax(test_preds[name], axis=1))
    pd.DataFrame({'id': test_ids, 'class': labels}).to_csv(
        f"submissions/{name}_{oof_scores[name]:.5f}.csv", index=False)

# Save stacking submission
labels = le.inverse_transform(np.argmax(meta_test, axis=1))
pd.DataFrame({'id': test_ids, 'class': labels}).to_csv(
    f"submissions/stacking_{stack_acc:.5f}.csv", index=False)

print(f"\n  Target: 0.96720 | Achieved: {best_score:.5f}")
if best_score >= 0.9672:
    print("  >>> TARGET MET! Ready to submit when you give the go-ahead. <<<")
else:
    print(f"  Gap: {0.9672 - best_score:.5f}")
