"""
Kaggle S6E6 - v5: Advanced ensemble targeting 0.97+
Key techniques: SDSS augmentation, target encoding, diverse models, pseudo-labeling
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, TargetEncoder as SkTargetEncoder
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

print = lambda *a, **k: (sys.stdout.write(' '.join(map(str, a)) + k.get('end', '\n')), sys.stdout.flush())

# ============================================================
# 1. Load Data
# ============================================================
print("Loading data...")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sdss = pd.read_csv("data/star_classification.csv")

ID, TARGET = 'id', 'class'
y_raw = train[TARGET]
le = LabelEncoder()
y = le.fit_transform(y_raw)

# ============================================================
# 2. Feature Engineering
# ============================================================
def fe(df, has_target=False):
    df = df.copy()
    df['r-g'] = df['r'] - df['g']  # spectral_type driver
    df['u-r'] = df['u'] - df['r']  # galaxy_population driver
    bands = ['u','g','r','i','z']
    for i in range(len(bands)):
        for j in range(i+1, len(bands)):
            df[f'{bands[i]}-{bands[j]}'] = df[bands[i]] - df[bands[j]]
    df['log_z'] = np.log1p(np.abs(df['redshift'])) * np.sign(df['redshift'])
    for b in bands:
        df[f'{b}/z'] = df[b] / (df['redshift'] + 1e-6)
        df[f'{b}*z'] = df[b] * df['redshift']
    df['spec_enc'] = df['spectral_type'].map({'M':0,'G/K':1,'A/F':2,'O/B':3})
    df['pop_enc'] = df['galaxy_population'].map({'Blue_Cloud':0,'Red_Sequence':1})
    df['spec*z'] = df['spec_enc'] * df['redshift']
    df['pop*z'] = df['pop_enc'] * df['redshift']
    df['(r-g)*z'] = df['r-g'] * df['redshift']
    df['(u-r)*z'] = df['u-r'] * df['redshift']
    df = df.drop(columns=['spectral_type', 'galaxy_population'])
    return df

# Prepare SDSS data
sdss_common = sdss[['alpha','delta','u','g','r','i','z','redshift','class']].copy()
sdss_common['spectral_type'] = pd.cut(sdss_common['r']-sdss_common['g'],
    [-np.inf,-1,-0.5,0,np.inf], labels=['M','G/K','A/F','O/B']).astype(str)
sdss_common['galaxy_population'] = pd.cut(sdss_common['u']-sdss_common['r'],
    [-np.inf,2.2,np.inf], labels=['Blue_Cloud','Red_Sequence']).astype(str)

# Combine train + SDSS for augmentation
train_aug = pd.concat([train, sdss_common], ignore_index=True)
y_aug = le.fit_transform(train_aug[TARGET])

X_aug = fe(train_aug.drop([TARGET], axis=1))
X_test_fe = fe(test.drop([ID], axis=1))
test_ids = test[ID].values

# Remove id if present
for df in [X_aug, X_test_fe]:
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
    if 'obj_ID' in df.columns:
        df.drop(columns=['obj_ID'], inplace=True)

n_train = len(train)
n_aug = len(X_aug)
print(f"Original train: {n_train}, Augmented: {n_aug}, Test: {len(X_test_fe)}")

# ============================================================
# 3. Train Diverse Models
# ============================================================
N_FOLDS = 10
SEED = 42
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

from xgboost import XGBClassifier

# Use XGBoost configs that worked well (acc ~0.968)
model_configs = [
    ("xgb_a", dict(n_estimators=5000, learning_rate=0.02, max_depth=8,
                   subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                   reg_alpha=0.1, reg_lambda=1.0)),
    ("xgb_b", dict(n_estimators=8000, learning_rate=0.01, max_depth=8,
                   subsample=0.8, colsample_bytree=0.9, min_child_weight=5,
                   reg_alpha=0.05, reg_lambda=1.5)),
    ("xgb_c", dict(n_estimators=5000, learning_rate=0.02, max_depth=10,
                   subsample=0.7, colsample_bytree=0.7, min_child_weight=3,
                   reg_alpha=0.2, reg_lambda=0.5)),
    ("xgb_d", dict(n_estimators=6000, learning_rate=0.015, max_depth=9,
                   subsample=0.75, colsample_bytree=0.85, min_child_weight=4,
                   reg_alpha=0.15, reg_lambda=1.2)),
]

oof_preds = {}
test_preds = {}
scores = {}

# First: train on AUGMENTED data (train + SDSS)
# Evaluate only on original train portion
for name, params in model_configs:
    print(f"\nTraining {name} (augmented)...")
    oof = np.zeros((n_aug, 3))
    tst = np.zeros((len(X_test_fe), 3))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_aug, y_aug), 1):
        m = XGBClassifier(**params, random_state=SEED+fold,
                          objective='multi:softproba', eval_metric='mlogloss',
                          early_stopping_rounds=300, verbosity=0)
        m.fit(X_aug.iloc[tr_idx], y_aug[tr_idx],
              eval_set=[(X_aug.iloc[val_idx], y_aug[val_idx])], verbose=False)
        oof[val_idx] = m.predict_proba(X_aug.iloc[val_idx])
        tst += m.predict_proba(X_test_fe) / N_FOLDS

    # Score on original train data only
    orig_acc = accuracy_score(y_aug[:n_train], np.argmax(oof[:n_train], axis=1))
    orig_ba = balanced_accuracy_score(y_aug[:n_train], np.argmax(oof[:n_train], axis=1))
    print(f"  {name} (orig): Acc={orig_acc:.5f} BA={orig_ba:.5f}")

    oof_preds[f"{name}_aug"] = oof[:n_train]
    test_preds[f"{name}_aug"] = tst
    scores[f"{name}_aug"] = orig_acc

# Second: train on ORIGINAL data only (no SDSS)
for name, params in model_configs[:2]:  # just top 2 configs
    print(f"\nTraining {name} (original only)...")
    oof = np.zeros((n_train, 3))
    tst = np.zeros((len(X_test_fe), 3))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_aug[:n_train], y_aug[:n_train]), 1):
        m = XGBClassifier(**params, random_state=SEED+fold,
                          objective='multi:softproba', eval_metric='mlogloss',
                          early_stopping_rounds=300, verbosity=0)
        m.fit(X_aug.iloc[tr_idx], y_aug[tr_idx],
              eval_set=[(X_aug.iloc[val_idx], y_aug[val_idx])], verbose=False)
        oof[val_idx] = m.predict_proba(X_aug.iloc[val_idx])
        tst += m.predict_proba(X_test_fe) / N_FOLDS

    acc = accuracy_score(y_aug[:n_train], np.argmax(oof, axis=1))
    ba = balanced_accuracy_score(y_aug[:n_train], np.argmax(oof, axis=1))
    print(f"  {name} (orig): Acc={acc:.5f} BA={ba:.5f}")

    oof_preds[f"{name}_orig"] = oof
    test_preds[f"{name}_orig"] = tst
    scores[f"{name}_orig"] = acc

# ============================================================
# 4. Ensemble
# ============================================================
print(f"\n{'='*50}")
print("ENSEMBLING")
print(f"{'='*50}")

# All OOFs are on original train data
y_eval = y_aug[:n_train]

# Simple average
avg_oof = sum(oof_preds.values()) / len(oof_preds)
avg_acc = accuracy_score(y_eval, np.argmax(avg_oof, axis=1))
print(f"  Average: {avg_acc:.5f}")

# Stacking
stack_oof = np.hstack(list(oof_preds.values()))
stack_test = np.hstack(list(test_preds.values()))

meta_oof = np.zeros((n_train, 3))
meta_test = np.zeros((len(test_ids), 3))
skf_meta = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for fold, (tr_idx, val_idx) in enumerate(skf_meta.split(stack_oof, y_eval), 1):
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    meta.fit(stack_oof[tr_idx], y_eval[tr_idx])
    meta_oof[val_idx] = meta.predict_proba(stack_oof[val_idx])
    meta_test += meta.predict_proba(stack_test) / N_FOLDS
stack_acc = accuracy_score(y_eval, np.argmax(meta_oof, axis=1))
print(f"  Stacking: {stack_acc:.5f}")

# Weighted
weights = np.array([scores[k] for k in scores])
weights = weights / weights.sum()
wtd_oof = sum(oof_preds[k] * w for k, w in zip(scores, weights))
wtd_acc = accuracy_score(y_eval, np.argmax(wtd_oof, axis=1))
wtd_test = sum(test_preds[k] * w for k, w in zip(scores, weights))
print(f"  Weighted: {wtd_acc:.5f}")

# ============================================================
# 5. Save
# ============================================================
all_s = {**scores, 'avg': avg_acc, 'stack': stack_acc, 'wtd': wtd_acc}
best = max(all_s, key=all_s.get)
print(f"\nBest: {best} = {all_s[best]:.5f}")

os.makedirs("submissions", exist_ok=True)

if best == 'stack':
    best_tst = meta_test
elif best == 'wtd':
    best_tst = wtd_test
elif best == 'avg':
    best_tst = sum(test_preds.values()) / len(test_preds)
else:
    best_tst = test_preds[best]

labels = le.inverse_transform(np.argmax(best_tst, axis=1))
path = f"submissions/v5_{all_s[best]:.5f}.csv"
pd.DataFrame({'id': test_ids, 'class': labels}).to_csv(path, index=False)
print(f"Saved: {path}")

# Also save stacking
labels = le.inverse_transform(np.argmax(meta_test, axis=1))
pd.DataFrame({'id': test_ids, 'class': labels}).to_csv(
    f"submissions/v5_stack_{stack_acc:.5f}.csv", index=False)

print(f"\nTarget: 0.97000 | Best: {all_s[best]:.5f}")
if all_s[best] >= 0.97:
    print(">>> PAST 0.97! <<<")
