"""
Kaggle S6E6 - Pseudo-labeling: use high-confidence test predictions as extra training
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print = lambda *a, **k: (sys.stdout.write(' '.join(map(str, a)) + k.get('end', '\n')), sys.stdout.flush())

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
ID, TARGET = 'id', 'class'
y_raw = train[TARGET]
le = LabelEncoder()
y = le.fit_transform(y_raw)

def fe(df):
    df = df.copy()
    df['r-g'] = df['r'] - df['g']
    df['u-r'] = df['u'] - df['r']
    bands = ['u','g','r','i','z']
    for i in range(len(bands)):
        for j in range(i+1, len(bands)):
            df[f'{bands[i]}-{bands[j]}'] = df[bands[i]] - df[bands[j]]
    df['log_z'] = np.log1p(np.abs(df['redshift'])) * np.sign(df['redshift'])
    for b in bands:
        df[f'{b}/z'] = df[b] / (df['redshift'] + 1e-6)
    df['spec_enc'] = df['spectral_type'].map({'M':0,'G/K':1,'A/F':2,'O/B':3})
    df['pop_enc'] = df['galaxy_population'].map({'Blue_Cloud':0,'Red_Sequence':1})
    df['spec*z'] = df['spec_enc'] * df['redshift']
    df['pop*z'] = df['pop_enc'] * df['redshift']
    df = df.drop(columns=['spectral_type','galaxy_population'])
    return df

X = fe(train.drop([ID, TARGET], axis=1))
X_test = fe(test.drop([ID], axis=1))
test_ids = test[ID].values

N_FOLDS = 10
SEED = 42
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Step 1: Train initial model to get test predictions
print("Step 1: Initial model for pseudo-labeling...")
initial_test = np.zeros((len(X_test), 3))
initial_oof = np.zeros((len(X), 3))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    m = XGBClassifier(
        n_estimators=5000, learning_rate=0.02, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0, random_state=SEED+fold,
        objective='multi:softproba', eval_metric='mlogloss',
        early_stopping_rounds=300, verbosity=0)
    m.fit(X.iloc[tr_idx], y[tr_idx], eval_set=[(X.iloc[val_idx], y[val_idx])], verbose=False)
    initial_oof[val_idx] = m.predict_proba(X.iloc[val_idx])
    initial_test += m.predict_proba(X_test) / N_FOLDS

init_acc = accuracy_score(y, np.argmax(initial_oof, axis=1))
print(f"  Initial OOF: {init_acc:.5f}")

# Step 2: Select high-confidence test samples
print("\nStep 2: Selecting high-confidence pseudo-labels...")
max_probs = np.max(initial_test, axis=1)
thresholds = [0.999, 0.995, 0.99, 0.98]

for thresh in thresholds:
    mask = max_probs >= thresh
    n_selected = mask.sum()
    print(f"  Threshold {thresh}: {n_selected} samples ({n_selected/len(X_test)*100:.1f}%)")

# Use threshold 0.999 (very conservative)
thresh = 0.999
pseudo_mask = max_probs >= thresh
pseudo_X = X_test[pseudo_mask].copy()
pseudo_y = le.inverse_transform(np.argmax(initial_test[pseudo_mask], axis=1))
pseudo_y_enc = np.argmax(initial_test[pseudo_mask], axis=1)
print(f"\n  Using {pseudo_mask.sum()} pseudo-labeled samples at threshold {thresh}")

# Step 3: Retrain with augmented data
print("\nStep 3: Training with pseudo-labeled data...")
X_aug = pd.concat([X, pseudo_X], ignore_index=True)
y_aug = np.concatenate([y, pseudo_y_enc])
print(f"  Augmented size: {len(X_aug)} (was {len(X)})")

pseudo_oof = np.zeros((len(X), 3))
pseudo_test = np.zeros((len(X_test), 3))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    # Add all pseudo-labels to training set
    pseudo_tr_idx = np.concatenate([tr_idx, np.arange(len(X), len(X_aug))])

    m = XGBClassifier(
        n_estimators=5000, learning_rate=0.02, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0, random_state=SEED+fold,
        objective='multi:softproba', eval_metric='mlogloss',
        early_stopping_rounds=300, verbosity=0)
    m.fit(X_aug.iloc[pseudo_tr_idx], y_aug[pseudo_tr_idx],
          eval_set=[(X_aug.iloc[val_idx], y_aug[val_idx])], verbose=False)
    pseudo_oof[val_idx] = m.predict_proba(X_aug.iloc[val_idx])
    pseudo_test += m.predict_proba(X_test) / N_FOLDS

pseudo_acc = accuracy_score(y, np.argmax(pseudo_oof, axis=1))
print(f"  Pseudo-label OOF: {pseudo_acc:.5f}")
print(f"  Improvement: {pseudo_acc - init_acc:+.5f}")

# Step 4: Blend with RealMLP
print("\nStep 4: Blending with RealMLP...")
rm_oof = pd.read_csv("oof_preds.csv")[['GALAXY','QSO','STAR']].values
rm_test = pd.read_csv("test_preds.csv")[['GALAXY','QSO','STAR']].values

# Try different weights
best_acc = 0
best_w = 0
for w in np.arange(0.3, 0.95, 0.01):
    blend = w * rm_oof + (1-w) * pseudo_oof
    acc = accuracy_score(y, np.argmax(blend, axis=1))
    if acc > best_acc:
        best_acc = acc
        best_w = w

print(f"  Best blend: rm={best_w:.2f} pseudo={1-best_w:.2f} acc={best_acc:.5f}")

# Also try 3-way: RealMLP + initial + pseudo
best_3way = 0
best_3w = None
for w1 in np.arange(0.3, 0.8, 0.05):
    for w2 in np.arange(0.05, 0.5, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05:
            continue
        blend = w1*rm_oof + w2*initial_oof + w3*pseudo_oof
        acc = accuracy_score(y, np.argmax(blend, axis=1))
        if acc > best_3way:
            best_3way = acc
            best_3w = (w1, w2, w3)

print(f"  Best 3-way: rm={best_3w[0]:.2f} init={best_3w[1]:.2f} pseudo={best_3w[2]:.2f} acc={best_3way:.5f}")

# Save best submission
os.makedirs("submissions", exist_ok=True)
final_acc = max(best_acc, best_3way)

if best_3way > best_acc:
    w1, w2, w3 = best_3w
    final_test = w1*rm_test + w2*initial_test + w3*pseudo_test
    desc = f"rm({w1:.2f})+init({w2:.2f})+pseudo({w3:.2f})"
else:
    final_test = best_w * rm_test + (1-best_w) * pseudo_test
    desc = f"rm({best_w:.2f})+pseudo({1-best_w:.2f})"

labels = le.inverse_transform(np.argmax(final_test, axis=1))
path = f"submissions/pseudo_{final_acc:.5f}.csv"
pd.DataFrame({'id': test_ids, 'class': labels}).to_csv(path, index=False)
print(f"\nSaved: {path} ({desc})")
print(f"Target: 0.97100 | Achieved: {final_acc:.5f}")
