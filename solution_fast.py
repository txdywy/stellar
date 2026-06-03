"""
Kaggle Playground S6E6 - Fast Final (4 models, no lgb_b)
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

print = lambda *a, **k: (sys.stdout.write(' '.join(map(str, a)) + k.get('end', '\n')), sys.stdout.flush())

# Load
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
ID, TARGET = 'id', 'class'
y_raw = train[TARGET]
le = LabelEncoder()
y = le.fit_transform(y_raw)
X = train.drop([ID, TARGET], axis=1)
X_test = test.drop([ID], axis=1)
test_ids = test[ID].values

# Feature engineering
def fe(df):
    df = df.copy()
    df['r-g'] = df['r'] - df['g']
    df['u-r'] = df['u'] - df['r']
    bands = ['u', 'g', 'r', 'i', 'z']
    for i in range(len(bands)):
        for j in range(i+1, len(bands)):
            df[f'{bands[i]}-{bands[j]}'] = df[bands[i]] - df[bands[j]]
    df['log_z'] = np.log1p(np.abs(df['redshift'])) * np.sign(df['redshift'])
    for b in bands:
        df[f'{b}/z'] = df[b] / (df['redshift'] + 1e-6)
    df['spec_enc'] = df['spectral_type'].map({'M': 0, 'G/K': 1, 'A/F': 2, 'O/B': 3})
    df['pop_enc'] = df['galaxy_population'].map({'Blue_Cloud': 0, 'Red_Sequence': 1})
    df['spec*z'] = df['spec_enc'] * df['redshift']
    df['pop*z'] = df['pop_enc'] * df['redshift']
    df = df.drop(columns=['spectral_type', 'galaxy_population'])
    return df

X_fe = fe(X)
X_test_fe = fe(X_test)

N_FOLDS = 10
SEED = 42
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

models = [
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
]

oof_preds = {}
test_preds = {}
scores = {}

for name, fn in models:
    print(f"Training {name}...")
    oof = np.zeros((len(X_fe), 3))
    tst = np.zeros((len(X_test_fe), 3))
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_fe, y), 1):
        m = fn(SEED + fold)
        if name.startswith("xgb"):
            m.fit(X_fe.iloc[tr_idx], y[tr_idx],
                  eval_set=[(X_fe.iloc[val_idx], y[val_idx])], verbose=False)
        else:
            import lightgbm as lgb
            m.fit(X_fe.iloc[tr_idx], y[tr_idx],
                  eval_set=[(X_fe.iloc[val_idx], y[val_idx])],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[val_idx] = m.predict_proba(X_fe.iloc[val_idx])
        tst += m.predict_proba(X_test_fe) / N_FOLDS

    acc = accuracy_score(y, np.argmax(oof, axis=1))
    ba = balanced_accuracy_score(y, np.argmax(oof, axis=1))
    print(f"  {name}: Acc={acc:.5f} BA={ba:.5f}")
    oof_preds[name] = oof
    test_preds[name] = tst
    scores[name] = acc

# Ensemble
print("\nEnsembling...")
avg_oof = sum(oof_preds.values()) / len(oof_preds)
avg_acc = accuracy_score(y, np.argmax(avg_oof, axis=1))
print(f"  Average: {avg_acc:.5f}")

# Stacking
stack_oof = np.hstack(list(oof_preds.values()))
stack_test = np.hstack(list(test_preds.values()))
meta_oof = np.zeros((len(y), 3))
meta_test = np.zeros((len(test_ids), 3))
for fold, (tr_idx, val_idx) in enumerate(skf.split(stack_oof, y), 1):
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    meta.fit(stack_oof[tr_idx], y[tr_idx])
    meta_oof[val_idx] = meta.predict_proba(stack_oof[val_idx])
    meta_test += meta.predict_proba(stack_test) / N_FOLDS
stack_acc = accuracy_score(y, np.argmax(meta_oof, axis=1))
print(f"  Stacking: {stack_acc:.5f}")

# Weighted
weights = np.array([scores[m[0]] for m in models])
weights = weights / weights.sum()
wtd_oof = sum(oof_preds[m[0]] * w for m, w in zip(models, weights))
wtd_acc = accuracy_score(y, np.argmax(wtd_oof, axis=1))
wtd_test = sum(test_preds[m[0]] * w for m, w in zip(models, weights))
print(f"  Weighted: {wtd_acc:.5f}")

# Pick best
all_s = {**scores, 'avg': avg_acc, 'stack': stack_acc, 'wtd': wtd_acc}
best = max(all_s, key=all_s.get)
print(f"\nBest: {best} = {all_s[best]:.5f}")

os.makedirs("submissions", exist_ok=True)

# Save best
if best == 'stack':
    best_tst = meta_test
elif best == 'wtd':
    best_tst = wtd_test
elif best == 'avg':
    best_tst = sum(test_preds.values()) / len(test_preds)
else:
    best_tst = test_preds[best]

labels = le.inverse_transform(np.argmax(best_tst, axis=1))
path = f"submissions/best_{all_s[best]:.5f}.csv"
pd.DataFrame({'id': test_ids, 'class': labels}).to_csv(path, index=False)
print(f"Saved: {path}")

# Save stacking too
labels = le.inverse_transform(np.argmax(meta_test, axis=1))
path = f"submissions/stacking_{stack_acc:.5f}.csv"
pd.DataFrame({'id': test_ids, 'class': labels}).to_csv(path, index=False)
print(f"Saved: {path}")

# Save all individual
for name in scores:
    labels = le.inverse_transform(np.argmax(test_preds[name], axis=1))
    path = f"submissions/{name}_{scores[name]:.5f}.csv"
    pd.DataFrame({'id': test_ids, 'class': labels}).to_csv(path, index=False)
    print(f"Saved: {path}")

print(f"\nTarget: 0.96720 | Best: {all_s[best]:.5f}")
print("READY TO SUBMIT when you give the go-ahead!")
