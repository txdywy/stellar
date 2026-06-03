"""
Kaggle S6E6 - FINAL PUSH: All-out optimization for #1
Strategy: Maximize diverse models + optimal blending + pseudo-labeling
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
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
realmlp_oof = pd.read_csv("oof_preds.csv")
realmlp_test = pd.read_csv("test_preds.csv")

ID, TARGET = 'id', 'class'
y = LabelEncoder().fit_transform(train[TARGET])
classes = ['GALAXY', 'QSO', 'STAR']
rm_oof = realmlp_oof[classes].values
rm_test = realmlp_test[classes].values
test_ids = test[ID].values

# ============================================================
# 2. Multiple Feature Engineering Strategies
# ============================================================
def fe_v1(df):
    """Basic color indices + redshift interactions"""
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

def fe_v2(df):
    """Extended: more interactions + polynomial features"""
    df = fe_v1(df)
    # Additional interactions
    df['(r-g)*z'] = df['r-g'] * df['redshift']
    df['(u-r)*z'] = df['u-r'] * df['redshift']
    df['(r-g)^2'] = df['r-g'] ** 2
    df['(u-r)^2'] = df['u-r'] ** 2
    df['z^2'] = df['redshift'] ** 2
    # Ratio combinations
    bands = ['u','g','r','i','z']
    for b in bands:
        df[f'{b}*z2'] = df[b] * df['redshift'] ** 2
    return df

def fe_v3(df):
    """Raw features only (for CatBoost native categoricals)"""
    df = df.copy()
    return df

# ============================================================
# 3. Model Configs (very diverse)
# ============================================================
N_FOLDS = 10
SEED = 42
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

xgb_configs = [
    # name, fe_func, params
    ("xgb_a1", fe_v1, dict(n_estimators=5000, learning_rate=0.02, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0)),
    ("xgb_b1", fe_v1, dict(n_estimators=8000, learning_rate=0.01, max_depth=8,
        subsample=0.8, colsample_bytree=0.9, min_child_weight=5, reg_alpha=0.05, reg_lambda=1.5)),
    ("xgb_c1", fe_v1, dict(n_estimators=5000, learning_rate=0.02, max_depth=10,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=3, reg_alpha=0.2, reg_lambda=0.5)),
    ("xgb_d1", fe_v1, dict(n_estimators=6000, learning_rate=0.015, max_depth=9,
        subsample=0.75, colsample_bytree=0.85, min_child_weight=4, reg_alpha=0.15, reg_lambda=1.2)),
    ("xgb_a2", fe_v2, dict(n_estimators=5000, learning_rate=0.02, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0)),
    ("xgb_b2", fe_v2, dict(n_estimators=8000, learning_rate=0.01, max_depth=8,
        subsample=0.8, colsample_bytree=0.9, min_child_weight=5, reg_alpha=0.05, reg_lambda=1.5)),
    # Different seeds
    ("xgb_e1", fe_v1, dict(n_estimators=5000, learning_rate=0.02, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0)),
    ("xgb_f1", fe_v1, dict(n_estimators=5000, learning_rate=0.02, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0)),
]

lgb_configs = [
    ("lgb_a1", fe_v1, dict(n_estimators=5000, learning_rate=0.02, max_depth=8,
        num_leaves=127, subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0)),
    ("lgb_b1", fe_v1, dict(n_estimators=5000, learning_rate=0.02, max_depth=10,
        num_leaves=255, subsample=0.7, colsample_bytree=0.7, min_child_samples=10,
        reg_alpha=0.05, reg_lambda=1.5)),
]

# ============================================================
# 4. Train All Models
# ============================================================
all_oof = {'realmlp': rm_oof}
all_test = {'realmlp': rm_test}
all_scores = {'realmlp': accuracy_score(y, np.argmax(rm_oof, axis=1))}
print(f"RealMLP OOF: {all_scores['realmlp']:.5f}")

# XGBoost models
for name, fe_func, params in xgb_configs:
    print(f"\nTraining {name}...")
    X = fe_func(train.drop([ID, TARGET], axis=1))
    X_test = fe_func(test.drop([ID], axis=1))

    oof = np.zeros((len(X), 3))
    tst = np.zeros((len(X_test), 3))
    seed_offset = 0 if 'e' not in name else (100 if 'e' in name else 200)
    if 'f' in name: seed_offset = 300

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        m = XGBClassifier(**params, random_state=SEED+seed_offset+fold,
                          objective='multi:softproba', eval_metric='mlogloss',
                          early_stopping_rounds=300, verbosity=0)
        m.fit(X.iloc[tr_idx], y[tr_idx], eval_set=[(X.iloc[val_idx], y[val_idx])], verbose=False)
        oof[val_idx] = m.predict_proba(X.iloc[val_idx])
        tst += m.predict_proba(X_test) / N_FOLDS

    acc = accuracy_score(y, np.argmax(oof, axis=1))
    print(f"  {name}: {acc:.5f}")
    all_oof[name] = oof
    all_test[name] = tst
    all_scores[name] = acc

# CatBoost (native categoricals, no FE)
print(f"\nTraining cb_native...")
X_raw = train.drop([ID, TARGET], axis=1)
X_test_raw = test.drop([ID], axis=1)
oof = np.zeros((len(X_raw), 3))
tst = np.zeros((len(X_test_raw), 3))
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_raw, y), 1):
    m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=8,
        random_seed=SEED+fold, loss_function='MultiClass', verbose=0,
        early_stopping_rounds=200, cat_features=['spectral_type','galaxy_population'])
    m.fit(X_raw.iloc[tr_idx], y[tr_idx], eval_set=(X_raw.iloc[val_idx], y[val_idx]), use_best_model=True)
    oof[val_idx] = m.predict_proba(X_raw.iloc[val_idx])
    tst += m.predict_proba(X_test_raw) / N_FOLDS
acc = accuracy_score(y, np.argmax(oof, axis=1))
print(f"  cb_native: {acc:.5f}")
all_oof['cb_native'] = oof
all_test['cb_native'] = tst
all_scores['cb_native'] = acc

# ============================================================
# 5. Find Optimal Blend
# ============================================================
print(f"\n{'='*60}")
print("OPTIMAL BLEND SEARCH")
print(f"{'='*60}")

model_names = list(all_scores.keys())
n_models = len(model_names)
print(f"Models: {n_models}")
for name in model_names:
    print(f"  {name}: {all_scores[name]:.5f}")

# Stack all OOFs
stack_oof = np.hstack([all_oof[n] for n in model_names])
stack_test = np.hstack([all_test[n] for n in model_names])

# Try pairwise blends with RealMLP
print(f"\n--- Pairwise blends with RealMLP ---")
best_pair_acc = 0
best_pair = None
for name in model_names:
    if name == 'realmlp':
        continue
    for w in np.arange(0.3, 0.95, 0.05):
        blend = w * rm_oof + (1-w) * all_oof[name]
        acc = accuracy_score(y, np.argmax(blend, axis=1))
        if acc > best_pair_acc:
            best_pair_acc = acc
            best_pair = (name, w)
print(f"Best pair: realmlp + {best_pair[0]} (w={best_pair[1]:.2f}) = {best_pair_acc:.5f}")

# Try 3-way blends
print(f"\n--- 3-way blends ---")
best_3way_acc = 0
best_3way = None
top_models = sorted(all_scores, key=all_scores.get, reverse=True)[:5]
for i in range(len(top_models)):
    for j in range(i+1, len(top_models)):
        for w1 in np.arange(0.3, 0.8, 0.1):
            for w2 in np.arange(0.1, 0.5, 0.1):
                w3 = 1 - w1 - w2
                if w3 < 0.05:
                    continue
                blend = w1*all_oof[top_models[i]] + w2*all_oof[top_models[j]] + w3*rm_oof
                acc = accuracy_score(y, np.argmax(blend, axis=1))
                if acc > best_3way_acc:
                    best_3way_acc = acc
                    best_3way = (top_models[i], top_models[j], w1, w2, w3)
if best_3way:
    print(f"Best 3-way: {best_3way[0]}({best_3way[2]:.2f}) + {best_3way[1]}({best_3way[3]:.2f}) + realmlp({best_3way[4]:.2f}) = {best_3way_acc:.5f}")

# Stacking with LR
print(f"\n--- LR Stacking ---")
from sklearn.linear_model import LogisticRegression
meta_oof = np.zeros((len(y), 3))
meta_test = np.zeros((len(test_ids), 3))
for fold, (tr_idx, val_idx) in enumerate(skf.split(stack_oof, y), 1):
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    meta.fit(stack_oof[tr_idx], y[tr_idx])
    meta_oof[val_idx] = meta.predict_proba(stack_oof[val_idx])
    meta_test += meta.predict_proba(stack_test) / N_FOLDS
stack_acc = accuracy_score(y, np.argmax(meta_oof, axis=1))
print(f"LR Stacking: {stack_acc:.5f}")

# ============================================================
# 6. Select Best & Submit
# ============================================================
all_strategies = {
    'pair_blend': best_pair_acc,
    'stacking': stack_acc,
}
if best_3way:
    all_strategies['3way_blend'] = best_3way_acc

best_strat = max(all_strategies, key=all_strategies.get)
best_acc = all_strategies[best_strat]
print(f"\n{'='*60}")
print(f"BEST STRATEGY: {best_strat} = {best_acc:.5f}")
print(f"{'='*60}")

os.makedirs("submissions", exist_ok=True)

if best_strat == 'pair_blend':
    w = best_pair[1]
    best_tst = w * rm_test + (1-w) * all_test[best_pair[0]]
    desc = f"RealMLP({w:.2f})+{best_pair[0]}({1-w:.2f})"
elif best_strat == '3way_blend':
    m1, m2, w1, w2, w3 = best_3way
    best_tst = w1*all_test[m1] + w2*all_test[m2] + w3*rm_test
    desc = f"{m1}({w1:.2f})+{m2}({w2:.2f})+rm({w3:.2f})"
else:
    best_tst = meta_test
    desc = "LR_stacking"

le = LabelEncoder().fit(train[TARGET])
labels = le.inverse_transform(np.argmax(best_tst, axis=1))
path = f"submissions/final_{best_acc:.5f}.csv"
pd.DataFrame({'id': test_ids, 'class': labels}).to_csv(path, index=False)
print(f"\nSaved: {path}")
print(f"Description: {desc}")
print(f"\nTarget: 0.97100 | Achieved: {best_acc:.5f}")
if best_acc >= 0.971:
    print(">>> PAST #1! <<<")
elif best_acc >= 0.97:
    print(">>> PAST 0.97! <<<")
