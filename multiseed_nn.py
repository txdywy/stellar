"""Multi-seed RealMLP variant for ensemble diversity"""
import sys, math, random, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.utils.class_weight import compute_class_weight
import torch, torch.nn as nn, torch.nn.functional as F
import warnings; warnings.filterwarnings('ignore')
import os

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 123
print(f"Running with SEED={SEED}", flush=True)
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
device = torch.device('cpu')

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
ID, TARGET = 'id', 'class'
train[TARGET] = train[TARGET].map({'GALAXY':0,'QSO':1,'STAR':2})
X = train.drop([ID, TARGET], axis=1); train_id = train[ID]; y = train[TARGET]
X_test = test.drop([ID], axis=1); test_id = test[ID]

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
category_map = {}
color_pairs = [('u','g'),('u','r')]
important_combos = sorted([('alpha_cat_','delta_cat_'),('u_cat_','z_cat_')])

def fe(df, fit=False):
    df = df.copy()
    df['_g_/_redshift'] = (df['g']/(df['redshift']+1e-6)).astype('float32')
    df['_i_/_redshift'] = (df['i']/(df['redshift']+1e-6)).astype('float32')
    for a,b in color_pairs:
        df[f'_{a}-{b}'] = (df[a]-df[b]).astype('float32')
    for col in cat_cols:
        if fit:
            codes, uniques = df[col].factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            cm = {c:i for i,c in enumerate(uniques)}
            codes = df[col].map(cm).fillna(-1).astype('int32')
        df[col] = codes; df[col] = df[col].astype('category')
    for col in num_cols:
        cn = f'{col}_cat_'
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            cm = {c:i for i,c in enumerate(uniques)}
            codes = np.floor(df[col]).map(cm).fillna(-1).astype('int32')
        df[cn] = codes; df[cn] = df[cn].astype('category')
    for col, bins_l in {'delta':[100,500]}.items():
        for nb in bins_l:
            bn = f'{col}_{nb}_quantile_bin_'
            if fit:
                kb = KBinsDiscretizer(n_bins=nb, encode='ordinal', strategy='quantile', subsample=None)
                binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
                category_map[bn] = kb
            else:
                kb = category_map[bn]
                binned = kb.transform(df[[col]]).ravel().astype('int32')
            df[bn] = binned; df[bn] = df[bn].astype('category')
    for cols in important_combos:
        cn = '_'.join(cols) + '_'
        cs = df[cols[0]].astype(str)
        for c in cols[1:]:
            cs = cs + '_' + df[c].astype(str)
        if fit:
            codes, uniques = pd.factorize(cs, sort=False)
            category_map[cn] = uniques
        else:
            uniques = category_map[cn]
            cm = {c:i for i,c in enumerate(uniques)}
            codes = cs.map(cm).fillna(-1).astype('int32')
        df[cn] = codes; df[cn] = df[cn].astype('category')
    return df

print("Feature engineering...", flush=True)
X = fe(X, fit=True)
X_test = fe(X_test, fit=False)
cat_cols = sorted(cat_cols + [c for c in X.columns if c.endswith('_')])
X = X.reindex(sorted(X.columns), axis=1)
X_test = X_test.reindex(sorted(X_test.columns), axis=1)

nc = [c for c in X.columns if c not in cat_cols]
Xn = X[nc].values.astype(np.float32)
Xtn = X_test[nc].values.astype(np.float32)
med = np.median(Xn, 0)
iqr = np.quantile(Xn, 0.75, 0) - np.quantile(Xn, 0.25, 0)
iqr[iqr == 0] = 1
Xn = (Xn - med) / iqr
Xtn = (Xtn - med) / iqr

Xc = X[cat_cols].values.astype(np.int64)
Xtc = X_test[cat_cols].values.astype(np.int64)
for i in range(Xc.shape[1]):
    Xtc[:, i] = np.clip(Xtc[:, i], 0, Xc[:, i].max())
cat_dims = (Xc.max(0) + 1).tolist()
print(f"Features: num={len(nc)}, cat={len(cat_cols)}", flush=True)

class PBLD(nn.Module):
    def __init__(self, nf, od=4):
        super().__init__()
        self.w = nn.Parameter(torch.randn(nf, 16) * 1.0)
        self.b = nn.Parameter(torch.randn(nf, 16))
        self.w2 = nn.Parameter(torch.randn(nf, 16, od-1) / 4)
        self.b2 = nn.Parameter(torch.zeros(nf, od-1))
        nn.init.uniform_(self.b, -math.pi, math.pi)
    def forward(self, x):
        p = torch.cos(2*math.pi*(x.unsqueeze(-1)*self.w + self.b))
        t = torch.relu(torch.einsum('bfh,fhd->bfd', p, self.w2) + self.b2)
        return torch.cat([x.unsqueeze(-1), t], -1).flatten(1)

class Net(nn.Module):
    def __init__(self, nn_, cd, nc_):
        super().__init__()
        self.pb = PBLD(nn_)
        self.ce = nn.ModuleList([nn.Embedding(d, 8) for d in cd])
        tot = nn_ * 4 + len(cd) * 8
        self.mlp = nn.Sequential(
            nn.Linear(tot, 512), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(512, nc_))
    def forward(self, xn, xc):
        cat = torch.cat([e(xc[:, i]) for i, e in enumerate(self.ce)], 1)
        return self.mlp(torch.cat([self.pb(xn), cat], 1))

N_FOLDS = 5
EPOCHS = 12
BS = 2048
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros((len(Xn), 3))
tst = np.zeros((len(Xtn), 3))
cw = torch.tensor(compute_class_weight('balanced', classes=np.arange(3), y=y.values), dtype=torch.float32)

for fold, (tr, val) in enumerate(skf.split(Xn, y), 1):
    print(f"Fold {fold}/{N_FOLDS}...", end=' ', flush=True)
    m = Net(Xn.shape[1], cat_dims, 3).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    Xtn_ = torch.tensor(Xn[tr], device=device)
    Xtc_ = torch.tensor(Xc[tr], device=device)
    yt_ = torch.tensor(y.values[tr], device=device)
    Xvn_ = torch.tensor(Xn[val], device=device)
    Xvc_ = torch.tensor(Xc[val], device=device)
    best = 0
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(tr), BS):
            idx = perm[i:i+BS]
            out = m(Xtn_[idx], Xtc_[idx])
            loss = F.cross_entropy(out, yt_[idx], weight=cw, label_smoothing=0.05)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            vp = torch.cat([m(Xvn_[i:i+BS], Xvc_[i:i+BS]).softmax(1)
                           for i in range(0, len(val), BS)]).cpu().numpy()
        acc = accuracy_score(y.values[val], np.argmax(vp, 1))
        if acc > best:
            best = acc; oof[val] = vp
    print(f"best={best:.5f}", flush=True)
    m.eval()
    with torch.no_grad():
        tp = torch.cat([m(torch.tensor(Xtn[i:i+BS], device=device),
                         torch.tensor(Xtc[i:i+BS], device=device)).softmax(1)
                       for i in range(0, len(Xtn), BS)]).cpu().numpy()
    tst += tp / N_FOLDS

overall = accuracy_score(y, np.argmax(oof, 1))
print(f"\nOOF Accuracy (seed={SEED}): {overall:.5f}", flush=True)

# Save predictions
oof_df = pd.DataFrame({'id': train_id})
test_df = pd.DataFrame({'id': test_id})
for i, cls in enumerate(['GALAXY','QSO','STAR']):
    oof_df[cls] = oof[:, i]; test_df[cls] = tst[:, i]
oof_df.to_csv(f'oof_seed{SEED}.csv', index=False)
test_df.to_csv(f'test_seed{SEED}.csv', index=False)

# Blend with original RealMLP
rm_oof = pd.read_csv('oof_preds.csv')[['GALAXY','QSO','STAR']].values
rm_test = pd.read_csv('test_preds.csv')[['GALAXY','QSO','STAR']].values
best_w = 0; best_acc = 0
for w in np.arange(0.3, 0.95, 0.01):
    blend = w * rm_oof + (1-w) * oof
    acc = accuracy_score(y, np.argmax(blend, 1))
    if acc > best_acc:
        best_acc = acc; best_w = w
print(f"Blend: rm={best_w:.2f} s{SEED}={1-best_w:.2f} acc={best_acc:.5f}", flush=True)

os.makedirs('submissions', exist_ok=True)
blend_test = best_w * rm_test + (1-best_w) * tst
le = LabelEncoder().fit(['GALAXY','QSO','STAR'])
labels = le.inverse_transform(np.argmax(blend_test, 1))
path = f'submissions/multiseed_{best_acc:.5f}.csv'
pd.DataFrame({'id': test_id, 'class': labels}).to_csv(path, index=False)
print(f"Saved: {path}", flush=True)
