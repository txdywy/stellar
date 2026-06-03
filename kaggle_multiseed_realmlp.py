"""
Multi-seed RealMLP Ensemble for Kaggle
Runs the RealMLP architecture with 3 different seeds and blends predictions.
Each seed produces slightly different models due to random initialization.
"""
import math
import random
import warnings
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Load Data
# ============================================================
train = pd.read_csv("/kaggle/input/competitions/playground-series-s6e6/train.csv")
test = pd.read_csv("/kaggle/input/competitions/playground-series-s6e6/test.csv")
print(f"Train: {train.shape}, Test: {test.shape}")

ID = 'id'
TARGET = 'class'
train[TARGET] = train[TARGET].map({'GALAXY': 0, 'QSO': 1, 'STAR': 2})
train_id = train[ID]
test_id = test[ID]
y = train[TARGET]
n_classes = y.nunique()

# ============================================================
# Feature Engineering (same as original)
# ============================================================
category_map = {}
color_pairs = [('u', 'g'), ('u', 'r')]
important_combos = sorted([('alpha_cat_', 'delta_cat_'), ('u_cat_', 'z_cat_')])

def feature_engineering(df, fit=False, cat_cols_ref=None, num_cols_ref=None):
    df = df.copy()
    df['_g_/_redshift'] = (df['g'] / (df['redshift'] + 1e-6)).astype('float32')
    df['_i_/_redshift'] = (df['i'] / (df['redshift'] + 1e-6)).astype('float32')
    for a, b in color_pairs:
        df[f"_{a}-{b}"] = (df[a] - df[b]).astype('float32')

    if fit:
        cat_cols_ref = df.select_dtypes(include=['object']).columns.tolist()
        num_cols_ref = df.select_dtypes(exclude=['object']).columns.tolist()

    for col in cat_cols_ref:
        if fit:
            codes, uniques = df[col].factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = df[col].map(code_map).fillna(-1).astype('int32')
        df[col] = codes
        df[col] = df[col].astype('category')

    for col in num_cols_ref:
        cat_name = f"{col}_cat_"
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype('int32')
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype('category')

    for col, bins_list in {'delta': [100, 500]}.items():
        for n_bins in bins_list:
            bin_name = f"{col}_{n_bins}_quantile_bin_"
            if fit:
                kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)
                binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
                category_map[bin_name] = kb
            else:
                kb = category_map[bin_name]
                binned = kb.transform(df[[col]]).ravel().astype('int32')
            df[bin_name] = binned
            df[bin_name] = df[bin_name].astype('category')

    combo_names = []
    for cols in important_combos:
        combo_name = '_'.join(cols) + '_'
        combo_names.append(combo_name)
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + '_' + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            category_map[combo_name] = uniques
        else:
            uniques = category_map[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype('int32')
        df[combo_name] = codes
        df[combo_name] = df[combo_name].astype('category')

    new_cat_cols = [col for col in df.columns if col.endswith('_')]
    new_num_cols = [col for col in df.columns if col.startswith('_')]
    return df, cat_cols_ref, num_cols_ref, new_cat_cols, new_num_cols, combo_names

# Feature engineering
X_raw = train.drop([ID, TARGET], axis=1)
X_test_raw = test.drop([ID], axis=1)

X, cat_cols, num_cols, new_cat, new_num, combo_names = feature_engineering(X_raw, fit=True)
X_test, _, _, _, _, _ = feature_engineering(X_test_raw, fit=False, cat_cols_ref=cat_cols, num_cols_ref=num_cols)

cat_cols += new_cat; num_cols += new_num
cat_cols = sorted(cat_cols)
X = X.reindex(sorted(X.columns), axis=1)
X_test = X_test.reindex(sorted(X_test.columns), axis=1)
del train, test, X_raw, X_test_raw

print(f"Features: {X.shape[1]}, Cat: {len(cat_cols)}, Num: {len(num_cols)}")

# ============================================================
# Model Components (same as original)
# ============================================================
class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, tfms):
        self._tfms = [t for t in tfms if t in ("median_center", "robust_scale", "smooth_clip", "l2_normalize")]
    def fit(self, X, y=None):
        if "median_center" in self._tfms or "robust_scale" in self._tfms:
            self._median = np.median(X, axis=0)
            q_diff = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
            zero_idx = q_diff == 0.0
            q_diff[zero_idx] = 0.5 * (X.max(axis=0)[zero_idx] - X.min(axis=0)[zero_idx])
            self._iqr_factors = 1.0 / (q_diff + 1e-30)
            self._iqr_factors[q_diff == 0.0] = 0.0
        return self
    def transform(self, X, y=None):
        X = X.copy().astype(np.float32)
        for tfm in self._tfms:
            if tfm == "median_center": X -= self._median[None, :]
            elif tfm == "robust_scale": X *= self._iqr_factors[None, :]
            elif tfm == "smooth_clip": X = X / np.sqrt(1 + (X / 3) ** 2)
            elif tfm == "l2_normalize":
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                X /= np.where(norms == 0, 1.0, norms)
        return X

class CategoricalFeatureLayer(nn.Module):
    def __init__(self, n_ens, cat_dims, embed_dim=8, onehot_thresh=8):
        super().__init__()
        self.n_ens = n_ens; self.cat_dims = cat_dims
        self.onehot_features = []; self.embed_layers = nn.ModuleList(); self._embed_feature_indices = []
        for i, dim in enumerate(cat_dims):
            if dim <= onehot_thresh: self.onehot_features.append(i)
            else:
                self.embed_layers.append(nn.ModuleList([nn.Embedding(dim, embed_dim) for _ in range(n_ens)]))
                self._embed_feature_indices.append(i)
    def forward(self, x):
        batch_size, n_ens, _ = x.shape; features = []
        if self.onehot_features:
            onehot_x = x[:, :, self.onehot_features]
            onehot_dims = [self.cat_dims[i] for i in self.onehot_features]
            total_oh = sum(onehot_dims)
            encoded = torch.zeros(batch_size, n_ens, total_oh, device=x.device)
            start = 0
            for idx, dim in enumerate(onehot_dims):
                pos = onehot_x[:, :, idx:idx+1].long()
                encoded.scatter_(2, pos + start, 1.0); start += dim
            features.append(encoded)
        for emb_list, feat_idx in zip(self.embed_layers, self._embed_feature_indices):
            feat_embs = []
            for model_idx in range(self.n_ens):
                indices = x[:, model_idx, feat_idx:feat_idx+1].long()
                feat_embs.append(emb_list[model_idx](indices))
            features.append(torch.cat(feat_embs, dim=1))
        return torch.cat(features, dim=2)

class ScalingLayer(nn.Module):
    def __init__(self, n_ens, n_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))
    def forward(self, x): return x * self.scale[None, :, :]

class NTPLinear(nn.Module):
    def __init__(self, n_ens, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features; self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None
    def forward(self, x):
        x = torch.einsum("bki,kio->bko", x, self.weight) / math.sqrt(self.in_features)
        if self.bias is not None: x = x + self.bias
        return x

class PBLDEmbedding(nn.Module):
    def __init__(self, n_ens, n_features, hidden_dim=16, out_dim=4, freq_scale=0.1, activation=nn.GELU):
        super().__init__()
        self.n_ens = n_ens; self.n_features = n_features; self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) / math.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(n_ens, n_features, out_dim - 1))
        self.act = activation()
        nn.init.uniform_(self.b1, -math.pi, math.pi)
    def forward(self, x):
        periodic = torch.cos(2 * math.pi * (x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0)))
        transformed = self.act(torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2) + self.b2.unsqueeze(0))
        feat = torch.cat([x.unsqueeze(-1), transformed], dim=-1)
        return feat.flatten(start_dim=2)

class RealMLP(nn.Module):
    def __init__(self, output_dim, cat_dims, n_numerical, cfg):
        super().__init__()
        n_ens = cfg["n_ens"]; embed_dim = cfg["embed_dim"]; self.n_ens = n_ens
        self.cate = CategoricalFeatureLayer(n_ens=n_ens, cat_dims=cat_dims, embed_dim=embed_dim, onehot_thresh=cfg["onehot_thresh"])
        self.num_embed = PBLDEmbedding(n_ens=n_ens, n_features=n_numerical, hidden_dim=cfg["pbld_hidden_dim"], out_dim=cfg["pbld_out_dim"], freq_scale=cfg["pbld_freq_scale"], activation=cfg["pbld_activation"])
        num_emb_dim = n_numerical * cfg["pbld_out_dim"]
        cat_emb_dim = sum(c if c <= cfg["onehot_thresh"] else embed_dim for c in cat_dims)
        total_dim = num_emb_dim + cat_emb_dim
        hidden_dims = cfg["hidden_dims"]; act = cfg["activation"]
        layers = []
        if cfg["add_front_scale"]: layers.append(ScalingLayer(n_ens=n_ens, n_features=total_dim))
        self._dropout_modules = []
        in_dim = total_dim
        for i, out_dim_h in enumerate(hidden_dims):
            linear = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=out_dim_h)
            if i == 0: self.first_linear = linear
            drop = nn.Dropout(cfg["dropout"])
            self._dropout_modules.append(drop)
            layers += [linear, act(), drop]; in_dim = out_dim_h
        self.hidden = nn.Sequential(*layers)
        self.output_layer = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=output_dim)
    def forward(self, x_num, x_cat):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_num = self.num_embed(x_num); x_cat = self.cate(x_cat)
        combined = torch.cat([x_num, x_cat], dim=2)
        x = self.hidden(combined); x = self.output_layer(x)
        return F.softmax(x, dim=2)

def apply_schedule(init_value, progress, sched, flat_ratio=0.3):
    if sched == "constant": return init_value
    elif sched == "cos": return init_value * (math.cos(math.pi * progress) + 1) / 2
    elif sched == "flat_cos":
        if progress < flat_ratio: return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (math.cos(math.pi * t) + 1) / 2
    elif sched == "expm4t": return init_value * math.exp(-4 * progress)
    return init_value

def get_parameter_groups(model, p):
    first_linear_weight_id = id(model.first_linear.weight)
    scale_p, pbld_p, first_w_p, other_w_p, bias_p = [], [], [], [], []
    for name, param in model.named_parameters():
        if "num_embed" in name: pbld_p.append(param)
        elif "scale" in name: scale_p.append(param)
        elif id(param) == first_linear_weight_id: first_w_p.append(param)
        elif "bias" in name: bias_p.append(param)
        else: other_w_p.append(param)
    LR = p["lr"]; WD = p["weight_decay"]
    return [
        {"params": scale_p, "lr": LR * p["lr_scale_mult"], "weight_decay": WD * p["wd_scale_mult"]},
        {"params": pbld_p, "lr": LR * p["pbld_lr_factor"], "weight_decay": WD},
        {"params": first_w_p, "lr": LR * p["first_layer_lr_factor"], "weight_decay": WD * p["first_layer_wd_factor"]},
        {"params": other_w_p, "lr": LR, "weight_decay": WD},
        {"params": bias_p, "lr": LR * p["lr_bias_mult"], "weight_decay": WD * p["wd_bias_mult"]},
    ]

def smooth_ce_loss(y_true, y_pred, ls=0.0, class_weights=None):
    n_classes = y_pred.size(1)
    y_smooth = torch.full_like(y_pred, ls / n_classes)
    y_smooth.scatter_(1, y_true.unsqueeze(1), 1.0 - ls + ls / n_classes)
    per_sample_loss = -(y_smooth * torch.log(y_pred.clamp(1e-15, 1))).sum(dim=1)
    if class_weights is not None:
        sample_weights = class_weights[y_true]
        return (per_sample_loss * sample_weights).sum() / sample_weights.sum()
    return per_sample_loss.mean()

class RealMLP_TD_Classifier(BaseEstimator):
    def __init__(self, **kwargs):
        self.params = {**CONFIG, **kwargs}
    def fit(self, X_train, y_train, X_val, y_val, cat_col_names=None, ckpt_path="ckpt.pth", X_test=None):
        p = self.params
        dev = torch.device(p["device"] if torch.cuda.is_available() else "cpu")
        verbose = p["verbosity"]
        cat_col_names = cat_col_names or []
        num_col_names = [c for c in X_train.columns if c not in cat_col_names]
        X_tr_num = X_train[num_col_names].values.astype(np.float32)
        X_val_num = X_val[num_col_names].values.astype(np.float32)
        X_tr_cat = X_train[cat_col_names].values.astype(np.int64)
        X_val_cat = X_val[cat_col_names].values.astype(np.int64)
        y_tr = np.asarray(y_train); y_v = np.asarray(y_val)
        self.preprocessor_ = NumericalPreprocessor(p["tfms"])
        self.preprocessor_.fit(X_tr_num)
        X_tr_num = self.preprocessor_.transform(X_tr_num)
        X_val_num = self.preprocessor_.transform(X_val_num)
        self.cat_col_names_ = cat_col_names; self.num_col_names_ = num_col_names
        if cat_col_names:
            all_cat = [X_tr_cat, X_val_cat]
            if X_test is not None: all_cat.append(X_test[cat_col_names].values.astype(np.int64))
            cat_dims = (np.concatenate(all_cat, axis=0).max(axis=0) + 1).tolist()
        else: cat_dims = []
        self.cat_dims_ = cat_dims
        if cat_dims:
            cat_max = np.array(cat_dims) - 1
            X_tr_cat = np.clip(X_tr_cat, 0, cat_max); X_val_cat = np.clip(X_val_cat, 0, cat_max)
        classes = np.unique(y_tr); self.classes_ = classes
        weights_np = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
        class_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=dev)
        n_classes = len(classes)
        self.model_ = RealMLP(output_dim=n_classes, cat_dims=cat_dims, n_numerical=X_tr_num.shape[1], cfg=p).to(dev)
        param_groups = get_parameter_groups(self.model_, p)
        for g in param_groups: g["lr_base"] = g["lr"]
        optimizer = torch.optim.AdamW(param_groups, betas=(p["mom"], p["sq_mom"]))
        Xtn = torch.as_tensor(X_tr_num, dtype=torch.float32, device=dev)
        Xtc = torch.as_tensor(X_tr_cat, dtype=torch.long, device=dev)
        ytt = torch.as_tensor(y_tr, dtype=torch.long, device=dev)
        Xvn = torch.as_tensor(X_val_num, dtype=torch.float32, device=dev)
        Xvc = torch.as_tensor(X_val_cat, dtype=torch.long, device=dev)
        n_ens = p["n_ens"]; train_bs = p["train_bs"]; eval_bs = p["eval_bs"]
        epochs = p["epochs"]; lr_sched = p["lr_sched"]; flat_ratio = p["flat_ratio"]
        total_steps = epochs * len(y_tr); train_order = np.arange(len(y_tr))
        best_score = -np.inf; best_epoch = 0; best_val_probs = None
        for epoch in range(epochs):
            self.model_.train()
            for start in range(0, len(y_tr), train_bs):
                progress = (epoch * len(y_tr) + start) / total_steps
                idx_batch = train_order[start:start+train_bs]
                for g in optimizer.param_groups: g["lr"] = apply_schedule(g["lr_base"], progress, lr_sched, flat_ratio)
                optimizer.zero_grad()
                y_pred = self.model_(Xtn[idx_batch], Xtc[idx_batch])
                ls_val = apply_schedule(p["ls_eps"], progress, p["ls_eps_sched"], flat_ratio)
                drop_val = apply_schedule(p["dropout"], progress, p["p_drop_sched"], flat_ratio)
                for dm in self.model_._dropout_modules: dm.p = drop_val
                loss = smooth_ce_loss(ytt[idx_batch].repeat_interleave(n_ens), y_pred.reshape(-1, n_classes), ls=ls_val, class_weights=class_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), p["grad_clip"])
                optimizer.step()
            np.random.shuffle(train_order)
            self.model_.eval()
            with torch.no_grad():
                val_probs = np.concatenate([self.model_(Xvn[s:s+eval_bs], Xvc[s:s+eval_bs]).mean(dim=1).cpu().numpy() for s in range(0, len(y_v), eval_bs)], axis=0)
            epoch_score = balanced_accuracy_score(y_v, np.argmax(val_probs, axis=1))
            if epoch_score > best_score:
                best_score = epoch_score; best_epoch = epoch + 1; best_val_probs = val_probs.copy()
                torch.save(self.model_.state_dict(), ckpt_path)
            if verbose >= 2: print(f"  epoch {epoch+1}/{epochs} score={epoch_score:.5f} best={best_score:.5f}" + (" ✓" if epoch_score == best_score else ""))
        self.model_.load_state_dict(torch.load(ckpt_path))
        self.best_score_ = best_score; self.best_val_probs_ = best_val_probs; self._dev = dev
        if verbose >= 1: print(f"  → best: {best_score:.5f} (epoch {best_epoch})")
        return self
    def predict_proba(self, X):
        eval_bs = self.params["eval_bs"]
        X_num = self.preprocessor_.transform(X[self.num_col_names_].values.astype(np.float32))
        X_cat = X[self.cat_col_names_].values.astype(np.int64)
        X_cat = np.clip(X_cat, 0, np.array(self.cat_dims_) - 1)
        Xn = torch.as_tensor(X_num, dtype=torch.float32, device=self._dev)
        Xc = torch.as_tensor(X_cat, dtype=torch.long, device=self._dev)
        self.model_.eval()
        with torch.no_grad():
            return np.concatenate([self.model_(Xn[s:s+eval_bs], Xc[s:s+eval_bs]).mean(dim=1).cpu().numpy() for s in range(0, len(X_num), eval_bs)], axis=0)

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "n_ens": 8, "embed_dim": 8, "onehot_thresh": 8,
    "hidden_dims": [512, 512, 512], "dropout": 0.06, "p_drop_sched": "expm4t",
    "activation": nn.SiLU, "add_front_scale": True,
    "pbld_hidden_dim": 20, "pbld_out_dim": 5, "pbld_freq_scale": 5.0,
    "pbld_activation": nn.PReLU, "pbld_lr_factor": 0.093,
    "lr": 0.01, "mom": 0.9, "sq_mom": 0.98,
    "lr_sched": "flat_cos", "flat_ratio": 0.3,
    "first_layer_lr_factor": 1.0, "first_layer_wd_factor": 0.1,
    "lr_scale_mult": 10.0, "lr_bias_mult": 0.1,
    "weight_decay": 0.013, "wd_scale_mult": 0.1, "wd_bias_mult": 0.5, "grad_clip": 1.0,
    "ls_eps": 0.04, "ls_eps_sched": "cos",
    "tfms": ["median_center", "robust_scale"],
    "epochs": 6, "train_bs": 256, "eval_bs": 10240, "verbosity": 2,
    "use_early_stopping": False,
    "early_stopping_additive_patience": 10, "early_stopping_multiplicative_patience": 1,
    "device": "cuda", "random_state": 42,
}

# ============================================================
# Train with multiple seeds
# ============================================================
SEEDS = [42, 123, 777]  # 3 different seeds for diversity
FOLDS = 5
TE = True

all_test_preds = []
all_oof_preds = []

for seed_idx, SEED in enumerate(SEEDS):
    print(f"\n{'='*60}")
    print(f"  TRAINING WITH SEED={SEED} ({seed_idx+1}/{len(SEEDS)})")
    print(f"{'='*60}\n")

    # Reset random state for this seed
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    CONFIG["random_state"] = SEED

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros((len(X), n_classes))
    test_preds = np.zeros((len(X_test), n_classes))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr = X.iloc[tr_idx].copy(); X_val = X.iloc[val_idx].copy(); X_tst = X_test.copy()

        if TE:
            te_cols = combo_names
            encoder = TargetEncoder(cv=FOLDS, smooth='auto', shuffle=True, random_state=SEED)
            tr_enc = encoder.fit_transform(X_tr[te_cols], y.iloc[tr_idx])
            val_enc = encoder.transform(X_val[te_cols])
            tst_enc = encoder.transform(X_tst[te_cols])
            te_names = [f"_{col}TE_class{cls}" for col in te_cols for cls in range(n_classes)]
            X_tr[te_names] = tr_enc; X_val[te_names] = val_enc; X_tst[te_names] = tst_enc

        print(f"### Seed {SEED} | Fold {fold}/{FOLDS}")
        model = RealMLP_TD_Classifier(**CONFIG)
        model.fit(X_tr, y.iloc[tr_idx], X_val, y.iloc[val_idx],
                  cat_col_names=cat_cols, ckpt_path=f"model_s{SEED}_f{fold}.pth")
        oof_preds[val_idx] = model.best_val_probs_
        test_preds += model.predict_proba(X_tst) / FOLDS
        torch.cuda.empty_cache()

    oof_acc = accuracy_score(y, np.argmax(oof_preds, axis=1))
    oof_ba = balanced_accuracy_score(y, np.argmax(oof_preds, axis=1))
    print(f"\nSeed {SEED} OOF: Acc={oof_acc:.5f} BA={oof_ba:.5f}\n")

    all_test_preds.append(test_preds)
    all_oof_preds.append(oof_preds)

# ============================================================
# Blend all seeds
# ============================================================
print(f"\n{'='*60}")
print("BLENDING MULTIPLE SEEDS")
print(f"{'='*60}")

# Simple average of all seeds
avg_oof = sum(all_oof_preds) / len(all_oof_preds)
avg_test = sum(all_test_preds) / len(all_test_preds)
avg_acc = accuracy_score(y, np.argmax(avg_oof, axis=1))
print(f"Average {len(SEEDS)} seeds: Acc={avg_acc:.5f}")

# Try weighted average
best_acc = 0; best_weights = None
from itertools import product
for w1 in np.arange(0.2, 0.6, 0.05):
    for w2 in np.arange(0.1, 0.5, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.1: continue
        blend = w1 * all_oof_preds[0] + w2 * all_oof_preds[1] + w3 * all_oof_preds[2]
        acc = accuracy_score(y, np.argmax(blend, axis=1))
        if acc > best_acc:
            best_acc = acc; best_weights = (w1, w2, w3)

print(f"Best weighted: {best_weights} Acc={best_acc:.5f}")

# Generate submission with best blend
best_blend_test = best_weights[0]*all_test_preds[0] + best_weights[1]*all_test_preds[1] + best_weights[2]*all_test_preds[2]
sub = pd.DataFrame({ID: test_id, TARGET: np.argmax(best_blend_test, axis=1)})
sub[TARGET] = sub[TARGET].map({0: 'GALAXY', 1: 'QSO', 2: 'STAR'})
sub.to_csv('submission.csv', index=False)

print(f"\nSaved submission.csv")
print(f"Best blend accuracy: {best_acc:.5f}")
print(f"Target: 0.97100")
