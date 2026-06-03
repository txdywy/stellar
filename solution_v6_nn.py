"""
Kaggle S6E6 - v6: Simplified RealMLP Neural Network
Based on the #3 leaderboard approach. PBLD embeddings for numerical features.
"""
import sys, os
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, TargetEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

print = lambda *a, **k: (sys.stdout.write(' '.join(map(str, a)) + k.get('end', '\n')), sys.stdout.flush())
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 1. Load & Feature Engineering
# ============================================================
print("Loading data...")
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
    df = df.drop(columns=['spectral_type', 'galaxy_population'])
    return df

X_fe = fe(train.drop([ID, TARGET], axis=1))
X_test_fe = fe(test.drop([ID], axis=1))
test_ids = test[ID].values
n_classes = 3

# Median/IQR scaling for numerical features
num_cols = X_fe.columns.tolist()
medians = X_fe[num_cols].median()
iqrs = X_fe[num_cols].quantile(0.75) - X_fe[num_cols].quantile(0.25)
iqrs = iqrs.replace(0, 1)

X_num = ((X_fe[num_cols] - medians) / iqrs).values.astype(np.float32)
X_test_num = ((X_test_fe[num_cols] - medians) / iqrs).values.astype(np.float32)
n_features = X_num.shape[1]

print(f"Features: {n_features}, Samples: {len(X_num)}")

# ============================================================
# 2. Model: Simplified RealMLP with PBLD Embeddings
# ============================================================
class PBLDEmbedding(nn.Module):
    """Periodic Basis with Learned Decay for numerical features."""
    def __init__(self, n_features, hidden_dim=16, out_dim=4, freq_scale=1.0):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_features, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_features, hidden_dim, out_dim - 1) / math.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(n_features, out_dim - 1))
        nn.init.uniform_(self.b1, -math.pi, math.pi)
        self.act = nn.PReLU()

    def forward(self, x):
        # x: (batch, n_features)
        x_exp = x.unsqueeze(-1)  # (batch, n_features, 1)
        periodic = torch.cos(2 * math.pi * (x_exp * self.w1.unsqueeze(0) + self.b1.unsqueeze(0)))
        transformed = self.act(torch.einsum("bfh,fhd->bfd", periodic, self.w2) + self.b2.unsqueeze(0))
        feat = torch.cat([x_exp, transformed], dim=-1)
        return feat.flatten(start_dim=1)  # (batch, n_features * out_dim)


class StellarMLP(nn.Module):
    def __init__(self, n_features, n_classes, pbld_out=4, hidden=512, dropout=0.1):
        super().__init__()
        self.pbld = PBLDEmbedding(n_features, hidden_dim=16, out_dim=pbld_out, freq_scale=1.0)
        input_dim = n_features * pbld_out
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        x = self.pbld(x)
        return self.net(x)


# ============================================================
# 3. Training
# ============================================================
N_FOLDS = 5
N_EPOCHS = 15
BATCH_SIZE = 4096
LR = 0.001
DEVICE = 'cpu'

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof = np.zeros((len(X_num), n_classes))
tst = np.zeros((len(X_test_num), n_classes))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_num, y), 1):
    print(f"\n--- Fold {fold}/{N_FOLDS} ---")

    X_tr = torch.tensor(X_num[tr_idx], device=DEVICE)
    y_tr = torch.tensor(y[tr_idx], device=DEVICE)
    X_val = torch.tensor(X_num[val_idx], device=DEVICE)
    X_tst = torch.tensor(X_test_num, device=DEVICE)

    # Class weights
    class_counts = np.bincount(y[tr_idx], minlength=n_classes)
    class_weights = torch.tensor(len(tr_idx) / (n_classes * class_counts), dtype=torch.float32, device=DEVICE)

    model = StellarMLP(n_features, n_classes, pbld_out=4, hidden=512, dropout=0.1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    best_acc = 0
    best_state = None

    for epoch in range(N_EPOCHS):
        model.train()
        perm = torch.randperm(len(tr_idx))
        total_loss = 0
        for i in range(0, len(tr_idx), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            out = model(X_tr[idx])
            loss = F.cross_entropy(out, y_tr[idx], weight=class_weights, label_smoothing=0.05)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_out = []
            for i in range(0, len(val_idx), BATCH_SIZE):
                val_out.append(model(X_val[i:i+BATCH_SIZE]).softmax(dim=1))
            val_probs = torch.cat(val_out).cpu().numpy()
        val_acc = accuracy_score(y[val_idx], np.argmax(val_probs, axis=1))

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()
            oof[val_idx] = val_probs

        print(f"  Epoch {epoch+1:2d}: loss={total_loss:.3f} val_acc={val_acc:.5f} best={best_acc:.5f}")

    # Predict test with best model
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_out = []
        for i in range(0, len(X_test_num), BATCH_SIZE):
            test_out.append(model(X_tst[i:i+BATCH_SIZE]).softmax(dim=1))
        tst += torch.cat(test_out).cpu().numpy() / N_FOLDS

    print(f"  Fold {fold} best: {best_acc:.5f}")

# ============================================================
# 4. Results
# ============================================================
overall_acc = accuracy_score(y, np.argmax(oof, axis=1))
print(f"\n{'='*50}")
print(f"OOF Accuracy: {overall_acc:.5f}")
print(f"{'='*50}")

os.makedirs("submissions", exist_ok=True)
labels = le.inverse_transform(np.argmax(tst, axis=1))
path = f"submissions/v6_nn_{overall_acc:.5f}.csv"
pd.DataFrame({'id': test_ids, 'class': labels}).to_csv(path, index=False)
print(f"Saved: {path}")
print(f"Target: 0.97000 | Achieved: {overall_acc:.5f}")
