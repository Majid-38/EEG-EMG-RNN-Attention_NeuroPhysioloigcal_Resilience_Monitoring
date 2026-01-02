# ============================================================
# FULL PIPELINE (End-to-End) using df_rec 
# Subject-wise K-Fold (EEG / EMG / EEG+EMG) with/without Attention
# ============================================================

import os, json, random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------
# SOURCE PATHS 
# -------------------------
DATASET_ROOT = "/content/drive/MyDrive/dataset_resilience"
LABELS_CSV   = "/content/drive/MyDrive/dataset_resilience/ihri/labels_fixed_scaled.csv"

# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

# -------------------------
# Config
# -------------------------
@dataclass
class CFG:
    k_folds: int = 5
    val_ratio_within_train: float = 0.33

    num_epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    min_delta: float = 1e-4

    fs: int = 250
    seconds: int = 20
    stride_seconds: int = 20
    zscore_per_channel: bool = True

    hidden: int = 128
    layers: int = 2
    dropout: float = 0.2

    num_workers: int = 2
    pin_memory: bool = True
    use_amp: bool = True

    out_dir: str = "/content/drive/MyDrive/dataset_resilience/results_kfold"
    run_name: str = "eeg9_emg1_rnn_fullsuite"

    cache_windows_to_disk: bool = True
    cache_dir: Optional[str] = None

    save_subject_predictions: bool = True

cfg = CFG()
if cfg.cache_dir is None:
    cfg.cache_dir = os.path.join(cfg.out_dir, f"cache_{cfg.run_name}_fs{cfg.fs}_win{cfg.seconds}s_stride{cfg.stride_seconds}s")

# -------------------------
# Utilities
# -------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def zscore(x: np.ndarray, axis=0, eps=1e-8) -> np.ndarray:
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x - mu) / (sd + eps)

def sanity_check_emg(emg: np.ndarray, name: str = "EMG"):
    if emg.ndim != 1:
        raise ValueError(f"{name} expected 1D, got shape {emg.shape}")
    if not np.isfinite(emg).all():
        raise ValueError(f"{name} contains NaN/Inf")
    if np.std(emg) < 1e-10:
        raise ValueError(f"{name} looks constant/flat (std~0). Wrong EMG column?")

def make_windows(x: np.ndarray, win_len: int, stride: int) -> np.ndarray:
    if x.ndim == 1:
        x = x[:, None]
    T, C = x.shape
    if T < win_len:
        pad = win_len - T
        x = np.pad(x, ((0, pad), (0, 0)), mode="edge")
        T = x.shape[0]
    starts = list(range(0, T - win_len + 1, stride))
    if len(starts) == 0:
        starts = [0]
    return np.stack([x[s:s+win_len, :] for s in starts], axis=0)

def slugify_path(p: str) -> str:
    return p.replace("/", "_").replace("\\", "_").replace(":", "_")

# -------------------------
# Load EEG+EMG from paths
# -------------------------
def load_eeg_emg_from_paths(eeg_path: str, exg_path: str, eeg_cols: List[str], emg_col: str):
    eeg_df = safe_read_csv(eeg_path)
    exg_df = safe_read_csv(exg_path)

    missing_eeg = [c for c in eeg_cols if c not in eeg_df.columns]
    if missing_eeg:
        raise ValueError(f"Missing EEG cols in {eeg_path}: {missing_eeg}")

    eeg = eeg_df[eeg_cols].to_numpy(dtype=np.float32)

    if emg_col not in exg_df.columns:
        cand = [c for c in exg_df.columns if c.lower() == emg_col.lower()]
        if len(cand) == 1:
            emg_col = cand[0]
        else:
            raise ValueError(f"EMG col '{emg_col}' not found in {exg_path}. Available (first 50): {list(exg_df.columns)[:50]}")

    emg = exg_df[emg_col].to_numpy(dtype=np.float32)
    sanity_check_emg(emg, name=f"EMG({emg_col})")

    T = min(len(emg), eeg.shape[0])
    return eeg[:T, :], emg[:T]

# -------------------------
# Disk cache windows
# -------------------------
def cache_windows_npz(row: pd.Series, modality: str) -> str:
    ensure_dir(cfg.cache_dir)
    win_len = cfg.fs * cfg.seconds
    stride = cfg.fs * cfg.stride_seconds

    key = (
        f"{slugify_path(row.eeg_path)}__{slugify_path(row.exg_path)}__"
        f"{modality}__fs{cfg.fs}__win{cfg.seconds}s__stride{cfg.stride_seconds}s__z{int(cfg.zscore_per_channel)}.npz"
    )
    out_path = os.path.join(cfg.cache_dir, key)
    if os.path.exists(out_path):
        return out_path

    eeg, emg = load_eeg_emg_from_paths(row.eeg_path, row.exg_path, row.eeg_cols, row.emg_col)

    if cfg.zscore_per_channel:
        eeg = zscore(eeg, axis=0)
        emg = zscore(emg, axis=0)

    if modality == "EEG":
        sig = eeg
    elif modality == "EMG":
        sig = emg
    elif modality == "EEG+EMG":
        sig = np.concatenate([eeg, emg[:, None]], axis=1)
    else:
        raise ValueError("Unknown modality")

    X = make_windows(sig, win_len, stride).astype(np.float32)
    y = np.array([int(row.label)], dtype=np.int64)
    subject = str(row.subject)

    np.savez_compressed(out_path, X=X, y=y, subject=subject)
    return out_path

# -------------------------
# Dataset
# -------------------------
class WindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, modality: str):
        self.df = df.reset_index(drop=True)
        self.modality = modality
        self.index = []  # (row_i, win_i, npz_path)

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            npz = cache_windows_npz(row, modality)
            with np.load(npz, allow_pickle=True) as z:
                n = z["X"].shape[0]
            for w in range(n):
                self.index.append((i, w, npz))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, w, npz = self.index[idx]
        row = self.df.iloc[i]
        with np.load(npz, allow_pickle=True) as z:
            X = z["X"][w]
        y = float(row.label)
        s = str(row.subject)
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32), s

def collate_fn(batch):
    xs, ys, subs = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), list(subs)

# -------------------------
# Models
# -------------------------
class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden, layers, dropout, use_attention):
        super().__init__()
        self.use_attention = use_attention
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True
        )
        rnn_out = hidden * 2

        if use_attention:
            self.attn = nn.Sequential(
                nn.Linear(rnn_out, rnn_out),
                nn.Tanh(),
                nn.Linear(rnn_out, 1)
            )

        self.head = nn.Sequential(
            nn.Linear(rnn_out, rnn_out // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_out // 2, 1)
        )

    def forward(self, x):
        h, _ = self.rnn(x)
        if self.use_attention:
            a = self.attn(h)
            w = torch.softmax(a, dim=1)
            pooled = (h * w).sum(dim=1)
        else:
            pooled = h[:, -1, :]
        return self.head(pooled).squeeze(-1)

# -------------------------
# Subject-level eval
# -------------------------
@torch.no_grad()
def predict_probs_by_subject(model, loader, device):
    model.eval()
    subj_probs = {}
    subj_labels = {}

    for x, y, subs in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.amp.autocast(device_type="cuda", enabled=(cfg.use_amp and device.type == "cuda")):
            logits = model(x)
            probs = torch.sigmoid(logits)

        probs = probs.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        for p, yy, s in zip(probs, y_np, subs):
            subj_probs.setdefault(s, []).append(float(p))
            subj_labels.setdefault(s, []).append(int(yy))

    subjects = sorted(subj_probs.keys())
    y_true = np.array([int(round(np.mean(subj_labels[s]))) for s in subjects], dtype=int)
    y_score = np.array([float(np.mean(subj_probs[s])) for s in subjects], dtype=float)
    return subjects, y_true, y_score

def find_best_threshold(y_true, y_score):
    best_thr, best_f1 = 0.5, -1
    for thr in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (y_score >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(thr)
    return best_thr, best_f1

def metrics_at_threshold(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    out = {
        "subj_f1": f1_score(y_true, y_pred, zero_division=0),
        "subj_precision": precision_score(y_true, y_pred, zero_division=0),
        "subj_recall": recall_score(y_true, y_pred, zero_division=0),
    }
    try:
        out["subj_auc"] = roc_auc_score(y_true, y_score)
    except Exception:
        out["subj_auc"] = np.nan
    return out

# -------------------------
# Train one variant
# -------------------------
def train_one_variant(train_df, val_df, test_df, modality, use_attention, fold_name, variant_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_tr = WindowDataset(train_df, modality)
    ds_va = WindowDataset(val_df, modality)
    ds_te = WindowDataset(test_df, modality)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                       pin_memory=cfg.pin_memory, collate_fn=collate_fn)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                       pin_memory=cfg.pin_memory, collate_fn=collate_fn)
    dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                       pin_memory=cfg.pin_memory, collate_fn=collate_fn)

    x0, y0, _ = next(iter(dl_tr))
    input_dim = x0.shape[-1]

    model = RNNClassifier(input_dim, cfg.hidden, cfg.layers, cfg.dropout, use_attention).to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.type == "cuda"))

    best_f1, best_thr, best_state = -1, 0.5, None
    no_imp = 0

    for ep in range(1, cfg.num_epochs + 1):
        model.train()
        losses = []

        for x, y, _ in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=(cfg.use_amp and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))

        _, yv_true, yv_score = predict_probs_by_subject(model, dl_va, device)
        thr, f1 = find_best_threshold(yv_true, yv_score)

        print(f"[{fold_name}|{modality}|{'RNN+ATT' if use_attention else 'RNN'}] "
              f"ep {ep:02d}/{cfg.num_epochs} | loss={np.mean(losses):.4f} | VAL subjF1={f1:.3f} | thr={thr:.2f}")

        if f1 > best_f1 + cfg.min_delta:
            best_f1, best_thr = f1, thr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= cfg.patience:
                print(f"[{fold_name}|{modality}|{'RNN+ATT' if use_attention else 'RNN'}] Early stop. "
                      f"Best VAL subjF1={best_f1:.3f} | thr={best_thr:.2f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_subs, yv_true, yv_score = predict_probs_by_subject(model, dl_va, device)
    val_m = metrics_at_threshold(yv_true, yv_score, best_thr)

    test_subs, yt_true, yt_score = predict_probs_by_subject(model, dl_te, device)
    test_m = metrics_at_threshold(yt_true, yt_score, best_thr)

    pred_path = ""
    if cfg.save_subject_predictions:
        ensure_dir(cfg.out_dir)
        pred_df = pd.DataFrame({
            "fold": fold_name,
            "variant": variant_name,
            "subject": test_subs,
            "y_true": yt_true.astype(int),
            "y_score": yt_score.astype(float),
            "thr": best_thr,
            "y_pred": (yt_score >= best_thr).astype(int),
        })
        pred_path = os.path.join(cfg.out_dir, f"{cfg.run_name}_{fold_name}_{variant_name}_test_subject_preds.csv")
        pred_df.to_csv(pred_path, index=False)

    return {
        "fold": fold_name,
        "modality": modality,
        "attention": int(use_attention),
        "variant": variant_name,
        "input_dim": int(input_dim),

        "best_val_thr": float(best_thr),
        "best_val_subjF1": float(best_f1),

        "val_subjF1": float(val_m["subj_f1"]),
        "val_precision": float(val_m["subj_precision"]),
        "val_recall": float(val_m["subj_recall"]),
        "val_auc": float(val_m["subj_auc"]) if np.isfinite(val_m["subj_auc"]) else np.nan,

        "test_subjF1": float(test_m["subj_f1"]),
        "test_precision": float(test_m["subj_precision"]),
        "test_recall": float(test_m["subj_recall"]),
        "test_auc": float(test_m["subj_auc"]) if np.isfinite(test_m["subj_auc"]) else np.nan,

        "n_train_subj": int(train_df.subject.nunique()),
        "n_val_subj": int(val_df.subject.nunique()),
        "n_test_subj": int(test_df.subject.nunique()),

        "n_train_rec": int(len(train_df)),
        "n_val_rec": int(len(val_df)),
        "n_test_rec": int(len(test_df)),

        "test_subject_preds_csv": pred_path,
        "error": ""
    }

# -------------------------
# Run KFold
# -------------------------
def run_kfold(paired_df: pd.DataFrame):
    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.cache_dir)

    required = ["subject", "task", "eeg_path", "exg_path", "eeg_cols", "emg_col", "label"]
    miss = [c for c in required if c not in paired_df.columns]
    if miss:
        raise ValueError(f"paired_df missing columns: {miss}")

    # ensure eeg_cols list
    if isinstance(paired_df.iloc[0].eeg_cols, str):
        paired_df = paired_df.copy()
        paired_df["eeg_cols"] = paired_df["eeg_cols"].apply(
            lambda s: json.loads(s) if s.strip().startswith("[") else [x.strip() for x in s.split(",")]
        )

    subjects = np.array(sorted(paired_df["subject"].unique()))
    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=42)

    variants = [
        ("EEG", False, "EEG_RNN"),
        ("EEG", True,  "EEG_RNN_ATT"),
        ("EMG", False, "EMG_RNN"),
        ("EMG", True,  "EMG_RNN_ATT"),
        ("EEG+EMG", False, "FUSION_RNN"),
        ("EEG+EMG", True,  "FUSION_RNN_ATT"),
    ]

    all_rows = []

    for fold_i, (tr_idx, te_idx) in enumerate(kf.split(subjects), start=1):
        test_subs = subjects[te_idx]
        trainval_subs = subjects[tr_idx].copy()

        rng = np.random.RandomState(1000 + fold_i)
        rng.shuffle(trainval_subs)
        n_val = max(1, int(cfg.val_ratio_within_train * len(trainval_subs)))

        val_subs = trainval_subs[:n_val]
        train_subs = trainval_subs[n_val:]

        train_df = paired_df[paired_df.subject.isin(train_subs)].reset_index(drop=True)
        val_df   = paired_df[paired_df.subject.isin(val_subs)].reset_index(drop=True)
        test_df  = paired_df[paired_df.subject.isin(test_subs)].reset_index(drop=True)

        fold_name = f"K{cfg.k_folds}_fold{fold_i:02d}"
        print("\n" + "#" * 110)
        print(f"RUN: {fold_name}")
        print(f"Train subjects: {train_df.subject.nunique()} | Val subjects: {val_df.subject.nunique()} | Test subjects: {test_df.subject.nunique()}")
        print(f"Train rec: {len(train_df)} | Val rec: {len(val_df)} | Test rec: {len(test_df)}")
        print(f"Cache dir: {cfg.cache_dir}")

        for modality, use_att, vname in variants:
            try:
                row = train_one_variant(train_df, val_df, test_df, modality, use_att, fold_name, vname)
                all_rows.append(row)
            except Exception as e:
                msg = str(e)
                print(f"[{fold_name}|{modality}|{'RNN+ATT' if use_att else 'RNN'}] ERROR: {msg}")
                all_rows.append({
                    "fold": fold_name, "modality": modality, "attention": int(use_att),
                    "variant": vname, "error": msg
                })

    results_df = pd.DataFrame(all_rows)
    per_fold_path = os.path.join(cfg.out_dir, f"{cfg.run_name}_results_per_fold.csv")
    results_df.to_csv(per_fold_path, index=False)

    metric_cols = [
        "best_val_subjF1","val_subjF1","val_precision","val_recall","val_auc",
        "test_subjF1","test_precision","test_recall","test_auc","best_val_thr"
    ]
    ok = results_df[(results_df.get("error").fillna("") == "")].copy()

    if len(ok) > 0:
        summary = ok.groupby(["modality","attention","variant"], as_index=False)[metric_cols].agg(["mean","std"])
        summary.columns = ["_".join([c for c in col if c]).strip("_") for col in summary.columns.values]
        summary = summary.rename(columns={"modality_":"modality","attention_":"attention","variant_":"variant"})
    else:
        summary = pd.DataFrame()

    summary_path = os.path.join(cfg.out_dir, f"{cfg.run_name}_results_mean_std.csv")
    summary.to_csv(summary_path, index=False)

    print("\n✅ Saved:")
    print(" -", per_fold_path)
    print(" -", summary_path)
    if cfg.save_subject_predictions:
        print(" - Subject prediction CSVs per fold/variant in:", cfg.out_dir)

    return results_df, summary


# ============================================================
print("✅ DATASET_ROOT:", DATASET_ROOT, "| exists:", os.path.exists(DATASET_ROOT))
print("✅ LABELS_CSV  :", LABELS_CSV,   "| exists:", os.path.exists(LABELS_CSV))


# ============================================================
# df_rec already contains: eeg_path, exg_path, subject, task, label, eeg_cols, emg_col
paired_df = df_rec.copy()

# hard check: paths must be under DATASET_ROOT
if (~paired_df["eeg_path"].astype(str).str.startswith(DATASET_ROOT)).any():
    raise ValueError("Some eeg_path entries are not under DATASET_ROOT")
if (~paired_df["exg_path"].astype(str).str.startswith(DATASET_ROOT)).any():
    raise ValueError("Some exg_path entries are not under DATASET_ROOT")

print("✅ paired_df shape:", paired_df.shape)
print(paired_df.head(3))

# ============================================================
# 3) QUICK EMG CHECK (first few rows)
# ============================================================
for i in range(min(3, len(paired_df))):
    r = paired_df.iloc[i]
    exg = pd.read_csv(r.exg_path)
    if r.emg_col not in exg.columns:
        raise ValueError(f"Row {i}: EMG column '{r.emg_col}' not found in {r.exg_path}")
    stdv = float(exg[r.emg_col].std())
    print(f"🔎 Row {i} EMG std ({r.emg_col}):", stdv)
    if stdv < 1e-10:
        raise ValueError(f"Row {i}: EMG seems flat (std~0). Check EMG column or file content.")

print("✅ EMG looks readable (non-flat) for quick check rows.")

# ============================================================
# 4) TRAIN + SAVE CSVs
# ============================================================
results_df, summary_df = run_kfold(paired_df)

print("\nDone.")
print("Per-fold results CSV:", os.path.join(cfg.out_dir, f"{cfg.run_name}_results_per_fold.csv"))
print("Mean/Std summary CSV:", os.path.join(cfg.out_dir, f"{cfg.run_name}_results_mean_std.csv"))

