import argparse, pandas as pd, numpy as np
from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import train_test_split

LABELS_14 = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
    "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia"
]

def parse_labels(s):
    labs = [x.strip() for x in str(s).split('|') if x.strip()]
    y = np.zeros(len(LABELS_14), dtype=np.float32)
    if "No Finding" in labs or len(labs)==0: return y
    for i,lab in enumerate(LABELS_14):
        if lab in labs: y[i]=1.0
    return y

def attach_onehot(df):
    Y = np.stack([parse_labels(s) for s in df["Finding Labels"].values])
    for i,lab in enumerate(LABELS_14):
        df[lab] = Y[:,i]
    return df

def patientwise_stratified(df, seed=42):
    # Try to split by patient with multilabel balance (if enough patients)
    patients = df["Patient ID"].unique()
    if len(patients) < 50:
        return None  # too few to do a decent patient-wise stratification

    # aggregate per patient
    lab_map = {pid: np.zeros(len(LABELS_14), dtype=np.float32) for pid in patients}
    for _,r in df.iterrows():
        lab_map[r["Patient ID"]] = np.maximum(lab_map[r["Patient ID"]], parse_labels(r["Finding Labels"]))

    P = np.array(list(lab_map.keys()))
    Y = np.stack([lab_map[pid] for pid in P])

    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, pool_idx = list(mskf.split(P, Y))[0]
    P_train, P_pool = P[train_idx], P[pool_idx]

    # val/test split 50/50 from pool
    mskf2 = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=seed+1)
    val_idx, test_idx = list(mskf2.split(P[pool_idx], Y[pool_idx]))[0]
    P_val, P_test = P_pool[val_idx], P_pool[test_idx]

    tr = df[df["Patient ID"].isin(set(P_train))].copy()
    va = df[df["Patient ID"].isin(set(P_val))].copy()
    te = df[df["Patient ID"].isin(set(P_test))].copy()
    return tr, va, te

def imagewise_stratified(df, seed=42):
    # image-wise multilabel stratification (fallback when too few patients)
    Y = np.stack([parse_labels(s) for s in df["Finding Labels"].values])
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    idx_train, idx_pool = list(mskf.split(np.zeros(len(df)), Y))[0]
    pool = df.iloc[idx_pool]
    Y_pool = Y[idx_pool]
    # val/test 50/50
    mskf2 = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=seed+1)
    idx_val, idx_test = list(mskf2.split(np.zeros(len(pool)), Y_pool))[0]
    return df.iloc[idx_train].copy(), pool.iloc[idx_val].copy(), pool.iloc[idx_test].copy()

def main(args):
    df = pd.read_csv(args.csv)
    # Expect columns like the full NIH CSV: "Image Index","Finding Labels","Patient ID",...
    assert {"Image Index","Finding Labels","Patient ID"}.issubset(df.columns), "CSV missing required columns"

    # Attach one-hot columns
    df = attach_onehot(df)

    splits = patientwise_stratified(df, seed=args.seed)
    if splits is None:
        print("⚠️ Few patients in sample; using image‑wise multilabel stratification.")
        splits = imagewise_stratified(df, seed=args.seed)

    tr, va, te = splits
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tr.to_csv(out/"train.csv", index=False); va.to_csv(out/"val.csv", index=False); te.to_csv(out/"test.csv", index=False)
    print("Saved splits to", out)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--imgdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
