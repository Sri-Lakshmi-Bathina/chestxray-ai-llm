import argparse, os, json, copy, numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import torchxrayvision as xrv
from datasets_nih import NIHImageDataset, LABELS_14

def pos_weights_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    Y = df[LABELS_14].values
    pos = Y.sum(axis=0)
    neg = (Y.shape[0] - pos)
    w = (neg / np.clip(pos, 1, None)).astype("float32")
    return torch.tensor(w)

def build_model(num_labels=14, freeze_backbone=False):
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    in_feat = model.classifier.in_features
    model.classifier = nn.Linear(in_feat, num_labels)
    if freeze_backbone:
        for n,p in model.named_parameters():
            if "classifier" not in n:
                p.requires_grad = False
    return model

def evaluate(model, loader, device):
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs); all_true.append(y.numpy())
    P = np.concatenate(all_probs); T = np.concatenate(all_true)
    aurocs, aps = [], []
    for i in range(P.shape[1]):
        try:
            aurocs.append(roc_auc_score(T[:,i], P[:,i]))
            aps.append(average_precision_score(T[:,i], P[:,i]))
        except Exception:
            aurocs.append(float("nan")); aps.append(float("nan"))
    return {"AUROC_macro": float(np.nanmean(aurocs)),
            "AP_macro": float(np.nanmean(aps)),
            "AUROC_per_class": aurocs, "AP_per_class": aps}

def main(args):
    device = ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_ds = NIHImageDataset(args.train_csv, args.imgdir, img_size=args.img_size, augment=not args.no_aug)
    val_ds   = NIHImageDataset(args.val_csv,   args.imgdir, img_size=args.img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(num_labels=len(LABELS_14), freeze_backbone=bool(args.freeze_backbone)).to(device)
    pos_w = pos_weights_from_csv(args.train_csv).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, args.epochs))

    best = {"metric": -1, "state": None}
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x,y in pbar:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        metrics = evaluate(model, val_loader, device)
        sched.step()
        print(f"[val] AUROC_macro={metrics['AUROC_macro']:.4f} AP_macro={metrics['AP_macro']:.4f}")
        if metrics["AUROC_macro"] > best["metric"]:
            best["metric"] = metrics["AUROC_macro"]
            best["state"] = copy.deepcopy(model.state_dict())
            torch.save(best["state"], os.path.join(args.out, "best.pt"))
            with open(os.path.join(args.out, "val_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--imgdir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--freeze_backbone", type=int, default=1)
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args)
