import argparse, json, numpy as np, torch, pandas as pd, torchxrayvision as xrv
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from datasets_nih import NIHImageDataset, LABELS_14

def load_model(ckpt, num_labels=14):
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    in_feat = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_feat, num_labels)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model

def main(args):
    device = ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = NIHImageDataset(args.csv, args.imgdir, img_size=args.img_size, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model = load_model(args.ckpt, num_labels=len(LABELS_14)).to(device)

    all_probs, all_true = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs); all_true.append(y.numpy())

    P = np.concatenate(all_probs); T = np.concatenate(all_true)
    metrics = {"per_class": {}}
    aurocs, aps = [], []
    for i,lab in enumerate(LABELS_14):
        try:
            auroc = roc_auc_score(T[:,i], P[:,i]); ap = average_precision_score(T[:,i], P[:,i])
        except Exception:
            auroc, ap = float("nan"), float("nan")
        metrics["per_class"][lab] = {"AUROC": float(auroc), "AP": float(ap)}
        aurocs.append(auroc); aps.append(ap)
    metrics["macro_AUROC"] = float(np.nanmean(aurocs))
    metrics["macro_AP"] = float(np.nanmean(aps))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--imgdir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    main(args)
