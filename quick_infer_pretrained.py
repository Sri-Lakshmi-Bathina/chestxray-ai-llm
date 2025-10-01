import argparse, os, glob, json, sys, time, numpy as np, torch, warnings
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchxrayvision as xrv
from datasets_nih import build_transforms

warnings.filterwarnings("ignore", message=".*normalized correctly.*", category=UserWarning)

def last_conv_layer(module: torch.nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d for CAM.")
    return last

def read_gray_uint8(path):
    with Image.open(path) as im:
        if im.mode != "L":
            im = im.convert("L")
        arr = np.array(im, dtype=np.uint8)
    return arr  # HxW uint8

def resize_cam_like(cam_small: np.ndarray, H: int, W: int) -> np.ndarray:
    u8 = np.uint8(np.clip(cam_small, 0, 1) * 255)
    out = np.asarray(Image.fromarray(u8).resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0
    return out

def main(args):
    print(f"[info] Python: {sys.executable}", flush=True)
    print("[info] Loading pretrained DenseNet121...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device).eval()
    print(f"[info] Model ready on {device} (took {time.time()-t0:.1f}s)", flush=True)

    label_names = getattr(model, "pathologies", None)
    if not label_names:
        out_dim = getattr(model.classifier, "out_features", None) or 18
        label_names = [f"class_{i}" for i in range(out_dim)]

    # Expand image patterns
    image_paths = []
    for pat in args.images:
        image_paths.extend(glob.glob(pat))
    image_paths = sorted(image_paths)
    if args.limit and args.limit > 0:
        image_paths = image_paths[:args.limit]
    if not image_paths:
        print("[error] No images matched your pattern(s).", flush=True)
        return

    os.makedirs(args.out, exist_ok=True)
    results = []

    for idx, p in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] Reading {p}", flush=True)
        gray_u8 = read_gray_uint8(p)                         # 0..255
        gray_f01 = gray_u8.astype(np.float32)/255.0          # for overlay
        img_norm = xrv.datasets.normalize(gray_u8, 255)      # [-1024,1024]
        img_chw = img_norm[None, :, :]
        x = build_transforms(args.img_size)(img_chw)
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x_t)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        n_labels = min(len(label_names), probs.shape[0])
        order = np.argsort(probs[:n_labels])[::-1][:min(args.topk, n_labels)]
        topk = [{"label": label_names[i], "prob": float(probs[i])} for i in order]
        print(f"     Top1: {topk[0]['label']} ({topk[0]['prob']:.3f})", flush=True)

        out_png = ""
        if not args.no_cam:
            print("     Computing Grad-CAM...", flush=True)
            cam = GradCAM(model=model, target_layers=[last_conv_layer(model)])  # keep simple for wide compat
            cam_small = cam(input_tensor=x_t, targets=None)[0]
            H, W = gray_f01.shape
            cam_resized = resize_cam_like(cam_small, H, W)
            img_rgb = np.stack([gray_f01, gray_f01, gray_f01], axis=2).astype(np.float32)
            overlay = show_cam_on_image(img_rgb, cam_resized, use_rgb=True)
            out_png = os.path.join(args.out, os.path.basename(p).rsplit('.',1)[0] + "_cam.png")
            Image.fromarray(overlay).save(out_png)
            print(f"     Saved CAM -> {out_png}", flush=True)

        results.append({"image": os.path.basename(p), "topk": topk, "cam": out_png})

    with open(os.path.join(args.out, "preds.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] Wrote {len(results)} results to {args.out}/preds.json", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_cam", action="store_true", help="Skip Grad-CAM to speed up / debug")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N images (0=all)")
    args = ap.parse_args()
    main(args)
