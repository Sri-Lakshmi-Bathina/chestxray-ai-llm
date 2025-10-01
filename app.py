import os, json, torch, numpy as np, warnings, html
import streamlit as st
from PIL import Image
from datasets_nih import LABELS_14, build_transforms
import torchxrayvision as xrv

# Try Grad-CAM; keep app usable if it's not installed
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    CAM_AVAILABLE = True
except Exception:
    CAM_AVAILABLE = False

# Silence TorchXRayVision's heuristic normalization warning
warnings.filterwarnings("ignore", message=".*normalized correctly.*", category=UserWarning)

st.set_page_config(page_title="NIH Sample CXR + DenseNet121 + LLM", layout="wide")
st.title("🩻 NIH Chest X-ray — DenseNet121 + LLM (Research Demo)")
st.caption("Educational demo — not for medical use.")

use_pretrained = st.toggle("Use pretrained model (no fine-tuning)", value=True)
ckpt = ""
if not use_pretrained:
    ckpt = st.text_input("Path to fine-tuned checkpoint (.pt)", "runs/sample_d121/best.pt")

img_size = st.slider("Image size", 192, 512, 224, 32)
threshold = st.slider("Positive threshold", 0.0, 1.0, 0.5, 0.01)
uploaded = st.file_uploader("Upload PNG/JPG chest X-ray (convert DICOM to PNG first)", type=["png","jpg","jpeg"])

def last_conv_layer(module: torch.nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d for CAM.")
    return last

def read_gray_uint8(filelike) -> np.ndarray:
    """Read an uploaded file or path as 8-bit grayscale (0..255)."""
    with Image.open(filelike) as im:
        if im.mode != "L":
            im = im.convert("L")
        return np.array(im, dtype=np.uint8)

def resize_cam_like(cam_small: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize a 2D CAM [0,1] to (H,W) via PIL bilinear."""
    u8 = np.uint8(np.clip(cam_small, 0, 1) * 255)
    out = np.asarray(Image.fromarray(u8).resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0
    return out

# ----- LLM helper -----
def run_llm(topk):
    # Lazy import so the app still runs without it until needed
    from llm_reasoning import run_reasoning_and_precautions
    return run_reasoning_and_precautions(topk)

if uploaded:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------- Load model ----------
    if use_pretrained or not ckpt or not os.path.exists(ckpt):
        model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device).eval()
        label_names = getattr(model, "pathologies", None)
        if not label_names:
            out_dim = getattr(model.classifier, "out_features", None) or 18
            label_names = [f"class_{i}" for i in range(out_dim)]
    else:
        model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
        in_feat = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_feat, len(LABELS_14))
        sd = torch.load(ckpt, map_location=device)
        model.load_state_dict(sd)
        # Disable XRV's 18-label op_norm & set schema to NIH-14 for our fine-tuned head
        model.op_threshs = None
        model.pathologies = LABELS_14
        model.eval()
        label_names = LABELS_14

    os.makedirs("runs/app", exist_ok=True)
    tmp_path = os.path.join("runs/app", uploaded.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded.read())

    # --------- Read & normalize exactly as XRV expects ----------
    gray_u8 = read_gray_uint8(tmp_path)                            # 0..255
    gray_f01 = gray_u8.astype(np.float32) / 255.0                  # for display & overlay
    img_norm = xrv.datasets.normalize(gray_u8, 255).astype("float32")  # [-1024, 1024]
    img_chw = img_norm[None, :, :]                                 # (1, H, W)
    x = build_transforms(img_size)(img_chw)                        # XRV center-crop + resize
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)              # [1,1,H',W']

    # --------- Predict ----------
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    n_labels = min(len(label_names), probs.shape[0])
    order = np.argsort(probs[:n_labels])[::-1]
    topk = [{"label": label_names[i],
             "prob": float(probs[i]),
             "positive@thr": bool(probs[i] >= threshold)} for i in order[:min(6, n_labels)]]

    st.subheader("Top findings")
    st.table(topk)

    # --------- Show image & Grad-CAM (if available) ----------
    colA, colB = st.columns(2)
    colA.subheader("Image")
    colA.image(gray_f01, clamp=True, use_container_width=True)   # <-- no deprecation

    if CAM_AVAILABLE:
        try:
            cam = GradCAM(model=model, target_layers=[last_conv_layer(model)])
            cam_small = cam(input_tensor=x_t, targets=None)[0]
            H, W = gray_f01.shape
            cam_resized = resize_cam_like(cam_small, H, W)
            img_rgb = np.stack([gray_f01, gray_f01, gray_f01], axis=2).astype(np.float32)
            overlay = show_cam_on_image(img_rgb, cam_resized, use_rgb=True)  # uint8
            cam_path = os.path.join("runs/app", "gradcam.png")
            Image.fromarray(overlay).save(cam_path)

            colB.subheader("Grad-CAM")
            colB.image(cam_path, use_container_width=True)
        except Exception as e:
            colB.warning(f"Grad-CAM unavailable: {e}")
    else:
        colB.info("Grad-CAM not installed. Install `pytorch-grad-cam` to enable heatmaps.")

    # --------- LLM explanation & precautions ----------
    if st.button("Generate LLM explanation + precautions"):
        try:
            out = run_llm(topk)
            st.subheader("Explanation (LLM)")
            st.write(out["reasoning"])
            st.subheader("Precautions (LLM)")
            st.write(out["precautions"])
            st.caption("Automatically generated; may be inaccurate. Consult a clinician.")
        except Exception as e:
            st.error(f"LLM error: {e}")
