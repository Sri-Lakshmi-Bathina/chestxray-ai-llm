# 🩺 Chest X-ray AI with Grad-CAM + LLM (Education Demo)

**Research/Education only — not for clinical use or medical advice.**

GitHub repo: https://github.com/Sri-Lakshmi-Bathina/chestxray-ai-llm

This project applies **DenseNet121** and **Explainable AI** to detect 14 chest conditions on the NIH ChestX-ray14 (sample) dataset.  
Includes **Grad-CAM heatmaps**, an **LLM** for plain-English summaries (non-diagnostic), and a **Streamlit** demo.

## ✨ Features
- Pretrained inference via **TorchXRayVision**
- Light **fine-tuning** on the Kaggle *Random Sample of NIH Chest X-ray*
- **Grad-CAM** overlays for interpretability
- **LLM reasoning** for plain-English, non-diagnostic summaries
- **Streamlit** app for interactive inference + XAI

## 📦 Requirements
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## 📁 Dataset Layout
mkdir -p data/images data/splits runs docs

## Place your files like:
 data/
 
   --images/                 
   --sample_labels.csv

## 🚀 Run Commands (End-to-End)
1) Create splits
python src/prepare_sample.py --csv data/sample_labels.csv --imgdir data/images --out data/splits --seed 42

2) Pretrained inference
python src/quick_infer_pretrained.py --images "data/images/*.png" --out runs/pretrained_infer

3) Copy a few CAM images into docs/ for the README
cp runs/pretrained_infer/*cam*.png docs/ 2>/dev/null || true

4) Light fine-tuning (optional)
python src/train_nih.py --train_csv data/splits/train.csv --val_csv data/splits/val.csv \
  --imgdir data/images --epochs 5 --batch_size 16 --lr 1e-4 --freeze_backbone 1 \
  --out runs/sample_d121

5) Evaluation
python src/eval_nih.py --csv data/splits/test.csv --imgdir data/images \
  --ckpt runs/sample_d121/best.pt --out runs/sample_d121/test_metrics.json

6) Streamlit app
streamlit run src/app.py

## 📸 Examples

## 🧰 Tech
Python · PyTorch · TorchXRayVision · scikit-learn · Streamlit

## 📂 Structure
src/ (all code), data/ (local only), runs/ (outputs), docs/ (images for README)

## 🔒 License
MIT (see LICENSE)

## ⚠ Disclaimer
Research/Education only; not a medical device.
