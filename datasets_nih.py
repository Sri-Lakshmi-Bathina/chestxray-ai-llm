import os, torch, pandas as pd, numpy as np, torchvision, torchxrayvision as xrv
from torch.utils.data import Dataset
from PIL import Image

LABELS_14 = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
    "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia"
]

def build_transforms(img_size=224):
    return torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(img_size),
    ])

def _read_gray_float01(path):
    with Image.open(path) as im:
        if im.mode != "L":
            im = im.convert("L")
        arr = np.array(im, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / (255.0 if arr.max() <= 255 else arr.max())
    return arr  # HxW

class NIHImageDataset(Dataset):
    def __init__(self, csv_path, imgdir, img_size=224, augment=False):
        self.df = pd.read_csv(csv_path)
        self.imgdir = imgdir
        self.pathologies = LABELS_14
        self.transforms = build_transforms(img_size)
        self.augment = augment
        self.t_aug = torchvision.transforms.RandomApply([
            torchvision.transforms.RandomRotation(degrees=7),
        ], p=0.5)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.imgdir, row["Image Index"])
        gray = _read_gray_float01(img_path)       # HxW
        img_chw = gray[None, :, :]                # 1xH xW
        x = self.transforms(img_chw)              # 1xH xW after XRV transforms
        x = torch.from_numpy(x)                   # tensor [1,H,W]
        if self.augment:
            x = self.t_aug(x)
        y = torch.tensor(row[self.pathologies].values.astype("float32"))
        return x, y
