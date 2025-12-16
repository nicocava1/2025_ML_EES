import os
import rasterio
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import box
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================================
# PATHS & PARAMS
# ================================
ORTHO_PATH = r"C:\Users\cavan\Desktop\Master_SA25\Machine_learning\Machine_Learning_project\Data\Swissimage.tif"
CLEAN_POLY = r"C:\Users\cavan\Desktop\Master_SA25\Machine_learning\Machine_Learning_project\Data\Clean_Train.gpkg"

PATCH_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
PATIENCE = 5

RESULTS_DIR = r"C:\Users\cavan\Desktop\Master_SA25\Machine_learning\Machine_Learning_project\Risultati"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================================
# OPEN ORTHO LAZY
# ================================
print("Opening orthophoto in lazy mode...")
src = rasterio.open(ORTHO_PATH)

profile = src.profile
transform = src.transform
bounds = src.bounds

H, W = src.height, src.width
print("Image size:", H, W)

# ================================
# LOAD CLEAN POLYGONS
# ================================
print("Loading clean glacier polygons...")
gdf = gpd.read_file(CLEAN_POLY).to_crs(profile["crs"])
bbox = box(*bounds)
gdf = gdf[gdf.intersects(bbox)]

# ================================
# RASTERIZE MASK
# ================================
print("Rasterizing glacier mask...")
mask = rasterize(
    [(geom, 1) for geom in gdf.geometry],
    out_shape=(H, W),
    transform=transform,
    fill=0,
    dtype="uint8"
)

# ================================
# AUGMENTATION
# ================================
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    ToTensorV2()
])

val_transform = A.Compose([ToTensorV2()])

# ================================
# PATCH DATASET (LAZY)
# ================================
class PatchDataset(Dataset):
    def __init__(self, mask, patch_size, transform,
                 n_pos=6000, n_neg=5000, min_pos_px=50):

        self.mask = mask
        self.ps = patch_size
        self.transform = transform

        H, W = mask.shape
        self.coords = []

        pos = neg = 0
        attempts = 0
        max_attempts = (n_pos + n_neg) * 50

        print("Sampling patches...")

        while (pos < n_pos or neg < n_neg) and attempts < max_attempts:
            i = np.random.randint(0, H-self.ps)
            j = np.random.randint(0, W-self.ps)

            s = mask[i:i+self.ps, j:j+self.ps].sum()

            if s >= min_pos_px and pos < n_pos:
                self.coords.append((i, j))
                pos += 1
            elif s == 0 and neg < n_neg:
                self.coords.append((i, j))
                neg += 1

            attempts += 1

        np.random.shuffle(self.coords)
        print(f"Sampled patches: pos={pos}, neg={neg}")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        i, j = self.coords[idx]

        window = rasterio.windows.Window(j, i, self.ps, self.ps)
        img = src.read(window=window).astype(np.float32) / 255.0
        img = np.transpose(img, (1, 2, 0))

        msk = self.mask[i:i+self.ps, j:j+self.ps]

        if self.transform:
            aug = self.transform(image=img, mask=msk)
            img = aug["image"]
            msk = aug["mask"].unsqueeze(0)
        else:
            img = torch.tensor(img).permute(2,0,1)
            msk = torch.tensor(msk).unsqueeze(0)

        return img.float(), msk.float()

# ================================
# DATASET + SPLIT
# ================================
dataset = PatchDataset(mask, PATCH_SIZE, None)

idx = np.arange(len(dataset))
train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)

train_ds.dataset.transform = train_transform
val_ds.dataset.transform = val_transform

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ================================
# MODEL
# ================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(3,64)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(64,128)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(128,256)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(256,512)

        self.u1 = nn.ConvTranspose2d(512,256,2,2)
        self.c1 = DoubleConv(512,256)

        self.u2 = nn.ConvTranspose2d(256,128,2,2)
        self.c2 = DoubleConv(256,128)

        self.u3 = nn.ConvTranspose2d(128,64,2,2)
        self.c3 = DoubleConv(128,64)

        self.out = nn.Conv2d(64,1,1)

    def forward(self,x):
        d1 = self.d1(x); p1 = self.p1(d1)
        d2 = self.d2(p1); p2 = self.p2(d2)
        d3 = self.d3(p2); p3 = self.p3(d3)
        d4 = self.d4(p3)

        x = self.u1(d4); x = self.c1(torch.cat([x,d3],1))
        x = self.u2(x);  x = self.c2(torch.cat([x,d2],1))
        x = self.u3(x);  x = self.c3(torch.cat([x,d1],1))

        return torch.sigmoid(self.out(x))

# ================================
# LOSS + IoU FUNCTION
# ================================
class DiceLoss(nn.Module):
    def forward(self,p,t,eps=1e-6):
        p = p.flatten()
        t = t.flatten()
        i = (p*t).sum()
        return 1 - (2*i + eps)/(p.sum()+t.sum()+eps)

def iou_metric(pred, target):
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - inter
    return (inter / (union + 1e-6)).item()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(device)

bce = nn.BCELoss()
dice = DiceLoss()
opt = torch.optim.Adam(model.parameters(), lr=LR)

# ================================
# TRAIN WITH IoU + VAL LOSS
# ================================
train_losses = []
val_losses   = []
train_ious   = []
val_ious     = []

best_val_loss = np.inf
pat = 0

print("Starting training...")

for ep in range(EPOCHS):

    # ---- TRAIN ----
    model.train()
    train_loss = 0
    train_iou_epoch = 0

    for x,y in tqdm(train_loader, desc=f"Epoch {ep+1}/{EPOCHS}"):
        x,y = x.to(device), y.to(device)

        pred = model(x)
        loss = bce(pred,y) + dice(pred,y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss += loss.item()
        train_iou_epoch += iou_metric(pred, y)

    train_loss /= len(train_loader)
    train_iou_epoch /= len(train_loader)

    train_losses.append(train_loss)
    train_ious.append(train_iou_epoch)

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0
    val_iou_epoch = 0

    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            pred = model(x)
            loss = bce(pred,y) + dice(pred,y)

            val_loss += loss.item()
            val_iou_epoch += iou_metric(pred, y)

    val_loss /= len(val_loader)
    val_iou_epoch /= len(val_loader)

    val_losses.append(val_loss)
    val_ious.append(val_iou_epoch)

    print(f"Epoch {ep+1}: "
          f"TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}  "
          f"TrainIoU={train_iou_epoch:.4f}  ValIoU={val_iou_epoch:.4f}")

    # ---- EARLY STOPPING ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        pat = 0
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR,"best_model_valLoss.pth"))
        print("Saved new BEST model")
    else:
        pat += 1
        if pat >= PATIENCE:
            print("Early stopping triggered.")
            break

print("Training complete.")

# ================================
# PLOT TRAIN vs VAL LOSS & IoU
# ================================
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss over Epochs")
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR,"loss_plot.png"))
plt.show()

plt.figure(figsize=(8,5))
plt.plot(train_ious, label="Train IoU")
plt.plot(val_ious, label="Val IoU")
plt.legend()
plt.title("IoU over Epochs")
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR,"iou_plot.png"))
plt.show()
