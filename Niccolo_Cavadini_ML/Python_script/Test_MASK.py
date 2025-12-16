import rasterio
import torch
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
import os

# ================================
# PATHS
# ================================
TEST_ORTHO = r"C:\Users\cavan\Desktop\Master_SA25\Machine_learning\Machine_Learning_project\Data\Swissimage_t.tif"
MODEL_PATH = r"C:\Users\cavan\Desktop\Master_SA25\Machine_learning\Machine_Learning_project\Risultati\best_model_valLoss.pth"
CLEAN_POLY = r"C:\Users\cavan\Desktop\Master_SA25\Machine_learning\Machine_Learning_project\Data\Clean_Test.gpkg"
DEM_PATH    = r"C:\Users\cavan\Desktop\Master_SA25\Machine_learning\Machine_Learning_project\Data\switzerland_dem.tif"

OUT_DIR = r"C:\Users\cavan\Desktop\Master_SA25\Machine_learning\Machine_Learning_project\Risultati"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PRED = os.path.join(OUT_DIR, "prediction_binary_demmask.tif")
OUT_CONF = os.path.join(OUT_DIR, "confusion_map_demmask.tif")

PATCH_SIZE = 256
ELEV_THRESH = 2400

# ================================
# LOAD MODEL
# ================================
from U_Net_improved import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ================================
# LOAD TEST IMAGE
# ================================
with rasterio.open(TEST_ORTHO) as src:
    test_img = src.read().transpose(1,2,0).astype(np.float32) / 255.0
    test_profile = src.profile
    transform = src.transform
    crs = src.crs
    H, W, _ = test_img.shape

# ================================
# GT MASK
# ================================
gdf = gpd.read_file(CLEAN_POLY).to_crs(crs)

gt_mask = rasterize(
    [(geom, 1) for geom in gdf.geometry],
    out_shape=(H, W),
    transform=transform,
    fill=0,
    dtype="uint8"
)

# ================================
# LOAD + ALIGN DEM SAFELY
# ================================
with rasterio.open(DEM_PATH) as dem_src:
    dem_src_arr = dem_src.read(1).astype(np.float32)
    dem_aligned = np.zeros((H, W), dtype=np.float32)

    reproject(
        source=dem_src_arr,
        destination=dem_aligned,
        src_transform=dem_src.transform,
        src_crs=dem_src.crs,
        dst_transform=transform,
        dst_crs=crs,
        resampling=Resampling.bilinear
    )

# ================================
# RUN PREDICTION
# ================================
pred_map = np.zeros((H, W), dtype=np.float32)

for i in range(0, H, PATCH_SIZE):
    for j in range(0, W, PATCH_SIZE):

        patch = test_img[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
        if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
            continue

        patch_t = torch.tensor(patch).permute(2,0,1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred = model(patch_t).cpu().squeeze().numpy()

        pred_map[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = pred

# ================================
# BINARIZE
# ================================
pred_bin = (pred_map > 0.5).astype(np.uint8)

# ================================
# APPLY DEM FILTER
# ================================
pred_bin_dem = pred_bin.copy()
pred_bin_dem[dem_aligned < ELEV_THRESH] = 0

# ================================
# SAVE PREDICTION
# ================================
out_profile = test_profile.copy()
out_profile.update(count=1, dtype="uint8")

with rasterio.open(OUT_PRED, "w", **out_profile) as dst:
    dst.write(pred_bin_dem, 1)

# ================================
# CONFUSION MAP
# ================================
TP = (pred_bin_dem == 1) & (gt_mask == 1)
FP = (pred_bin_dem == 1) & (gt_mask == 0)
FN = (pred_bin_dem == 0) & (gt_mask == 1)

conf = np.zeros((H,W), dtype=np.uint8)
conf[TP] = 1
conf[FP] = 2
conf[FN] = 3

with rasterio.open(OUT_CONF, "w", **out_profile) as dst:
    dst.write(conf, 1)
    dst.write_colormap(1,{
        0:(0,0,0,255),
        1:(0,255,0,255),
        2:(255,0,0,255),
        3:(0,0,255,255)
    })

# ================================
# METRICS
# ================================
tp = TP.sum()
fp = FP.sum()
fn = FN.sum()

iou = tp / (tp + fp + fn + 1e-8)
prec = tp / (tp + fp + 1e-8)
rec  = tp / (tp + fn + 1e-8)
f1   = 2 * prec * rec / (prec + rec + 1e-8)

print("\nPOSTPROCESSING RESULTS (DEM reprojected + 2400m mask)")
print("=========================================================")
print(f"IoU:       {iou:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nSaved files:")
print(OUT_PRED)
print(OUT_CONF)
