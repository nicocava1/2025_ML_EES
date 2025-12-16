import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from shapely.geometry import box
import os

# -----------------------------
# 1. Paths ans directory results
# -----------------------------
ORTHO_PATH = r"C:\Users\cavan\Desktop\Master_SA25\Machine_learning\Machine_Learning_project\Data\Swissimage_t.tif"
CLEAN_POLY = r"C:\Users\cavan\Desktop\Master_SA25\Machine_learning\Machine_Learning_project\Data\Clean_Test.gpkg"
RESULTS_DIR = r"C:\Users\cavan\Desktop\Master_SA25\Machine_learning\Machine_Learning_project\Risultati\Baseline"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# 2. load orthophoto and glacier polygons
# -----------------------------
print("Opening orthophoto...")
src = rasterio.open(ORTHO_PATH)
profile   = src.profile
transform = src.transform
bounds    = src.bounds
H, W = src.height, src.width
print(f"Image size: {H} x {W}")

print("Loading glacier polygons...")
gdf = gpd.read_file(CLEAN_POLY).to_crs(profile["crs"])
gdf = gdf[gdf.intersects(box(*bounds))]

print("Rasterizing glacier mask...")
gt_mask = rasterize(
    [(geom, 1) for geom in gdf.geometry],
    out_shape=(H, W),
    transform=transform,
    fill=0,
    dtype="uint8"
)
print("GT glacier pixels:", gt_mask.sum())
assert gt_mask.sum() > 0, "Ground truth empty"

# -----------------------------
# 3. read RGB image and compute brightness
# -----------------------------
print("Reading RGB image...")
img = src.read().astype(np.float32) / 255.0
img = np.transpose(img, (1, 2, 0))  # (H, W, 3)

brightness = img.mean(axis=2)

# -----------------------------
# 4. Function to compute metrics
# -----------------------------
def compute_metrics(pred, gt):
    TP = np.logical_and(pred == 1, gt == 1).sum()
    FP = np.logical_and(pred == 1, gt == 0).sum()
    FN = np.logical_and(pred == 0, gt == 1).sum()

    iou = TP / (TP + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return iou, precision, recall, f1

# -----------------------------
# 5. Thresholding brightness and validation
# -----------------------------
THRESHOLDS = [0.65, 0.75, 0.85]
results = {}

print("\n=== BASELINE BRIGHTNESS RESULTS ===")
for thr in THRESHOLDS:
    pred = (brightness > thr).astype(np.uint8)
    iou, prec, rec, f1 = compute_metrics(pred, gt_mask)
    results[thr] = (iou, prec, rec, f1)

    print(f"\nThreshold {thr}")
    print(f"IoU       : {iou:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")


