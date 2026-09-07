# sanity_check.py — corrected threshold for 0/1 label-encoded masks
import cv2, numpy as np, matplotlib.pyplot as plt, glob, os

IMAGE_DIR = "./datasets/idd/image_archive"
MASK_DIR = "./datasets/idd/mask_archive"
import os


image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*")))[:5]
mask_files = sorted(glob.glob(os.path.join(MASK_DIR, "*")))[:5]

print("IMAGE_DIR:", IMAGE_DIR, "| exists:", os.path.isdir(IMAGE_DIR))
print("MASK_DIR:", MASK_DIR, "| exists:", os.path.isdir(MASK_DIR))
print("image_files found:", len(image_files))
print("mask_files found:", len(mask_files))

fig, axes = plt.subplots(5, 3, figsize=(10, 16))
for i, (img_path, mask_path) in enumerate(zip(image_files, mask_files)):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    road_fraction = (mask > 0).mean()  # fixed: >0, not >127
    print(
        f"{os.path.basename(mask_path)}  unique={np.unique(mask)}  road_fraction={road_fraction:.3f}"
    )

    axes[i, 0].imshow(img)
    axes[i, 0].axis("off")
    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 1].axis("off")
    overlay = img.copy()
    road = mask > 0
    overlay[road] = overlay[road] * 0.5 + np.array([0, 255, 0]) * 0.5
    axes[i, 2].imshow(overlay.astype(np.uint8))
    axes[i, 2].axis("off")

plt.tight_layout()
plt.savefig("sanity_check.png", dpi=150)
print(
    "Saved. Open it and actually look — does green land ON the road or everywhere except it?"
)
