#!/usr/bin/env python
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.metrics import structural_similarity as ssim 
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision import transforms
from torchvision.transforms import InterpolationMode

try:
    plt.style.use("seaborn-v0_8-paper")
    print("using seaborn-v0_8-paper style")
except OSError:
    print("falling back to ggplot style")
    plt.style.use("ggplot")

plt.rcParams["font.family"]    = "serif"
plt.rcParams["font.serif"]     = ["Nimbus Roman No9 L"]
plt.rcParams["font.size"]      = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["legend.fontsize"]= 10
plt.rcParams["xtick.labelsize"]= 10
plt.rcParams["ytick.labelsize"]= 10

DATA_RANGE = 2000.0    # CT range −1000…1000
ORIG_H, ORIG_W = 238, 366
PAD_L, PAD_T, PAD_R, PAD_B = 0, 64, 0, 64
RES_H, RES_W = 256, 256

_pad_h = ORIG_H + PAD_T + PAD_B
_pad_w = ORIG_W + PAD_L + PAD_R
TOP_CROP    = int(round((PAD_T / _pad_h) * RES_H))    # ≈ 45
BOTTOM_CROP = int(round((PAD_B / _pad_h) * RES_H))    # ≈ 45
LEFT_CROP   = int(round((PAD_L / _pad_w) * RES_W))    #   0
RIGHT_CROP  = int(round((PAD_R / _pad_w) * RES_W))    #   0

gt_transform = transforms.Compose([
    transforms.Pad((PAD_L, PAD_T, PAD_R, PAD_B), fill=-1000),
    transforms.Resize((RES_H, RES_W)),
])
mask_transform = transforms.Compose([
    transforms.Pad((PAD_L, PAD_T, PAD_R, PAD_B), fill=0),
    transforms.Resize((RES_H, RES_W), interpolation=InterpolationMode.NEAREST),
])

def apply_transform(img_np):
    """
    Pad a raw slice to 366×366 (fill=-1000), then resize to 256×256.
    Returns a NumPy array of shape 256×256.
    """
    t = torch.from_numpy(img_np).unsqueeze(0).float()
    out = gt_transform(t)  # Torch Tensor, (1×256×256)
    return out.squeeze(0).numpy()

def apply_transform_to_mask(mask_np):
    t = torch.from_numpy(mask_np.astype(np.uint8)).unsqueeze(0).float()
    out = mask_transform(t).squeeze(0).numpy()
    return out > 0.5

def crop_back(arr):
    """
    Crop out ~45 px top & bottom from a 256×256 array → 166×256.
    """
    return arr[TOP_CROP:RES_H - BOTTOM_CROP, LEFT_CROP:RES_W - RIGHT_CROP]

SLICE_RANGES = {
    3: None, 8: (0, 354), 12: (0, 320), 26: None,
    32: (69, 269), 33: (59, 249), 35: (91, 268),
    54: (0, 330), 59: (0, 311), 61: (0, 315),
    106: None, 116: None, 129: (5, 346)
}
VALID_VOLUMES = list(SLICE_RANGES.keys())

def get_slice_files(folder, vol_idx, is_cbct=False):
    """
    List slice file paths for a given volume index.
    If is_cbct is False, expects folder/volume-<idx>/volume-<idx>_slice_*.npy,
    otherwise folder/volume-<idx>_slice_*.npy.
    Applies SLICE_RANGES to filter slices.
    """
    base = folder if is_cbct else os.path.join(folder, f"volume-{vol_idx}")
    pattern = os.path.join(base, f"volume-{vol_idx}_slice_*.npy")
    files = sorted(glob.glob(pattern))
    rng = SLICE_RANGES.get(vol_idx)
    if rng:
        start, end = rng
        files = [
            f for f in files
            if start <= int(os.path.basename(f).split('_')[-1].split('.')[0]) <= end
        ]
    return files

def compute_hu_histogram(folder, vols, is_cbct=False, bins=np.linspace(-1000, 1000, 200)):
    """
    Compute aggregated histogram (counts) across all selected volumes and slices.
    If is_cbct, applies pad→resize→crop→256; else assumes already pre-cropped.
    """
    hist = np.zeros(len(bins) - 1, dtype=np.float64)
    for v in vols:
        print("Processing volume:", v)
        for fp in get_slice_files(folder, v, is_cbct):
            data = np.load(fp)
            if is_cbct:
                data = apply_transform(data)  
            data = crop_back(data)           
            hist += np.histogram(data.flatten(), bins=bins)[0]
    return hist

def plot_hu_distributions(
    gt_folder,
    cbct_folder,
    pred_folders,
    vols,
    save_path=None
):
    """
    Plot HU histograms for GT (CT), CBCT, and multiple prediction methods
    using the same style/tweaks as the profile‐plot.
    """
    bins = np.linspace(-1000, 1000, 200)
    ctr = (bins[:-1] + bins[1:]) / 2

    cth = compute_hu_histogram(gt_folder, vols, is_cbct=True,  bins=bins)  
    cbh = compute_hu_histogram(cbct_folder, vols, is_cbct=True, bins=bins)  

    cth_norm = cth / cth.sum() if cth.sum() > 0 else cth
    cbh_norm = cbh / cbh.sum() if cbh.sum() > 0 else cbh

    pred_norms = {}
    for label, folder in pred_folders:
        pth = compute_hu_histogram(folder, vols, is_cbct=False, bins=bins)
        pred_norms[label] = pth / pth.sum() if pth.sum() > 0 else pth

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.10, 0.12, 0.88, 0.82])  # [left, bottom, width, height]

    COLORS = {
        "CT":   "#1f77b4",  # muted blue
        "CBCT": "#ff7f0e",  # orange
        "sCT":   "#d62728",  # red
    }
    STYLES = {
        "CT":   {"linestyle": "-",  "linewidth": 3.0},
        "CBCT": {"linestyle": ":", "linewidth": 3.0},
        "sCT":   {"linestyle":"--",  "linewidth":3.0}, 
    }
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax.plot(
        ctr,
        cth_norm,
        label="CT",
        color=COLORS["CT"],
        **STYLES["CT"]
    )
    ax.plot(
        ctr,
        cbh_norm,
        label="CBCT",
        color=COLORS["CBCT"],
        **STYLES["CBCT"]
    )

    for idx, (label, _) in enumerate(pred_folders):
        color = COLORS[label]
        ax.plot(
            ctr,
            pred_norms[label],
            label=label,
            color=color,
            linestyle="--",    
            linewidth=2.5
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(-1050, 1050)
    ax.set_ylim(0, 0.10)
    ax.set_xlabel("Hounsfield Units (HU)", fontsize=18)
    ax.set_ylabel("Normalized Frequency",     fontsize=18)

    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.minorticks_off()

    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.xaxis.grid(False)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=len(pred_folders) + 2,  # +2 for CT & CBCT
        frameon=False,
        fontsize=16
    )

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        print("Saved histogram figure to:", save_path)
    plt.show()

if __name__ == "__main__":
    vols = VALID_VOLUMES

    gt_folder   = os.path.expanduser("~/thesis/training_data/CT/test")
    cbct_folder = os.path.expanduser("~/thesis/training_data/CBCT/test")

    pred_folders = [
        ("sCT", os.path.expanduser(
            "~/thesis/predictions/predictions_controlnet_v7-data-augmentation"
        )),
    ]

    save_path = os.path.expanduser("~/thesis/figures/hu_distribution_comparison.pdf")

    plot_hu_distributions(
        gt_folder=gt_folder,
        cbct_folder=cbct_folder,
        pred_folders=pred_folders,
        vols=vols,
        save_path=save_path
    )
