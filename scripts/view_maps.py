import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim 


DATA_RANGE = 2000.0  # HU range for SSIM computations (–1000…+1000)

ORIG_H, ORIG_W = 238, 366
PAD_L, PAD_T, PAD_R, PAD_B = 0, 64, 0, 64
RES_H, RES_W = 256, 256

_pad_h = ORIG_H + PAD_T + PAD_B  # = 366
_pad_w = ORIG_W + PAD_L + PAD_R  # = 366

TOP_CROP    = int(round((PAD_T / _pad_h) * RES_H))    # ≈ 45
BOTTOM_CROP = int(round((PAD_B / _pad_h) * RES_H))    # ≈ 45
LEFT_CROP   = int(round((PAD_L / _pad_w) * RES_W))    #   0
RIGHT_CROP  = int(round((PAD_R / _pad_w) * RES_W))    #   0

transform = transforms.Compose([
    transforms.Pad((PAD_L, PAD_T, PAD_R, PAD_B), fill=-1000),
    transforms.Resize((RES_H, RES_W)),
])

def apply_transform(np_img: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(np_img)
    out = transform(pil)       
    return np.array(out)      

def crop_back(arr: np.ndarray) -> np.ndarray:
    return arr[
        TOP_CROP : RES_H - BOTTOM_CROP,   # [45:211] → 166 rows
        LEFT_CROP: RES_W - RIGHT_CROP     # [0:256]   → 256 cols
    ]

def resize_to_256(slc: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(slc.astype(np.int16))
    resized = pil.resize((256, 256), resample=Image.BILINEAR)
    return np.array(resized)


def load_raw_slice(folder: str, volume_idx: int, slice_name: str) -> np.ndarray:
    candidate1 = os.path.join(folder, f"volume-{volume_idx}", slice_name)
    file = None
    if os.path.isfile(candidate1):
        file = np.load(candidate1)
    candidate2 = os.path.join(folder, slice_name)
    if os.path.isfile(candidate2):
        file =  np.load(candidate2)
    if file is None:
        raise FileNotFoundError(
            f"Could not find {slice_name} in {folder!r} "
            f"(checked both volume-{volume_idx}/{slice_name} and {slice_name})."
        )
    return np.fliplr(file)

def load_and_process_CT_CBCT_slice(
    folder: str,
    volume_idx: int,
    slice_name: str
) -> np.ndarray:
    raw = load_raw_slice(folder, volume_idx, slice_name)   # (238×366)
    padded = apply_transform(raw)                           # (256×256)
    cropped = crop_back(padded)                             # (166×256)
    final256 = resize_to_256(cropped)                       # (256×256)
    return final256

def plot_professional_maps(
    ct_img: np.ndarray,
    cbct_img: np.ndarray,
    pred_imgs: dict,
    volume_idx: int,
    slice_name: str
):
    labels = ["CT", "CBCT"] + list(pred_imgs.keys())
    imgs   = [ct_img, cbct_img] + [pred_imgs[l] for l in pred_imgs]
    n_cols = len(labels)

    # Compute SSIM, MAE, ΔHU
    SSIM_maps = []
    for img in imgs:
        _, s_map = ssim(ct_img, img, full=True, data_range=DATA_RANGE)
        SSIM_maps.append(s_map)
    MAE_maps  = [np.abs(img - ct_img) for img in imgs]
    DIFF_maps = [(img - ct_img).astype(np.float32) for img in imgs]

    plt.rcParams["font.family"]    = "serif"
    plt.rcParams["font.serif"]     = ["Nimbus Roman No9 L"]
    plt.rcParams["font.size"]      = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["figure.titlesize"] = 12

    cell_inch = 1.5
    fig_w = cell_inch * n_cols
    fig_h = cell_inch * 4

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        nrows=4,
        ncols=n_cols + 1,
        width_ratios=[1] * n_cols + [0.10], 
        height_ratios=[1, 1, 1, 1],
        wspace=0.02,
        hspace=0.12
    )

    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(4)]
    row_labels = ["Axial", "SSIM", "Absolute Error", "ΔHU"]

    for col_idx in range(n_cols):
        lab   = labels[col_idx]
        raw   = imgs[col_idx]
        s_map = SSIM_maps[col_idx]
        mae_m = MAE_maps[col_idx]
        diff_m= DIFF_maps[col_idx]

        ax0 = axes[0][col_idx]
        im0 = ax0.imshow(raw, cmap="gray", vmin=-1000, vmax=1000)
        ax0.axis("off")
        if col_idx == 0:
            ax0.text(-0.08, 0.5, row_labels[0],
                     va="center", ha="center", rotation="vertical",
                     fontsize=10, transform=ax0.transAxes)
        ax0.set_title(lab, pad=6)

        ax1 = axes[1][col_idx]
        im1 = ax1.imshow(s_map, cmap="viridis", vmin=0, vmax=1)
        ax1.axis("off")
        if col_idx == 0:
            ax1.text(-0.08, 0.5, row_labels[1],
                     va="center", ha="center", rotation="vertical",
                     fontsize=10, transform=ax1.transAxes)

        ax2 = axes[2][col_idx]
        im2 = ax2.imshow(mae_m, cmap="hot", vmin=0, vmax=300)
        ax2.axis("off")
        if col_idx == 0:
            ax2.text(-0.08, 0.5, row_labels[2],
                     va="center", ha="center", rotation="vertical",
                     fontsize=10, transform=ax2.transAxes)

        ax3 = axes[3][col_idx]
        im3 = ax3.imshow(diff_m, cmap="gray", vmin=-300, vmax=300)
        ax3.axis("off")
        if col_idx == 0:
            ax3.text(-0.08, 0.5, row_labels[3],
                     va="center", ha="center", rotation="vertical",
                     fontsize=10, transform=ax3.transAxes)

    cax_ssim = fig.add_subplot(gs[1, n_cols])
    cb1 = fig.colorbar(im1, cax=cax_ssim, orientation="vertical")
    cax_ssim.set_ylabel("")  
    cax_ssim.yaxis.set_tick_params(labelsize=8)

    cax_mae = fig.add_subplot(gs[2, n_cols])
    cb2 = fig.colorbar(im2, cax=cax_mae, orientation="vertical")
    cb2.set_ticks([0, 100, 200, 300])
    cax_mae.set_ylabel("")
    cax_mae.yaxis.set_tick_params(labelsize=8)

    cax_diff = fig.add_subplot(gs[3, n_cols])
    cb3 = fig.colorbar(im3, cax=cax_diff, orientation="vertical")
    cb3.set_ticks([-300, -150, 0, 150, 300])
    cax_diff.set_ylabel("")
    cax_diff.yaxis.set_tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()

    save_path = os.path.expanduser(
        f"~/thesis/figures/ssim_mae_diff_vol{volume_idx}_{slice_name}.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print("Saved figure to:", save_path)



if __name__ == "__main__":
    volume_idx = 8
    slice_idx = 145
    slice_name = f"volume-{volume_idx}_slice_{slice_idx}.npy"

    gt_folder   = os.path.expanduser("~/thesis/training_data/CT/test")
    cbct_folder = os.path.expanduser("~/thesis/training_data/CBCT/490/test")
    pred_folders = {
        "sCT":     os.path.expanduser(
            "~/thesis/predictions/predictions_controlnet_v7-data-augmentation"
        ),
    }

    ct_img   = load_and_process_CT_CBCT_slice(gt_folder, volume_idx, slice_name)
    cbct_img = load_and_process_CT_CBCT_slice(cbct_folder, volume_idx, slice_name)

    pred_imgs = {}
    for lbl, folder in pred_folders.items():
        raw_pred = load_raw_slice(folder, volume_idx, slice_name)
        cropped_pred = crop_back(raw_pred)      # (166x256)
        pred256 = resize_to_256(cropped_pred)   # (256x256)
        pred_imgs[lbl] = pred256

    plot_professional_maps(ct_img, cbct_img, pred_imgs, volume_idx, slice_name)
