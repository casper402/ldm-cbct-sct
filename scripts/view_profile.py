import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.measure import profile_line
from matplotlib.ticker import MultipleLocator

try:
    plt.style.use("seaborn-v0_8-paper")
    print("using style")
except OSError:
    print("falling back")
    plt.style.use("ggplot")

plt.rcParams["font.family"]    = "serif"
plt.rcParams["font.serif"]     = ["Nimbus Roman No9 L"]
plt.rcParams["font.size"]      = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["legend.fontsize"]= 10
plt.rcParams["xtick.labelsize"]= 10
plt.rcParams["ytick.labelsize"]= 10


DATA_RANGE = 2000.0  # HU range for SSIM computations (–1000…+1000)

ORIG_H, ORIG_W = 238, 366
PAD_L, PAD_T, PAD_R, PAD_B = 0, 64, 0, 64
RES_H, RES_W = 256, 256

_pad_h = ORIG_H + PAD_T + PAD_B  # = 366
_pad_w = ORIG_W + PAD_L + PAD_R  # = 366

TOP_CROP    = int(round((PAD_T / _pad_h) * RES_H))    # = 45
BOTTOM_CROP = int(round((PAD_B / _pad_h) * RES_H))    # = 45
LEFT_CROP   = int(round((PAD_L / _pad_w) * RES_W))    #   0
RIGHT_CROP  = int(round((PAD_R / _pad_w) * RES_W))    #   0

transform = transforms.Compose([
    transforms.Pad((PAD_L, PAD_T, PAD_R, PAD_B), fill=-1000),
    transforms.Resize((RES_H, RES_W)),
])

def apply_transform(np_img: np.ndarray) -> np.ndarray:
    """
    Pad a raw 238×366 slice to 366×366 (fill=-1000), then resize to 256×256.
    Returns a 256×256 numpy array.
    """
    pil = Image.fromarray(np_img)
    out = transform(pil)       
    return np.array(out)       

def crop_back(arr: np.ndarray) -> np.ndarray:
    """
    Crop the padded+resized array (256×256) to the central 166×256 region,
    removing ~45 px top and bottom.
    """
    return arr[
        TOP_CROP : RES_H - BOTTOM_CROP,   
        LEFT_CROP: RES_W - RIGHT_CROP     
    ]

def resize_to_256(slc: np.ndarray) -> np.ndarray:
    """
    Resize any 2D NumPy array (e.g. 166×256) to exactly 256×256 via PIL bilinear,
    but keep it in float32 so negative HU never wraps to unsigned int.
    """
    slc_f32 = slc.astype(np.float32)
    pil = Image.fromarray(slc_f32, mode='F')
    resized_pil = pil.resize((256, 256), resample=Image.BILINEAR)
    return np.array(resized_pil, dtype=np.float32)

def load_raw_slice(folder: str, volume_idx: int, slice_name: str) -> np.ndarray:
    """
    Attempt to load folder/volume-<volume_idx>/<slice_name>.npy or folder/<slice_name>.npy.
    Returns the raw NumPy array (shape ~238×366 or 256×256), flipped left-right.
    Raises FileNotFoundError if not found.
    """
    candidate1 = os.path.join(folder, f"volume-{volume_idx}", slice_name)
    if os.path.isfile(candidate1):
        return np.fliplr(np.load(candidate1))
    candidate2 = os.path.join(folder, slice_name)
    if os.path.isfile(candidate2):
        return np.fliplr(np.load(candidate2))
    raise FileNotFoundError(
        f"Could not find {slice_name} in {folder!r} "
        f"(checked both volume-{volume_idx}/{slice_name} and {slice_name})."
    )

def load_and_process_CT_CBCT_slice(
    folder: str,
    volume_idx: int,
    slice_name: str
) -> np.ndarray:
    """
    1) Load raw CT/CBCT slice (~238×366) from either folder/volume-<volume_idx>/<slice_name>
       or folder/<slice_name>.
    2) Apply pad→resize (→256×256), crop back (→166×256), then final resize (→256×256).
    Returns a 256×256 NumPy array (HU).
    """
    raw = load_raw_slice(folder, volume_idx, slice_name)   
    padded = apply_transform(raw)                           
    cropped = crop_back(padded)                           
    final256 = resize_to_256(cropped)                      
    return final256


dx = 0.0
def plot_profile_horizontal(
    slice_name,
    y_coord,
    ct_img: np.ndarray,
    cbct_img: np.ndarray,
    preds: dict,
    volume
):
    """
    Draw a horizontal HU‐profile (CT, CBCT, and each pred) at row y.
    The main plot has no title; inset is down in the bottom‐right corner
    (no inset title or y‐annotation). We drop 'sCT (CycleGAN)' and
    assign its color/linestyle to 'sCT (ours)' instead.
    
    Changes implemented:
      1. No main‐plot title.
      2. Do not plot "sCT (CycleGAN)" at all.
      3. Use CycleGAN’s red & dotted style for "sCT (ours)".
      4. Only vertical grid lines, made slightly bolder/less transparent.
      5. Inset has no title and no "y=<…>" label.
      6. Inset moved to bottom‐right.
    """
    H, W = ct_img.shape
    if y_coord is None:
        y = random.randint(0, H - 1)
    else:
        y = int(np.clip(y_coord, 0, H - 1))

    profs = {
        "CT":   profile_line(ct_img,   (y, 0), (y, W - 1), mode="constant", cval=np.nan),
        "CBCT": profile_line(cbct_img, (y, 0), (y, W - 1), mode="constant", cval=np.nan),
    }
    for lbl, img in preds.items():
        profs[lbl] = profile_line(img, (y, 0), (y, W - 1), mode="constant", cval=np.nan)

    profs.pop("sCT (CycleGAN)", None)

    COLORS = {
        "CT":           "#1f77b4",  # muted blue
        "CBCT":         "#ff7f0e",  # orange
        "sCT":   "#d62728",  # red
    }
    STYLES = {
        "CT":           {"linestyle":"-",  "linewidth":3.0},
        "CBCT":         {"linestyle":":", "linewidth":3.0},
        "sCT":   {"linestyle":"--",  "linewidth":3.0}, 
    }

    fig = plt.figure(figsize=(12, 5))
    ax_prof = fig.add_axes([0.07, 0.12, 0.88, 0.80])  # [left, bottom, width, height]

    global dx
    ax_img = fig.add_axes([0.632, 0.12, 0.45, 0.45]) 
    print("dx:", dx)
    dx += 0.001
    ax_img.imshow(ct_img, cmap="gray", vmin=-1000, vmax=1000)
    ax_img.axhline(y=y, color="red", linewidth=2)

    ax_img.axis("off")

    for lbl, prof in profs.items():
        ax_prof.plot(
            prof,
            label=lbl,
            color=COLORS.get(lbl, "k"),
            **STYLES.get(lbl, {"linestyle":"-", "linewidth":1.0})
        )

    ax_prof.spines["top"].set_visible(False)
    ax_prof.spines["right"].set_visible(False)

    ax_prof.set_xlim(0, W - 1)
    ax_prof.set_xlabel("Column (pixel index)", fontsize=18)
    ax_prof.set_ylabel("Hounsfield units (HU)",   fontsize=18)

    ax_prof.xaxis.set_major_locator(MultipleLocator(50))
    ax_prof.yaxis.set_major_locator(MultipleLocator(200))
    ax_prof.minorticks_off()

    ax_prof.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)
    ax_prof.xaxis.grid(False)

    ax_prof.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(profs),
        frameon=False,
        fontsize=16
    )

    plt.show()
    save_path = os.path.expanduser(f"~/thesis/figures/profile_vol{volume}_{slice_name[:-4]}.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    print("Saved to:", save_path)


if __name__ == "__main__":
    volume_idx = 8
    slice_idx = 107
    slice_name = f"volume-{volume_idx}_slice_{slice_idx}.npy"

    gt_folder   = os.path.expanduser("~/thesis/training_data/CT/test")
    cbct_folder = os.path.expanduser("~/thesis/training_data/CBCT/490/test")
    pred_folders = {
        "sCT":     os.path.expanduser(
            "~/thesis/predictions/predictions_controlnet_v7-data-augmentation"
        ),
        "sCT (CycleGAN)": os.path.expanduser(
            "~/thesis/predictions/predictions_tanh_v5"
        ),
    }

    ct_img   = load_and_process_CT_CBCT_slice(gt_folder, volume_idx, slice_name)
    cbct_img = load_and_process_CT_CBCT_slice(cbct_folder, volume_idx, slice_name)

    pred_imgs = {}
    for lbl, folder in pred_folders.items():
        raw_pred = load_raw_slice(folder, volume_idx, slice_name)
        cropped_pred = crop_back(raw_pred)      
        pred256 = resize_to_256(cropped_pred)   
        pred_imgs[lbl] = pred256

    while True:
        plot_profile_horizontal(
            slice_name=slice_name,
            y_coord=200,           # or None for random
            ct_img=ct_img,
            cbct_img=cbct_img,
            preds=pred_imgs,
            volume=volume_idx
        )
