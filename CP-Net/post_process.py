import os
import cv2
import matplotlib.pyplot as plt
import torch
import cupy as cp
import numpy as np
from PIL import Image
from numba import njit
import torch.nn.functional as F
import cupyx.scipy.ndimage as cpx_ndimage
from scipy.ndimage import label as cc_label
from scipy.ndimage import binary_fill_holes
from torch.utils.dlpack import to_dlpack, from_dlpack
from concurrent.futures import ThreadPoolExecutor
from skimage.segmentation import watershed
from skimage.morphology import h_minima
from skimage.feature import peak_local_max
from concurrent.futures import ThreadPoolExecutor, as_completed

def torch_to_cupy(t: torch.Tensor) -> cp.ndarray:
    """
    torch CUDA tensor -> CuPy array (shares memory via DLPack, no CPU hop).
    t must be on CUDA and contiguous.
    """
    if not t.is_cuda:
        raise ValueError("torch_to_cupy: tensor must be CUDA")
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(t.contiguous()))

def cupy_to_torch(a: cp.ndarray) -> torch.Tensor:
    """
    CuPy array (on GPU) -> torch CUDA tensor (shares memory via DLPack).
    """
    return torch.utils.dlpack.from_dlpack(a.toDlpack())

def build_disk_kernel(radius: int, device: torch.device):
    r = radius
    ys, xs = torch.meshgrid(
        torch.arange(-r, r+1, device=device),
        torch.arange(-r, r+1, device=device),
        indexing="ij"
    )
    disk = ((xs**2 + ys**2) <= (r**2)).float()
    return disk.view(1,1,2*r+1,2*r+1)  # [1,1,K,K]

def gpu_erode_full(bin_masks: torch.Tensor, radius: int):
    """
    Plain morphological erosion (min filter under a disk SE) on the WHOLE mask.
    bin_masks: [B,1,H,W] float/bool CUDA (1 fg, 0 bg)
    returns:  [B,1,H,W] bool CUDA
    """
    device = bin_masks.device
    kernel = build_disk_kernel(radius, device)   # [1,1,K,K]
    k_sum  = kernel.sum()

    x = (bin_masks > 0).float()
    r = radius
    # reflect pad so we don't kill border cells
    x_pad = F.pad(x, (r, r, r, r), mode="reflect")

    conv_out = F.conv2d(
        x_pad, kernel,
        bias=None,
        stride=1,
        padding=0,
        groups=1,
    )
    eroded = (conv_out >= k_sum - 1e-6)
    return eroded.bool()

@torch.no_grad()
def gpu_split_small_med_big_vbig_binary(
    x_bin: torch.Tensor,
    small_thr: int,
    big_thr: int,
    vbig_thr: int,
):
    """
    Split nuclei into 4 size groups (small / medium / big / very big)
    and return pure binary masks per group (no instance IDs).

    Args:
        x_bin: [B,1,H,W] float/bool CUDA tensor. (1 = foreground)
        small_thr: area cutoff for small (≤ small_thr)
        big_thr: area cutoff for big (≥ big_thr)
        vbig_thr: area cutoff for very big (≥ vbig_thr)

    Returns:
        small_batch, med_batch, big_batch, vbig_batch: [B,1,H,W] uint8 CUDA tensors
    """

    if not x_bin.is_cuda:
        raise ValueError("x_bin must be on CUDA")

    B, C, H, W = x_bin.shape
    assert C == 1, "expected [B,1,H,W]"

    small_list, med_list, big_list, vbig_list = [], [], [], []

    for b in range(B):
        # --- 1) to CuPy ---
        mask_b_torch = (x_bin[b, 0] > 0.5).to(torch.uint8)
        mask_b_cu = torch_to_cupy(mask_b_torch)

        # --- 2) connected components ---
        labels_cu, num_comp = cpx_ndimage.label(mask_b_cu)
        if num_comp == 0:
            z = torch.zeros((1, H, W), dtype=torch.uint8, device=x_bin.device)
            small_list.append(z); med_list.append(z.clone()); big_list.append(z.clone()); vbig_list.append(z.clone())
            continue

        # --- 3) compute component areas ---
        flat_lab = labels_cu.ravel()
        areas_cu = cp.bincount(flat_lab, minlength=num_comp + 1)

        # --- 4) classify IDs ---
        small_ids = cp.where((areas_cu > 0) & (areas_cu <= small_thr))[0]
        med_ids   = cp.where((areas_cu > small_thr) & (areas_cu < big_thr))[0]
        big_ids   = cp.where((areas_cu >= big_thr) & (areas_cu < vbig_thr))[0]
        vbig_ids  = cp.where(areas_cu >= vbig_thr)[0]

        # drop background id
        small_ids = small_ids[small_ids != 0]
        med_ids   = med_ids[med_ids != 0]
        big_ids   = big_ids[big_ids != 0]
        vbig_ids  = vbig_ids[vbig_ids != 0]

        # --- 5) build masks with cp.isin ---
        def make_mask(ids):
            if ids.size > 0:
                return cp.isin(labels_cu, ids, assume_unique=False).astype(cp.uint8)
            return cp.zeros_like(labels_cu, dtype=cp.uint8)

        small_bin_cu = make_mask(small_ids)
        med_bin_cu   = make_mask(med_ids)
        big_bin_cu   = make_mask(big_ids)
        vbig_bin_cu  = make_mask(vbig_ids)

        # --- 6) back to torch ---
        small_torch = cupy_to_torch(small_bin_cu).to(torch.uint8).unsqueeze(0)
        med_torch   = cupy_to_torch(med_bin_cu).to(torch.uint8).unsqueeze(0)
        big_torch   = cupy_to_torch(big_bin_cu).to(torch.uint8).unsqueeze(0)
        vbig_torch  = cupy_to_torch(vbig_bin_cu).to(torch.uint8).unsqueeze(0)

        small_list.append(small_torch)
        med_list.append(med_torch)
        big_list.append(big_torch)
        vbig_list.append(vbig_torch)

    # --- 7) stack ---
    small_batch = torch.stack(small_list, dim=0)
    med_batch   = torch.stack(med_list, dim=0)
    big_batch   = torch.stack(big_list, dim=0)
    vbig_batch  = torch.stack(vbig_list, dim=0)

    return small_batch, med_batch, big_batch, vbig_batch
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def to_uint8(x_bool):
    return (x_bool.astype(np.uint8) * 255)




@njit
def _center_of_mass_binary(mask):
    """
    Simple center-of-mass for a binary 2D mask.
    mask: uint8 or bool 2D array, 1 = foreground
    returns: (cy, cx) as float64
    """
    h, w = mask.shape
    s = 0.0
    sy = 0.0
    sx = 0.0
    for y in range(h):
        for x in range(w):
            if mask[y, x] != 0:
                v = 1.0
                s  += v
                sy += y * v
                sx += x * v
    if s == 0.0:
        return -1.0, -1.0
    return sy / s, sx / s


@njit
def _hv_inner(inst_map):
    """
    inst_map: int32 2D array, 0 = background, 1..N = instances
    Returns:
      H_map, V_map: float32 2D arrays in [-1,1] on instance pixels, 0 on background.
    """
    H, W = inst_map.shape
    H_map = np.zeros((H, W), np.float32)
    V_map = np.zeros((H, W), np.float32)

    max_id = int(inst_map.max())
    for nid in range(1, max_id + 1):
        # build mask for this id
        m = (inst_map == nid)
        if not m.any():
            continue

        # bounding box
        h, w = inst_map.shape
        rmin, rmax = h, -1
        cmin, cmax = w, -1
        for y in range(h):
            for x in range(w):
                if m[y, x]:
                    if y < rmin: rmin = y
                    if y > rmax: rmax = y
                    if x < cmin: cmin = x
                    if x > cmax: cmax = x
        if rmax < rmin or cmax < cmin:
            continue

        rmax += 1
        cmax += 1

        crop_h = rmax - rmin
        crop_w = cmax - cmin
        if crop_h < 2 or crop_w < 2:
            continue

        # copy crop into a small mask
        crop = np.zeros((crop_h, crop_w), np.uint8)
        for yy in range(crop_h):
            for xx in range(crop_w):
                if m[rmin + yy, cmin + xx]:
                    crop[yy, xx] = 1

        # center of mass
        cy_f, cx_f = _center_of_mass_binary(crop)
        if cy_f < 0:
            continue
        cy = int(round(cy_f))
        cx = int(round(cx_f))

        # coordinate grids (XX: horizontal, YY: vertical)
        XX = np.zeros((crop_h, crop_w), np.float32)
        YY = np.zeros((crop_h, crop_w), np.float32)
        for yy in range(crop_h):
            for xx in range(crop_w):
                if crop[yy, xx] != 0:
                    YY[yy, xx] = float(yy - cy)
                    XX[yy, xx] = float(xx - cx)

        # normalize XX to [-1,1]
        neg_min_x = 0.0
        pos_max_x = 0.0
        for yy in range(crop_h):
            for xx in range(crop_w):
                val = XX[yy, xx]
                if crop[yy, xx] != 0:
                    if val < 0.0 and val < neg_min_x:
                        neg_min_x = val
                    if val > 0.0 and val > pos_max_x:
                        pos_max_x = val
        if neg_min_x < 0.0:
            scale = -neg_min_x
            for yy in range(crop_h):
                for xx in range(crop_w):
                    if crop[yy, xx] != 0 and XX[yy, xx] < 0.0:
                        XX[yy, xx] /= scale
        if pos_max_x > 0.0:
            scale = pos_max_x
            for yy in range(crop_h):
                for xx in range(crop_w):
                    if crop[yy, xx] != 0 and XX[yy, xx] > 0.0:
                        XX[yy, xx] /= scale

        # normalize YY to [-1,1]
        neg_min_y = 0.0
        pos_max_y = 0.0
        for yy in range(crop_h):
            for xx in range(crop_w):
                val = YY[yy, xx]
                if crop[yy, xx] != 0:
                    if val < 0.0 and val < neg_min_y:
                        neg_min_y = val
                    if val > 0.0 and val > pos_max_y:
                        pos_max_y = val
        if neg_min_y < 0.0:
            scale = -neg_min_y
            for yy in range(crop_h):
                for xx in range(crop_w):
                    if crop[yy, xx] != 0 and YY[yy, xx] < 0.0:
                        YY[yy, xx] /= scale
        if pos_max_y > 0.0:
            scale = pos_max_y
            for yy in range(crop_h):
                for xx in range(crop_w):
                    if crop[yy, xx] != 0 and YY[yy, xx] > 0.0:
                        YY[yy, xx] /= scale

        # write H,V to global maps
        for yy in range(crop_h):
            for xx in range(crop_w):
                if crop[yy, xx] != 0:
                    gy = rmin + yy
                    gx = cmin + xx
                    H_map[gy, gx] = XX[yy, xx]
                    V_map[gy, gx] = YY[yy, xx]

    return H_map, V_map

def hv_from_instances(inst_map: np.ndarray):
    """
    Thin wrapper around the Numba-accelerated _hv_inner.

    inst_map: (H,W) int32 array, 0=background, 1..N=instances
    returns: H_map, V_map in [-1,1], shape (H,W)
    """
    inst_map_int = inst_map.astype(np.int32, copy=False)
    H_map, V_map = _hv_inner(inst_map_int)
    return H_map, V_map




def fuse_horizontal_vertical(oriented: np.ndarray,
                             angles: np.ndarray,
                             degrees: bool = False,
                             half_width: float = None):
    """
    oriented: (K,H,W) maps in [-1,1]
    angles : (K,) angles for those maps; radians unless degrees=True
    returns: H_fused, V_fused  -> (H,W)
    """
    if oriented.ndim != 3:
        raise ValueError("oriented must be (K,H,W)")
    K, H, W = oriented.shape

    if angles is None:
        raise ValueError("angles must be provided")
    angles = np.asarray(angles, dtype=np.float32)
    if angles.shape[0] != K:
        raise ValueError(f"angles length {angles.shape[0]} != K {K}")

    if degrees:
        angles = np.deg2rad(angles)

    # normalize to [0, π)
    angles = np.mod(angles, np.pi)

    # default ±45°
    if half_width is None:
        half_width = np.pi / 4.0

    # horizontal near 0 (and π modulo π)
    horiz_mask = (angles <= half_width) | (angles >= (np.pi - half_width))
    # vertical near π/2
    vert_mask  = np.abs(angles - (np.pi / 2.0)) <= half_width

    H_fused = oriented[horiz_mask].sum(axis=0) if np.any(horiz_mask) else np.zeros((H, W), oriented.dtype)
    V_fused = oriented[vert_mask].sum(axis=0)  if np.any(vert_mask)  else np.zeros((H, W), oriented.dtype)

    # normalize to [-1, 1] (safe if zeros)
    mH = np.max(np.abs(H_fused))
    mV = np.max(np.abs(V_fused))
    if mH > 0: H_fused = H_fused / mH
    if mV > 0: V_fused = V_fused / mV

    return H_fused.astype(np.float32), V_fused.astype(np.float32)



def postprocess_instances_from_hv_sobel(
    prob_map,     # (H,W) float [0..1] nuclei prob
    H_map,        # (H,W) float [-1..1]
    V_map,        # (H,W) float [-1..1]
    min_size=5,
    ksize=21,
    g_ksize=(3,3),
    thr=0.55):

    # threshold nuclei probability
    blb = (prob_map >= 0.5).astype(np.int32)

    # label + size filter
    #blb_labeled, _ = cc_label(blb)
    blb_labeled = blb

    #blb_labeled = remove_small_objects(blb_labeled, min_size=min_size)
    blb_bin = (blb_labeled > 0).astype(np.int32)

    # normalize H_map, V_map separately to [0,1]
    h_dir = cv2.normalize(
        H_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        V_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

    sobelh = 1 - cv2.normalize(
        sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    sobelv = 1 - cv2.normalize(
        sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb_bin)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb_bin
    dist = -cv2.GaussianBlur(dist, g_ksize, 0)  # basins for watershed

    overall_bin = (overall >= thr).astype(np.int32)
    marker = blb_bin - overall_bin
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)

    marker_lbl, _ = cc_label(marker)
    #marker_lbl = remove_small_objects(marker_lbl, min_size=object_size)
    final_inst = watershed(dist, markers=marker_lbl, mask=blb_bin).astype(np.int32)

    return final_inst


@torch.no_grad()
def gpu_label_connected_components(bin_mask_t: torch.Tensor):
    """
    GPU connected components labeling.

    Args:
        bin_mask_t: [B,1,H,W] torch.bool or float CUDA tensor.
                    (1 = foreground, 0 = background)

    Returns:
        labels_t: [B,1,H,W] int32 CUDA tensor with unique IDs per connected region.
        num_components: list[int] number of components per image.
    """
    if not bin_mask_t.is_cuda:
        raise ValueError("Input must be CUDA tensor")

    B, C, H, W = bin_mask_t.shape
    assert C == 1, "Expected [B,1,H,W]"

    label_list = []
    num_list = []

    for b in range(B):
        # Convert to CuPy (uint8)
        mask_cu = torch_to_cupy((bin_mask_t[b, 0] > 0.5).to(torch.uint8))
        # Label on GPU
        labels_cu, num = cpx_ndimage.label(mask_cu)
        # Convert back to torch int32
        labels_t = cupy_to_torch(labels_cu).to(torch.int32).unsqueeze(0)
        label_list.append(labels_t)
        num_list.append(int(num))

    labels_batch = torch.stack(label_list, dim=0)  # [B,1,H,W]
    return labels_batch, num_list


def markers_hmin(dist_smooth: np.ndarray, blb_u8: np.ndarray, h_min: float):
    minima = h_minima(-dist_smooth, h=float(h_min))
    markers, _ = cc_label(minima.astype(np.uint8))
    if markers.max() == 0:
        markers, _ = cc_label(blb_u8)
    return markers.astype(np.int32)

def markers_peakmax(dist_smooth: np.ndarray, blb_u8: np.ndarray, min_distance: int, threshold_abs: float):
    coords = peak_local_max(
        dist_smooth,
        labels=blb_u8,
        min_distance=int(min_distance),
        threshold_abs=float(threshold_abs),
        exclude_border=False,
    )
    markers = np.zeros_like(blb_u8, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    if markers.max() == 0:
        markers, _ = cc_label(blb_u8)
    return markers.astype(np.int32)




def distance_watershed_from_binary(
    blb_bin: np.ndarray,              # (H,W) 0/1
    g_ksize=(3, 3),
    marker_method="hmin",             # "hmin" or "peak"
    h_min=2.0,                        # for hmin method
    peak_min_distance=6,              # for peak method
    peak_threshold_abs=0.0,           # for peak method
):
    blb_u8 = (blb_bin > 0).astype(np.uint8)
    if blb_u8.max() == 0:
        H, W = blb_u8.shape
        return np.zeros((H, W), np.int32), np.zeros((H, W), np.float32), np.zeros((H, W), np.int32)

    dist = cv2.distanceTransform(blb_u8, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
    dist_smooth = cv2.GaussianBlur(dist, g_ksize, 0).astype(np.float32)

    if marker_method == "peak":
        markers = markers_peakmax(dist_smooth, blb_u8, peak_min_distance, peak_threshold_abs)
    else:
        markers = markers_hmin(dist_smooth, blb_u8, h_min)

    inst_final = watershed(-dist_smooth, markers=markers, mask=(blb_u8 > 0)).astype(np.int32)
    return inst_final, dist_smooth, markers


@torch.no_grad()
def remove_small_instances_gpu(inst_t: torch.Tensor, min_area: int):
    if min_area <= 0:
        return inst_t

    B, C, H, W = inst_t.shape
    assert C == 1

    out_list = []
    for b in range(B):
        inst_cu = torch_to_cupy(inst_t[b, 0].to(torch.int32))
        max_id = int(inst_cu.max())
        if max_id <= 0:
            out_list.append(inst_t[b:b+1])
            continue

        counts = cp.bincount(inst_cu.ravel(), minlength=max_id + 1)
        keep = counts >= int(min_area)
        keep[0] = True  # background

        cleaned = inst_cu * keep[inst_cu].astype(inst_cu.dtype)
        out_list.append(cupy_to_torch(cleaned).to(torch.int32).unsqueeze(0).unsqueeze(0))

    return torch.cat(out_list, dim=0)



def process_mask_batch_dist_map(
    Batch,                          # [B,1,H,W] float32 CUDA (0/1)
    erode_r=1,                      # erosion once
    gsize=(3, 3),
    marker_method="hmin",           # "hmin" | "peak"
    h_min=2.0,
    peak_min_distance=6,
    peak_threshold_abs=0.0,
    min_area=10,
    num_workers=0,                  # 0/1 => no parallel, >1 => parallel
):
    """
    Parallelizes ONLY the CPU part (distance + markers + watershed) using threads.
    GPU erosion + GPU small-instance filtering remain on the main thread.
    """

    # 1) GPU erosion once
    eroded = gpu_erode_full(Batch, erode_r)  # bool CUDA

    B = Batch.shape[0]
    bin_list = [
        eroded[i, 0].detach().cpu().numpy().astype(np.uint8)
        for i in range(B)
    ]

    def _cpu_worker(bin_np):
        inst_np, dist_np, _ = distance_watershed_from_binary(
            bin_np,
            g_ksize=gsize,
            marker_method=marker_method,
            h_min=h_min,
            peak_min_distance=peak_min_distance,
            peak_threshold_abs=peak_threshold_abs,
        )
        return inst_np.astype(np.int32), dist_np.astype(np.float32)

    # 2) CPU threaded section (keeps output order)
    if num_workers is None or int(num_workers) <= 1:
        results = [_cpu_worker(bin_list[i]) for i in range(B)]
    else:
        nw = min(int(num_workers), B)
        results = [None] * B
        with ThreadPoolExecutor(max_workers=nw) as ex:
            future_to_idx = {ex.submit(_cpu_worker, bin_list[i]): i for i in range(B)}
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                results[i] = fut.result()

    # 3) Back to GPU tensors
    inst_out = []
    for inst_np, dist_np in results:
        inst_out.append(torch.from_numpy(inst_np).to(dtype=torch.int32, device=Batch.device).unsqueeze(0))

    inst_final_batch = torch.stack(inst_out, dim=0)  # [B,1,H,W]

    # 4) GPU remove tiny instances
    inst_final_batch = remove_small_instances_gpu(inst_final_batch, min_area=min_area)
    return inst_final_batch



def process_mask_batch_HV_map(
        Batch,
        med_erode_r=3,
        big_erode_r=4,
        vbig_erode_r=5,
        w_thr=0.5,
        ksize=21,
        gsize=(3,3),
        small_thr=150,
        big_thr=250,
        vbig_thr=500,
        num_workers=1  # 0/1 = no parallel, >1 = use threads
    ):

    # --- GPU size split + erosion (same as before) ---

    small_batch, med_batch, big_batch, vbig_batch = gpu_split_small_med_big_vbig_binary(
        Batch,
        small_thr=small_thr,
        big_thr=big_thr,
        vbig_thr=vbig_thr,
    )
    small_batch = small_batch.bool()
    med_batch_erod = gpu_erode_full(med_batch, med_erode_r)
    big_batch_erod = gpu_erode_full(big_batch, big_erode_r)
    vbig_batch_erod = gpu_erode_full(vbig_batch, vbig_erode_r)


    eroded_small_med_big = small_batch + med_batch_erod + big_batch_erod + vbig_batch_erod

    eroded_small_med_big, num_components = gpu_label_connected_components(eroded_small_med_big)

    B = Batch.shape[0]

    # --- Move per-sample data to CPU as NumPy arrays ---
    orig_list   = []
    eroded_list = []
    for i in range(B):
        # Batch: [B,1,H,W] or [B,H,W]? Your code used Batch[i] and eroded[i,0]
        orig_mask_np = Batch[i, 0].detach().cpu().numpy()  # [H,W]
        eroded_np    = eroded_small_med_big[i, 0].detach().cpu().numpy().astype(np.uint8)
        orig_list.append(orig_mask_np)
        eroded_list.append(eroded_np)

    # --- Process each mask, optionally in parallel on CPU ---
    inst_final_batch = []

    if num_workers is None or num_workers <= 1:
        # Sequential path (original behavior)
        for i in range(B):
            inst_final = _process_single_mask_np(
                orig_list[i],
                eroded_list[i],
                w_thr,
                ksize,
                gsize,
            )
            inst_final_t = torch.from_numpy(inst_final).to(
                dtype=torch.int32,
                device=Batch.device
            )
            inst_final_batch.append(inst_final_t.unsqueeze(0))  # [1,H,W]
    else:
        # Parallel path
        # Reasonable default: min(num_workers, B)
        nw = min(num_workers, B)
        with ThreadPoolExecutor(max_workers=nw) as ex:
            futures = [
                ex.submit(
                    _process_single_mask_np,
                    orig_list[i],
                    eroded_list[i],
                    w_thr,
                    ksize,
                    gsize,
                )
                for i in range(B)
            ]
            for f in futures:
                inst_final = f.result()
                inst_final_t = torch.from_numpy(inst_final).to(
                    dtype=torch.int32,
                    device=Batch.device
                )
                inst_final_batch.append(inst_final_t.unsqueeze(0))

    # --- Stack into [B,1,H,W] ---
    inst_final_batch = torch.stack(inst_final_batch, dim=0)

    return inst_final_batch

def _process_single_mask_np(orig_mask_np,
                            eroded_np,
                            w_thr,
                            ksize,
                            gsize):
    """
    CPU worker for one mask: compute H/V maps + watershed postprocess.
    All inputs are NumPy arrays on CPU.
    Returns inst_final as np.int32 [H,W].
    """
    # H, V from your (Numba-accelerated) hvd_from_instances
    H_map, V_map = hv_from_instances(eroded_np)

    prob_map = orig_mask_np.astype(np.float32)

    inst_final = postprocess_instances_from_hv_sobel(
        prob_map,
        H_map,
        V_map,
        min_size=5,
        ksize=ksize,
        g_ksize=gsize,
        thr=w_thr
    )
    return inst_final.astype(np.int32)


# If not already defined above, keep these helpers:

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def to_uint8(x_bool):
    return (x_bool.astype(np.uint8) * 255)

def save_heatmap(arr, fname, vmin=-1, vmax=1, cmap="coolwarm"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,4))
    plt.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

def colorize_instances(inst_map: np.ndarray, seed: int = 1337):
    """
    inst_map: (H,W) int32, 0 = background, 1..N = nuclei ids
    returns RGB uint8 for visualization
    """
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids != 0]
    H, W = inst_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    if len(inst_ids) == 0:
        return rgb
    max_id = int(inst_ids.max())
    rng = np.random.default_rng(seed)
    colors = rng.integers(low=30, high=255, size=(max_id + 1, 3), dtype=np.uint8)
    colors[0] = (0, 0, 0)
    rgb = colors[inst_map]
    return rgb




def quick_test_on_masks(
    mask_dir,
    out_dir="out_test",
    med_erode_r=2,
    big_erode_r=3,
    vbig_erode_r=4,
    w_thr=0.6,
    ksize=15,
    g_ksize= (3,3),
    small_thr=150,
    big_thr=250,
    vbig_thr=350,
    device=None,
):
    """
    Test pipeline on a few masks, and save:
      - original mask
      - final eroded mask (binary)
      - H map
      - V map
      - colored instance map
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device != "cuda":
        raise RuntimeError("This pipeline currently assumes CUDA (CuPy + torch.cuda).")

    # collect .png masks
    mask_paths = [
        os.path.join(mask_dir, f)
        for f in os.listdir(mask_dir)
        if f.lower().endswith(".png")
    ]
    mask_paths.sort()

    if len(mask_paths) == 0:
        raise RuntimeError(f"No PNG masks found in {mask_dir}")

    print(f"[quick_test_on_masks] Using device={device}")
    print(f"[quick_test_on_masks] Found {len(mask_paths)} masks:")
    for p in mask_paths:
        print("   ", os.path.basename(p))

    ensure_dir(out_dir)

    # --- load masks into a NumPy batch [B,H,W] ---
    imgs_np = []
    bases   = []
    for p in mask_paths:
        im = Image.open(p).convert("L")
        arr = np.array(im)
        arr_bin = (arr > 0).astype(np.float32)   # binary 0/1
        imgs_np.append(arr_bin)
        bases.append(os.path.splitext(os.path.basename(p))[0])

    batch_np = np.stack(imgs_np, axis=0)              # [B,H,W]
    Batch = torch.from_numpy(batch_np).unsqueeze(1)   # [B,1,H,W]
    Batch = Batch.to(device=device, dtype=torch.float32)

    # --- GPU size split + erosion (same as in process_mask_batch) ---
    small_batch, med_batch, big_batch, vbig_batch = gpu_split_small_med_big_vbig_binary(
        Batch,
        small_thr=small_thr,
        big_thr=big_thr,
        vbig_thr=vbig_thr
    )

    small_batch = small_batch.bool()
    med_batch_erod = gpu_erode_full(med_batch, med_erode_r)
    big_batch_erod = gpu_erode_full(big_batch, big_erode_r)
    vbig_batch_erod = gpu_erode_full(vbig_batch, vbig_erode_r)

    eroded_small_med_big = small_batch + med_batch_erod + big_batch_erod + vbig_batch_erod

    eroded_small_med_big, num_components = gpu_label_connected_components(
        eroded_small_med_big
    )

    B = Batch.shape[0]

    for i in range(B):
        base = bases[i]
        out_sub = os.path.join(out_dir, base)
        ensure_dir(out_sub)

        # original mask [H,W], 0/1 float
        orig_mask_np = batch_np[i]

        # eroded label mask [H,W] int
        eroded_np = eroded_small_med_big[i, 0].detach().cpu().numpy().astype(np.int32)

        # H, V maps from labeled erosion result
        H_map, V_map = hv_from_instances(eroded_np)

        prob_map = orig_mask_np.astype(np.float32)

        inst_final = postprocess_instances_from_hv_sobel(
            prob_map,
            H_map,
            V_map,
            min_size=5,
            ksize=ksize,
            g_ksize=g_ksize,
            thr=w_thr,
        )

        # ---- Save outputs ----

        # 0) original binary mask
        orig_u8 = (orig_mask_np > 0).astype(np.uint8) * 255
        Image.fromarray(orig_u8).save(os.path.join(out_sub, f"{base}_0_orig.png"))

        # 1) eroded binary mask (foreground = eroded labels > 0)
        eroded_bin = (eroded_np > 0).astype(np.uint8) * 255
        Image.fromarray(eroded_bin).save(os.path.join(out_sub, f"{base}_1_eroded.png"))

        # 2) H and 3) V heatmaps
        save_heatmap(H_map, os.path.join(out_sub, f"{base}_2_Hmap.png"),
                     vmin=-1, vmax=1, cmap="coolwarm")
        save_heatmap(V_map, os.path.join(out_sub, f"{base}_3_Vmap.png"),
                     vmin=-1, vmax=1, cmap="coolwarm")

        # 4) colored instance map
        inst_rgb = colorize_instances(inst_final.astype(np.int32))
        Image.fromarray(inst_rgb).save(os.path.join(out_sub, f"{base}_4_instances_color.png"))

        print(f"[quick_test_on_masks] Saved outputs for {base} in {out_sub}")

@torch.no_grad()
def assign_type_to_instances(
    inst_batch: torch.Tensor,   # [B,1,H,W] int32, separated nuclei
    pred_type: torch.Tensor,    # [B,C,H,W] logits
    ignore_bg_class: bool = True,
):
    B, _, H, W = inst_batch.shape
    C = pred_type.shape[1]

    pix_cls = pred_type.argmax(dim=1)  # [B,H,W]

    typed_inst_batch = torch.zeros_like(inst_batch, dtype=torch.int32)
    inst_info_list = []

    for b in range(B):
        inst = inst_batch[b, 0]
        cls_map = pix_cls[b]

        inst_ids = torch.unique(inst)
        inst_ids = inst_ids[inst_ids != 0]

        inst_info = {}

        for iid in inst_ids.tolist():
            m = inst == iid
            ys, xs = torch.where(m)

            if ys.numel() == 0:
                continue

            cy = int(torch.round(ys.float().mean()).item())
            cx = int(torch.round(xs.float().mean()).item())

            cy = max(0, min(cy, H - 1))
            cx = max(0, min(cx, W - 1))

            t = int(cls_map[cy, cx].item())

            if ignore_bg_class and t == 0 and C > 1:
                cls_vals = cls_map[m]
                counts = torch.bincount(cls_vals, minlength=C)
                counts[0] = 0
                t = int(torch.argmax(counts).item())

            inst_info[int(iid)] = {
                "centroid": (cx, cy),
                "type": t,
            }

            typed_inst_batch[b, 0][m] = t

        inst_info_list.append(inst_info)

    return typed_inst_batch, inst_info_list


if __name__ == "__main__":
    import time

    # ✅ Customize these before running
    MASK_DIR  = "20x"              # folder containing your binary .png masks
    OUT_DIR   = "out_20x_test"     # output folder for results

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running quick test on device: {device}")

    start = time.time()

    quick_test_on_masks(
        mask_dir=MASK_DIR,
        out_dir=OUT_DIR,
        med_erode_r=3,
        big_erode_r=4,
        vbig_erode_r=5,
        w_thr=0.6,
        ksize=15,
        g_ksize = (3,3),
        small_thr=100,
        big_thr=400,
        vbig_thr=600,
        device=device,
    )
    end = time.time()
    print(f"✅ Test completed successfully in {end - start:.2f} seconds.")