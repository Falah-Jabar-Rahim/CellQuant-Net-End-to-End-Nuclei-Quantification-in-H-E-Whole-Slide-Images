import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import yaml
import re
import torch
import numpy as np
from tqdm import tqdm
import os, time, argparse, json
from PIL import Image, ImageDraw
import torch.nn.functional as F
from types import SimpleNamespace
from model.model import CellPriorNet
from dataset import make_np_loader
from cell_connectivitey import cell_neighborhood
from qupath_jsn import parse_tile_xy, make_qupath_cell_feature, save_qupath_geojson
from post_process import process_mask_batch_dist_map, assign_type_to_instances

def save_classify_dots(args, img_t_rgb3, inst_map, inst_info, out_prefix, dot_radius=3):
    """
    Save:
      1) RGB image with class-colored centroid dots
      2) class-colored instance mask

    Args:
        img_t_rgb3: [3,H,W]
        inst_map: [H,W] separated instance map
        inst_info: dict from assign_type_to_instances:
                   {inst_id: {"centroid": (cx, cy), "type": class_id}}
    """
    mean = np.array(args.imagenet_mean, dtype=np.float32)
    std  = np.array(args.imagenet_std, dtype=np.float32)

    img = img_t_rgb3.permute(1, 2, 0).detach().cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img_uint8 = (img * 255).astype(np.uint8)

    inst_np = inst_map.detach().cpu().numpy().astype(np.int32)

    class_colors = {int(k): tuple(v) for k, v in args.class_colors.items()}
    default_color = (255, 255, 0)

    # ---------- 1) dots ----------
    pil_img = Image.fromarray(img_uint8)
    draw = ImageDraw.Draw(pil_img)

    for inst_id, info in inst_info.items():
        cx, cy = info["centroid"]
        ctype = int(info["type"])
        color = class_colors.get(ctype, default_color)

        draw.ellipse(
            [(cx - dot_radius, cy - dot_radius),
             (cx + dot_radius, cy + dot_radius)],
            outline=color,
            fill=color,
        )

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    pil_img.save(out_prefix + "_dots.png")

    # ---------- 2) class-colored instance mask ----------
    H, W = inst_np.shape
    mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)

    for inst_id, info in inst_info.items():
        ctype = int(info["type"])
        color = class_colors.get(ctype, default_color)
        mask_rgb[inst_np == int(inst_id)] = color

    if args.save_seg:
        Image.fromarray(mask_rgb).save(out_prefix + "_mask.png")

    # return {
    #     "dots": out_prefix + "_dots.png",
    #     "mask": out_prefix + "_mask.png",
    # }

def _prepare_inputs(batch, device):
    """
    Returns:
      rgb3: [B,3,H,W]
      h1:   [B,1,H,W] or None
      mode: "pair" if provided separately, "split4" if from 4-ch tensor, "rgb_only" if only 3-ch
    """
    if "image_rgb" in batch and "image_h" in batch:
        rgb3 = batch["image_rgb"].to(device, non_blocking=True)
        h1   = batch["image_h"].to(device, non_blocking=True)
        return rgb3, h1, "pair"
    imgs = batch["image"].to(device, non_blocking=True)
    if imgs.shape[1] == 4:
        return imgs[:, :3], imgs[:, 3:4], "split4"
    if imgs.shape[1] == 3:
        return imgs, None, "rgb_only"
    raise ValueError(f"Unexpected channels in batch['image']: {imgs.shape[1]}")

def _forward_flexible(model, rgb3, h1):

    """
    Always make sure the model gets 4 channels.
    If H is missing, create a zero H channel.
    """
    if h1 is None:
        h1 = torch.zeros_like(rgb3[:, :1])  # [B,1,H,W]
    try:
        # If model supports two-input API
        return model(rgb3, h1)

    except TypeError:
        # If model supports single 4-channel input
        x4 = torch.cat([rgb3, h1], dim=1)   # [B,4,H,W]

        return model(x4)

def infer_magnification_from_ckpt_name(ckpt_path):
    name = os.path.basename(ckpt_path).lower()

    if re.search(r'(^|[_\-])40x([_\-\.]|$)', name) or "40x" in name:
        return "40x"
    if re.search(r'(^|[_\-])20x([_\-\.]|$)', name) or "20x" in name:
        return "20x"

    raise ValueError(
        f"Could not infer magnification from checkpoint filename: {name}\n"
        f"Please include '20x' or '40x' in the checkpoint name, e.g. "
        f"'last_40x.pth' or 'best_20x.pth'."
    )

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def infer_magnification(magnification, mpp_x):
    if pd.notna(magnification) and isinstance(magnification, (int, float)):
        return f"{int(magnification)}x"

    if pd.notna(mpp_x):
        mpp_x = float(mpp_x)

        if mpp_x <= 0.30:
            return "40x"
        elif mpp_x <= 0.60:
            return "20x"

    raise ValueError(
        f"Cannot infer magnification. magnification={magnification}, mpp_x={mpp_x}"
    )

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_mag = infer_magnification_from_ckpt_name(args.ckpt)
    print(f"Checkpoint magnification inferred from filename: {ckpt_mag}")

    model = CellPriorNet(
        num_nuclei_classes=args.num_classes,
        backbone=getattr(args, "backbone", "unireplknet_s"),
        pretrained_encoder_ckpt=getattr(args, "pretrained_encoder_ckpt", None),
        magnification=ckpt_mag,
        fuse_mode=args.fuse_mode,
        fuse_alpha=args.fuse_alpha,
        in_channels=4
    ).to(device)


    loader = make_np_loader(
        args.data_root,  args.batch_size, True, args.workers,
        size=args.tile_size, mode=args.input_mode, mean=args.imagenet_mean, std=args.imagenet_std, wsi_name=args.wsi_name)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()


    total_img = 0
    qupath_features = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Inferring tiles", unit="batch")):

            rgb3, h1, _ = _prepare_inputs(batch, device)
            img_b = rgb3.shape[0]
            total_img += img_b
            out= _forward_flexible(model, rgb3, h1)     # logits [B,1,h,w]
            # ---------- Unpack prediction into full dict ----------
            pred_type = out["nuclei_type_map"]  # [B, C, H, W] (C=num_classes)
            pred_bin = out["nuclei_binary_map"]  # [B, 2, H, W]
            prob_fg = F.softmax(pred_bin, dim=1)[:, 1:2]  # [B,1,H,W] in [0,1]
            bin_mask = (prob_fg >= args.prob_thr).float()  # [B,1,H,W] 0/1
            prob_up_split = process_mask_batch_dist_map(
                    bin_mask,  # [B,1,H,W] float32 CUDA (0/1)
                    erode_r=args.erode_r,  # erosion once
                    gsize=args.gsize,
                    marker_method=args.marker_method,  # "hmin" | "peak"
                    h_min=args.h_min,
                    peak_min_distance=args.peak_min_distance,
                    peak_threshold_abs=args.peak_threshold_abs,
                    min_area=args.min_area,
                    num_workers=args.workers,  # 0/1 => no parallel, >1 => parallel
                )

            typed_mask, inst2type = assign_type_to_instances(prob_up_split, pred_type)
            cell_class = {int(k): v for k, v in args.class_names.items()}

            for b in range(img_b):
                    tile_name = batch["id"][b]
                    _, x_start, y_start = parse_tile_xy(tile_name)

                    current_cells = inst2type[b]

                    for inst_id, info in current_cells.items():

                        cx_tile, cy_tile = info["centroid"]

                        ctype = int(info["type"])

                        ctype = ctype if ctype in cell_class else next(
                            k for k, v in cell_class.items() if v.lower() == "other")
                        cx_wsi = x_start + float(cx_tile)
                        cy_wsi = y_start + float(cy_tile)
                        class_name = args.class_names[ctype]
                        feature = make_qupath_cell_feature(cx_wsi=cx_wsi, cy_wsi=cy_wsi, ctype=ctype, class_name=class_name, radius=6)

                        qupath_features.append(feature)

            if args.save_overlays:
                B = rgb3.shape[0]
                for b in range(B):
                    save_classify_dots(
                        args=args,
                        img_t_rgb3=rgb3[b],
                        inst_map=prob_up_split[b, 0],
                        inst_info=inst2type[b],
                        out_prefix=os.path.join(args.out_dir, batch["id"][b]),
                    )

            if (i + 1) % 10 == 0:
                done = total_img
                print(f" Inferred {done} / {len(loader)*args.batch_size} images")


    geojson_path = os.path.join(args.out_dir, f"qupath_cells.geojson")
    save_qupath_geojson(qupath_features, geojson_path)

    print(f"Saved QuPath-compatible GeoJSON: {geojson_path}")
    print(f"Total exported cells: {len(qupath_features)}")
    return qupath_features, total_img




if __name__ == "__main__":

    t0 = time.time()
    parser = argparse.ArgumentParser(description="Run CP-Net inference")
    parser.add_argument("--cell_connectivity", action="store_true")
    parser.add_argument("--model_type", type=str, default="tnmi_20x.pth")

    cli = parser.parse_args()

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(PROJECT_ROOT, "CP-Net", "configs/configs_test.yaml"))

    wsi_info_df = pd.read_excel(os.path.join(PROJECT_ROOT, "output/QA/WSI_Summary.xlsx"))
    model_type = cli.model_type
    cell_connectivity = cli.cell_connectivity

    args = SimpleNamespace(
        # data
        backbone=cfg["model"]["backbone"],
        tile_size=cfg["data"]["tile_size"],
        input_mode=cfg["data"]["input_mode"],
        ckpt=os.path.join(PROJECT_ROOT, "CP-Net/weights", model_type),
        # model
        prob_thr=float(cfg["model"]["prob_threshold"]),
        save_overlays=cfg["data"]["save_overlays"],
        save_seg = cfg["data"]["save_seg"],
        # normalize
        imagenet_mean=tuple(cfg["normalize"]["mean"]),
        imagenet_std=tuple(cfg["normalize"]["std"]),
        fuse_mode=str(cfg["model"]["fuse_mode"]),
        fuse_alpha=float(cfg["model"]["fuse_alpha"]),
        num_classes=int(cfg["model"]["num_classes"]),
        class_colors = {
        int(k): tuple(v) for k, v in cfg["class_colors"].items()
        }
    )

    wsi_files = wsi_info_df.shape[0]
    for _, row in wsi_info_df.iterrows():

        wsi_name = row["wsi_name"]
        tile_size = row["tile_size"]
        args.workers=row["cpu_workers"]
        args.batch_size=row["batch_size"]
        args.wsi_name = wsi_name
        args.data_root= os.path.join(PROJECT_ROOT, "output/QA", wsi_name)
        args.magnification = infer_magnification(
            row["magnification"],
            row["mpp_x"]
        )

        args.out_dir = os.path.join(PROJECT_ROOT, "output/CP_Net", wsi_name)

        print(f"Processing {wsi_name}({ args.magnification})/{wsi_files}")

        # Add preprocessing params from YAML
        mag_key = args.magnification  # e.g. "20x" or "40x"
        if "postprocessing" not in cfg:
            raise KeyError("Config is missing 'postprocessing' section")
        if mag_key not in cfg["postprocessing"]:
            raise KeyError(f"No postprocessing block for magnification '{mag_key}' in config")
        pre_cfg = cfg["postprocessing"][mag_key]

        # Attach these as attributes to args
        args.erode_r = int(pre_cfg.get("erode_r", 1))
        args.gsize = tuple(pre_cfg.get("gsize", [3, 3]))
        args.marker_method = str(pre_cfg.get("marker_method", "peak"))
        args.h_min = float(pre_cfg.get("h_min", 2.0))
        args.peak_min_distance = int(pre_cfg.get("peak_min_distance", 6))
        args.peak_threshold_abs = float(pre_cfg.get("peak_threshold_abs", 2.0))
        args.min_area = int(pre_cfg.get("min_area", 10))
        args.use_raw_h = cfg["data"]["use_raw_h"]
        args.class_names = {
            int(k): str(v) for k, v in cfg["model"]["class_names"].items()
        }

        qupath_features, total_tiles = main(args)

        if cell_connectivity:
            results = cell_neighborhood(args,
                qupath_features=qupath_features,
                include_types="all",
                radius_um=30.0,
                mpp=float(row["mpp_x"]),
                out_dir=os.path.join(args.out_dir, "cell_neighborhood"),
                class_names=args.class_names,
                total_tiles = total_tiles,
                tile_size = tile_size,
            )

    dt = time.time() - t0
    print(f"Inference time: {dt:.2f}s")
