import argparse
import os
from glob import glob
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image


EXTS = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]


def normalize_staining_H(img, Io=240, alpha=1, beta=0.15):
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[..., :3]
    elif img.shape[2] != 3:
        raise ValueError(f"Unsupported channel count: {img.shape[2]}")

    h, w, c = img.shape
    img = img.reshape((-1, 3))

    OD = -np.log((img.astype(float) + 1) / Io)

    ODhat = OD[~np.any(OD < beta, axis=1)]

    if ODhat.shape[0] < 10:
        return img.reshape(h, w, 3).astype(np.uint8)

    _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    That = ODhat.dot(eigvecs[:, 1:3])
    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(
        np.array([[np.cos(minPhi)], [np.sin(minPhi)]])
    )
    vMax = eigvecs[:, 1:3].dot(
        np.array([[np.cos(maxPhi)], [np.sin(maxPhi)]])
    )

    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    Y = OD.T
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    maxC = np.array([
        np.percentile(C[0, :], 99),
        np.percentile(C[1, :], 99)
    ])

    tmp = maxC / maxCRef
    C2 = C / tmp[:, np.newaxis]

    H = Io * np.exp(
        np.expand_dims(-HERef[:, 0], axis=1).dot(
            np.expand_dims(C2[0, :], axis=0)
        )
    )

    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    return H


def process_one(args):
    path, save_folder, Io, alpha, beta = args

    img = np.array(Image.open(path).convert("RGB"))

    H = normalize_staining_H(
        img=img,
        Io=Io,
        alpha=alpha,
        beta=beta
    )

    name = os.path.basename(path)
    save_path = os.path.join(save_folder, name)

    Image.fromarray(H).save(save_path)

    return save_path


def get_image_paths(folder):
    paths = []
    for ext in EXTS:
        paths.extend(sorted(glob(os.path.join(folder, ext))))
    return paths


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--imageFolder", type=str, default="input")
    parser.add_argument("--saveFolder", type=str, default="H_Channel_Out")
    parser.add_argument("--Io", type=int, default=240)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=cpu_count())

    args = parser.parse_args()

    os.makedirs(args.saveFolder, exist_ok=True)

    paths = get_image_paths(args.imageFolder)
    print(f"Found {len(paths)} images")

    worker_args = [
        (p, args.saveFolder, args.Io, args.alpha, args.beta)
        for p in paths
    ]

    with Pool(processes=args.num_workers) as pool:
        for saved_path in pool.imap_unordered(process_one, worker_args):
            print(f"Saved: {saved_path}")

    print(f"Done. H channel saved in: {args.saveFolder}")