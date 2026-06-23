import os
import torch
import numpy as np
from PIL import Image
from glob import glob
import albumentations as A
from multiprocessing import Pool, cpu_count
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2


def extract_H_channel(img, Io=240, alpha=1, beta=0.15):
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[..., :3]

    h, w, _ = img.shape
    img_flat = img.reshape((-1, 3))

    OD = -np.log((img_flat.astype(float) + 1) / Io)
    ODhat = OD[~np.any(OD < beta, axis=1)]

    if ODhat.shape[0] < 10:
        return img.astype(np.uint8)

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


def process_H_image(args):
    img_path, h_dir, Io, alpha, beta = args

    name = os.path.basename(img_path)
    h_path = os.path.join(h_dir, name)

    if os.path.exists(h_path):
        return h_path

    img = np.array(Image.open(img_path).convert("RGB"))
    img_H = extract_H_channel(
        img,
        Io=Io,
        alpha=alpha,
        beta=beta
    )

    Image.fromarray(img_H).save(h_path)

    return h_path


class Load_Dataset(Dataset):
    """
    Testing/inference dataset.

    Expected folder structure:

    Dataset_name/
        images_HE/
            xxx.png
        images_H/
            xxx.png   # automatically created if missing

    No masks are needed for testing.
    """

    def __init__(
        self,
        split_dir,
        size=256,
        mode="he_plus_h",
        IMAGENET_MEAN=None,
        IMAGENET_STD=None,
        Io=240,
        alpha=1,
        beta=0.15,
        h_num_workers=None,
    ):
        self.mode = mode

        self.he_dir = os.path.join(split_dir, "Qualified")
        self.h_dir = os.path.join(split_dir, "Qualified_H")

        self.img_paths = sorted(glob(os.path.join(self.he_dir, "*.png")))

        if len(self.img_paths) == 0:
            raise FileNotFoundError(
                f"No PNG images found in {self.he_dir}"
            )

        if self.mode == "he_plus_h":
            self.prepare_H_images(
                Io=Io,
                alpha=alpha,
                beta=beta,
                num_workers=h_num_workers
            )

            self.h_paths = [
                os.path.join(self.h_dir, os.path.basename(p))
                for p in self.img_paths
            ]
        else:
            self.h_paths = [None] * len(self.img_paths)

        self.transform = self.build_aug(
            size=size,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )

    def build_aug(self, size, mean, std):
        return A.Compose(
            [
                A.Resize(size, size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ],
            additional_targets={
                "image_h": "image"
            } if self.mode == "he_plus_h" else {}
        )

    def prepare_H_images(self, Io, alpha, beta, num_workers=None):
        os.makedirs(self.h_dir, exist_ok=True)

        missing_paths = [
            p for p in self.img_paths
            if not os.path.exists(
                os.path.join(self.h_dir, os.path.basename(p))
            )
        ]

        if len(missing_paths) == 0:
            print("All H-channel images already exist.")
            return

        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)

        num_workers = min(num_workers, len(missing_paths))

        print(
            f"Generating {len(missing_paths)} H-channel images "
            f"using {num_workers} workers..."
        )

        worker_args = [
            (p, self.h_dir, Io, alpha, beta)
            for p in missing_paths
        ]

        with Pool(processes=num_workers) as pool:
            for _ in pool.imap_unordered(process_H_image, worker_args):
                pass

        print(f"H-channel images saved in: {self.h_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
            img_HE = np.array(
                Image.open(self.img_paths[idx]).convert("RGB")
            )

            if self.mode == "he_plus_h":
                img_H = np.array(
                    Image.open(self.h_paths[idx]).convert("L")
                )

                sp = self.transform(
                    image=img_HE,
                    image_h=img_H
                )

                img_HE_t = sp["image"]  # [3,H,W]
                img_H_t = sp["image_h"]  # [1,H,W]

                x = torch.cat([img_HE_t, img_H_t], dim=0)  # [4,H,W]

            else:
                sp = self.transform(image=img_HE)
                x = sp["image"]

            return {
                "image": x,
                "id": os.path.basename(self.img_paths[idx]),
            }


def make_np_loader(
    data_root,
    batch_size,
    shuffle,
    workers,
    size=256,
    mode="he_plus_h",
    mean=None,
    std=None,
    wsi_name=None,
    h_num_workers=None,
):
    split_dir = os.path.join(data_root)

    ds = Load_Dataset(
        split_dir=split_dir,
        size=size,
        mode=mode,
        IMAGENET_MEAN=mean,
        IMAGENET_STD=std,
        h_num_workers=h_num_workers,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )