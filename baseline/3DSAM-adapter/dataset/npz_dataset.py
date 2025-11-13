# npz_dataset.py
from typing import Dict, Hashable, Mapping, Sequence
import os, pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# MONAI transforms (옵션: 간단한 공간 증강/크롭만)
from monai.transforms import (
    Compose,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    MapTransform,
)

class BinarizeLabeld(MapTransform):
    """label > 0 -> 1 (float)로 이진화"""
    def __init__(self, keys, threshold: float = 0.5, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.threshold = threshold

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            t = d[key]
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t)
            d[key] = (t > self.threshold).to(torch.float32)
        return d


class NPZVolumeDataset(Dataset):
    """
    .npz 파일에서 (imgs: (D,H,W), gts: (D,H,W), spacing: (3,))을 읽어
    (image_tensor, label_tensor, spacing_array)를 반환.
    - image_tensor: (3, D, H, W)  # 3채널로 확장 (SAM 등 RGB 기대 모델 호환)
    - label_tensor: (D, H, W)     # float 0/1
    - spacing_array: np.ndarray shape (3,)
    """
    def __init__(
        self,
        file_paths,
        split: str = "train",
        augmentation: bool = False,
        rand_crop_spatial_size: Sequence[int] = (96, 96, 96),
        do_val_crop: bool = True,
        do_test_crop: bool = False,
        to_rgb: bool = True,  # 1채널 -> 3채널 반복
    ):
        self.file_paths = file_paths
        self.split = split
        self.augmentation = augmentation
        self.rand_crop_spatial_size = tuple(rand_crop_spatial_size)
        self.do_val_crop = do_val_crop
        self.do_test_crop = do_test_crop
        self.to_rgb = to_rgb

        self.transforms = self._build_transforms()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        with np.load(path) as npz:
            imgs = npz["imgs"].astype(np.float32)      # (D,H,W)
            gts  = npz["gts"].astype(np.float32)       # (D,H,W)
            spacing = np.array(npz["spacing"])         # (3,)

        # NaN 방지
        imgs[np.isnan(imgs)] = 0
        gts[np.isnan(gts)] = 0

        # MONAI dict 기반 파이프라인에 전달
        # 채널 차원은 transform에서 다루기 쉽게 torch 변환 전에 그대로 둠
        sample = {"image": imgs, "label": gts}

        # 변환 적용
        if self.transforms is not None:
            out = self.transforms(sample)
            # RandCropByPosNegLabeld 사용 시 list를 반환할 수 있어 [0] 처리
            if isinstance(out, list):
                out = out[0]
            imgs, gts = out["image"], out["label"]

        # numpy -> torch
        img_t = torch.as_tensor(imgs, dtype=torch.float32)   # (D,H,W)
        gt_t  = torch.as_tensor(gts,  dtype=torch.float32)   # (D,H,W)
        # gt_t  = torch.as_tensor(gts > 0, dtype=torch.bool)

        # 채널 확장(3채널)
        if self.to_rgb:
            # img_t = img_t.unsqueeze(0).repeat(3, 1, 1, 1)    # (3,D,H,W)
            img_t = img_t.unsqueeze(0).expand(3, -1, -1, -1)   # ✅ 메모리 복제 없음
        else:
            img_t = img_t.unsqueeze(0)                       # (1,D,H,W)

        return img_t, gt_t, spacing

    def _build_transforms(self):
        # 이미 기본 전처리됨 → 강한 강도 변환/정규화는 생략
        t = []

        if self.split == "train":
            if self.augmentation:
                # 간단한 공간 증강 (pad -> pos/neg crop -> 랜덤 flip/회전)
                t.extend([
                    BinarizeLabeld(keys=["label"]),
                    SpatialPadd(keys=["image", "label"],
                                spatial_size=[round(i * 1.1) for i in self.rand_crop_spatial_size]),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=[round(i * 1.1) for i in self.rand_crop_spatial_size],
                        label_key="label",
                        pos=2, neg=1, num_samples=1,
                    ),
                    RandSpatialCropd(keys=["image", "label"],
                                     roi_size=self.rand_crop_spatial_size, random_size=False),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
                ])
            else:
                # 증강 없이 중앙 크롭(or 유지)만 원하면 RandSpatialCropd만 조정
                t.extend([
                    BinarizeLabeld(keys=["label"]),
                ])

        elif self.split == "val":
            if self.do_val_crop:
                t.extend([
                    BinarizeLabeld(keys=["label"]),
                    SpatialPadd(keys=["image", "label"], spatial_size=list(self.rand_crop_spatial_size)),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=self.rand_crop_spatial_size,
                        label_key="label",
                        pos=1, neg=0, num_samples=1,
                    ),
                ])
            else:
                t.extend([
                    BinarizeLabeld(keys=["label"]),
                ])

        elif self.split == "test":
            # 테스트: 보통 이진화만 (crop은 do_test_crop이 True면 추가 가능)
            if self.do_test_crop:
                t.extend([
                    BinarizeLabeld(keys=["label"]),
                    SpatialPadd(keys=["image", "label"], spatial_size=list(self.rand_crop_spatial_size)),
                    RandSpatialCropd(keys=["image", "label"],
                                     roi_size=self.rand_crop_spatial_size,
                                     random_size=False),
                ])
            else:
                t.extend([
                    BinarizeLabeld(keys=["label"]),
                ])
        else:
            raise NotImplementedError

        return Compose(t) if t else None
