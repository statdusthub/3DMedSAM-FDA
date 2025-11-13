# npz_loader.py
import os
import pickle
from torch.utils.data import DataLoader
from .npz_dataset import NPZVolumeDataset


def load_npz_data_volume(
    *,
    path_prefix: str,
    split: str = "test",               # "train" / "val" / "test"
    batch_size: int = 1,
    data_dir: str | None = None,      # split.pkl 경로 (기본: <path_prefix>/split.pkl)
    deterministic: bool = True,
    augmentation: bool = False,
    rand_crop_spatial_size=(96, 96, 96),
    do_val_crop: bool = True,
    do_test_crop: bool = False,
    num_worker: int = 0,
    file_list: list[str] | None = None,  # split.pkl 없이 직접 리스트로 넘길 수도 있음
):
    if data_dir is None:
        data_dir = os.path.join(path_prefix, "split.pkl")

    if file_list is None:
        with open(data_dir, "rb") as f:
            d_all = pickle.load(f)            # {fold: {"train":{idx:path}, ...}}
        d = d_all[0][split]                   # 기본 fold=0 사용
        file_list = [os.path.join(path_prefix, d[i].strip("/")) for i in sorted(d.keys())]

    dataset = NPZVolumeDataset(
        file_paths=file_list,
        split=split,
        augmentation=augmentation,
        rand_crop_spatial_size=rand_crop_spatial_size,
        do_val_crop=do_val_crop,
        do_test_crop=do_test_crop,
        to_rgb=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=num_worker,
        pin_memory=False,
    )
    return loader
