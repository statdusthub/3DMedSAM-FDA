# make_split_pkl.py
import os
import pickle
import random

def make_split_pkl(
    data_dir: str,
    output_path: str = None,
    split_ratio=(0.7, 0.15, 0.15),
    seed: int = 42,
    relative_dir: str = "npz",
):
    """
    data_dir 내부의 .npz 파일들을 (train/val/test) 비율로 나누어 split.pkl 생성

    Args:
        data_dir (str): .npz 파일이 들어 있는 디렉토리
        output_path (str, optional): 저장할 pkl 경로 (기본: data_dir/split.pkl)
        split_ratio (tuple): (train, val, test) 비율, 합=1.0
        seed (int): 랜덤 시드
        relative_dir (str): pkl에 기록될 상대경로 prefix (예: "npz/파일명.npz")
    """

    # 1. 파일 목록 수집
    all_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".npz")
    ])
    if not all_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    # 2. 랜덤 셔플 및 분할
    random.seed(seed)
    random.shuffle(all_files)

    n_total = len(all_files)
    n_train = int(n_total * split_ratio[0])
    n_val   = int(n_total * split_ratio[1])
    n_test  = n_total - n_train - n_val

    train_files = all_files[:n_train]
    val_files   = all_files[n_train:n_train+n_val]
    test_files  = all_files[n_train+n_val:]

    # 3. dict 구조 생성
    def to_dict(file_list):
        return {i: f"{relative_dir}/{fname}" for i, fname in enumerate(file_list)}

    split_dict = {
        0: {   # fold 0
            "train": to_dict(train_files),
            "val":   to_dict(val_files),
            "test":  to_dict(test_files),
        }
    }

    # 4. 저장 경로 지정
    if output_path is None:
        output_path = os.path.join(data_dir, "split.pkl")

    with open(output_path, "wb") as f:
        pickle.dump(split_dict, f)

    # 5. 결과 출력
    print(f"✅ split.pkl saved at: {output_path}")
    print(f" - Total: {n_total}  | Train: {len(train_files)}  Val: {len(val_files)}  Test: {len(test_files)}")
    return split_dict


# if __name__ == "__main__":
#     # 예시 실행
#     SPLIT = make_split_pkl(
#         data_dir="./dataset_npz/npz",     # npz 폴더 경로
#         split_ratio=(0.7, 0.15, 0.15),
#         seed=42,
#         relative_dir="npz"               # path_prefix 아래의 상대경로로 저장
#     )
