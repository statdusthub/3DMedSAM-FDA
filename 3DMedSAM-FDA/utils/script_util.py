import argparse


def add_dict_to_argparser(parser, default_dict):
    """
    default_dict에 정의된 key-value 쌍을 기반으로 argparse parser에 argument를 자동 등록하는 함수.

    Args:
        parser (ArgumentParser): argparse 인자 파서
        default_dict (dict): {arg_name: default_value} 형식의 기본 설정값 dictionary

    기능:
        - default_dict의 key를 기반으로 --key 형식의 command-line argument를 parser에 추가
        - default 값의 타입을 자동 추론하여 type 설정
        - bool 타입은 argparse가 기본 지원하지 않으므로 str2bool로 처리
        - None은 명시적으로 타입을 string으로 처리
    """
    for k, v in default_dict.items():
        v_type = type(v)

        # None일 경우 type 추론이 불가능하므로 string으로 처리
        if v is None:
            v_type = str
        # bool 타입 → str2bool 변환 함수 사용
        elif isinstance(v, bool):
            v_type = str2bool

        # --key 형식의 argument 추가
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    """
    argparse Namespace → dictionary 변환 함수.

    Args:
        args (Namespace): argparse로 파싱된 인자 set
        keys (list): 가져오고 싶은 key 이름 목록

    Returns:
        dict: {key: args.key} 형태의 dictionary
    """
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    """
    문자열을 boolean(True/False)로 안전하게 변환하는 함수.
    argparse가 기본적으로 boolean 파싱을 잘 지원하지 않기 때문에 custom 처리.

    예시 입력:
        yes, true, t, y, 1  → True
        no, false, f, n, 0 → False
    """
    # 이미 bool이면 그대로 반환
    if isinstance(v, bool):
        return v
    # 소문자 기반 비교
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


import torch as th
import shutil
import os


def save_checkpoint(state, is_best, checkpoint):
    """
    학습 중 모델/optimizer 상태 등을 저장하는 checkpoint 함수.

    Args:
        state (dict): 저장할 state dict (모델 상태, optimizer 상태 등)
        is_best (bool): 현재 checkpoint가 최고 성능(best)인지 여부
        checkpoint (str): checkpoint 파일을 저장할 디렉토리 경로

    기능:
        - checkpoint 디렉토리가 없으면 생성
        - 항상 last.pth.tar 로 저장
        - is_best=True이면 best.pth.tar 로 복사 저장
    """
    filepath_last = os.path.join(checkpoint, "last.pth.tar")
    filepath_best = os.path.join(checkpoint, "best.pth.tar")
    # checkpoint 디렉토리 존재 여부 확인
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Masking directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint DIrectory exists!")
    
    # 항상 최근 모델 저장
    th.save(state, filepath_last)

    # best model이면 best.pth.tar 업데이트
    if is_best:
        # 기존 best 파일 삭제 후 교체
        if os.path.isfile(filepath_best):
            os.remove(filepath_best)
        shutil.copyfile(filepath_last, filepath_best)


import numpy as np

def flat_mean(x):
    """
    입력 텐서를 2D 형태로 flatten(D, N)한 후, 각 sample마다 평균을 반환하는 함수.

    Args:
        x (Tensor): shape (B, C, D, H, W, ...)처럼 여러 차원을 가진 텐서

    동작:
        - 첫 번째 차원(Batch)을 제외하고 모두 flatten → (B, -1)
        - dim=1 기준으로 평균 계산 → 각 배치마다 feature 평균

    Returns:
        Tensor: shape (B,)
    """
    # B × (C*D*H*W…) 형태로 변환
    x = x.reshape(x.shape[0], -1)
    # 각 sample마다 평균 계산
    return th.mean(x, dim=1)