from scipy import ndimage
import numpy as np
from medpy import metric
import logging
import os
import time
import torch


def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    """
    지정된 이름의 logger를 생성하고, 콘솔 출력 또는 파일 출력 옵션을 설정하는 함수.

    Args:
        logger_name (str): 생성할 logger의 이름
        root (str): 로그 파일을 저장할 디렉토리 경로
        level (int): logging level (기본: INFO)
        screen (bool): True일 경우, 로그를 콘솔(stdout)에도 출력
        tofile (bool): True일 경우, 로그를 파일로 저장

    Returns:
        logging.Logger: 구성된 logger 객체

    기능:
        - formatter: [HH:MM:SS.mmm] 메시지 형식으로 출력
        - 파일 출력 시: "{logger_name}_YYYYMMDD-HHMMSS.log" 파일 생성
        - screen=True → StreamHandler 추가
        - tofile=True → FileHandler 추가
    """
    lg = logging.getLogger(logger_name)  # 동일 이름으로 logger를 재사용함
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    lg.setLevel(level)
    # 파일로 저장하는 옵션
    if tofile:
        log_file = os.path.join(root, "{}_{}.log".format(logger_name, get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")  # 새 파일 생성
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    # 콘솔 출력 옵션
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
    return lg


def get_timestamp():
    """
    YYYYMMDD-HHMMSS 포맷의 시간 문자열을 생성하는 함수.
    파일명 또는 로그 파일 생성 시 사용.

    Returns:
        str: 예) "20251115-134512"
    """
    timestampTime = time.strftime("%H%M%S")    # HHMMSS
    timestampDate = time.strftime("%Y%m%d")    # YYYYMMDD
    return timestampDate + "-" + timestampTime
