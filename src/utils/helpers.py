# src/utils/helpers.py
import os
import time
from contextlib import contextmanager


def ensure_dir(path: str):
    """
    确保文件所在的目录存在。
    传入的是完整文件路径，例如 'results/charts/equity.png'
    """
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def print_section(title: str):
    """
    在终端打印一个清晰的分隔标题。
    """
    bar = "=" * (len(title) + 4)
    print(f"\n{bar}")
    print(f"| {title} |")
    print(f"{bar}\n")


@contextmanager
def time_block(name: str):
    """
    用法:
        with time_block("Grid Search"):
            ...
    """
    start = time.time()
    print(f"⏱  {name} ...")
    try:
        yield
    finally:
        end = time.time()
        print(f"✅  {name} 完成，用时 {end - start:.2f}s\n")
