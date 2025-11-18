# src/utils/helpers.py
import os
import time
from contextlib import contextmanager


def ensure_dir(path: str):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def print_section(title: str):
    bar = "=" * (len(title) + 4)
    print(f"\n{bar}")
    print(f"| {title} |")
    print(f"{bar}\n")


@contextmanager
def time_block(name: str):
    start = time.time()
    print(f"⏱  {name} ...")
    try:
        yield
    finally:
        end = time.time()
        print(f"✅  {name} using:  {end - start:.2f}s\n")
