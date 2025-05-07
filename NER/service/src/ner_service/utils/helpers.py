# -*- coding: utf-8 -*-

import json
import time
from functools import wraps
from typing import Callable, Any


def timing_decorator(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper


def safe_json_loads(data: str) -> dict:
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return {}


def load_label(file_path: str) -> list[str]:
    # 读取标签文件
    with open(file_path, 'r', encoding="utf-8") as f:
        labels = [label.strip() for label in f.readlines()]
    return labels
