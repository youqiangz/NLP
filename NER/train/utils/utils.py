# -*- coding: utf-8 -*-

import os
import yaml
import json
import re
import pandas as pd


def read_yaml(f):
    # 读取配置文件
    with open(f, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_label(file_path):
    # 读取标签文件
    with open(file_path, 'r', encoding="utf-8") as f:
        labels = [label.strip() for label in f.readlines()]
    return labels