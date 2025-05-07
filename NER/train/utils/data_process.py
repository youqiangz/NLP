# -*- coding: utf-8 -*-


"""
数据的预处理：
1.jsonl格式
2.每一行数据格式：{"text": "...",
                 "entities": [ {"id": 0, "label": "人名", "start": 5, "end": 8 },
                            ...
                        ]}

"""

import os
import json
import pickle
import random
from tqdm import tqdm
from pathlib import Path

random.seed(42)


def process_data(in_file, window_size=512, stride=512):
    """
    数据处理部分：文本分块、实体过滤、标签收集
    :param in_file: 输入文件路径
    :param window_size: 滑动窗口大小
    :param stride: 滑动窗口步长
    :return: (处理后的数据列表, 标签集合)
    """
    labels = set()
    data = []

    with open(in_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue

            line = json.loads(line)
            entities = line.get("entities", [])
            text = line.get("text", "")

            # 滑动窗口分块处理
            for i in range(0, len(text), stride):
                chunk_start = i
                chunk_end = min(i + window_size, len(text))
                chunk_text = text[chunk_start:chunk_end]

                # 筛选本窗口内的实体
                valid_entities = []
                for ent in entities:
                    if ent["start"] >= chunk_start and ent["end"] <= chunk_end:
                        valid_entities.append([
                            ent["start"] - chunk_start,
                            ent["end"] - chunk_start,
                            ent["label"]
                        ])
                        labels.add(ent["label"])

                data.append((chunk_text, valid_entities))
    random.shuffle(data)
    return data, labels


def process_data_plus(in_file, window_size=512, stride=512):
    """
    数据处理部分：文本分块、实体过滤、标签收集
    优化部分：1.针对左侧上下文不足（不到10个token）的情况进行截断处理，只针对需要利用上下文进行判断的实体类型，否则不需要使用该方法
             2.增加不包含实体的训练样本，条数与包含实体的样本数相同
    :param in_file: 输入文件路径
    :param window_size: 滑动窗口大小
    :param stride: 滑动窗口步长
    :return: (处理后的数据列表, 标签集合)
    """
    labels = set()

    with open(in_file, "r", encoding="utf-8") as f:
        data, data_no_entity = [], []
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue

            line = json.loads(line)
            entities = line.get("entities", [])
            text = line.get("text", "")

            # 分块处理
            for i in range(0, len(text), stride):
                chunk_start = i
                chunk_end = min(i + window_size, len(text))
                chunk_text = text[chunk_start:chunk_end]

                # 过滤实体
                valid_entities = []
                for ent in entities:
                    if ent["start"] >= chunk_start and ent["end"] <= chunk_end:
                        valid_entities.append([
                            ent["start"] - chunk_start,
                            ent["end"] - chunk_start,
                            ent["label"]
                        ])
                        labels.add(ent["label"])

                # 针对窗口中实体左侧上下文不足的情况进行截断处理
                if valid_entities:
                    # 检查是否存在某个实体的起始位置小于10
                    cutoff = None
                    for ent in valid_entities:
                        if ent[0] < 10:
                            cutoff = ent[1]  # 以该实体的尾部作为截断点
                            break
                    if cutoff is not None:
                        # 截断窗口文本
                        chunk_text = chunk_text[:cutoff]
                        # 更新实体索引：只保留起始位置在截断范围内的实体，
                        # 对于实体的结束位置超过截断点的，更新为截断点
                        new_valid_entities = []
                        for ent in valid_entities:
                            if ent[0] < cutoff:
                                new_ent = [ent[0], min(ent[1], cutoff), ent[2]]
                                new_valid_entities.append(new_ent)
                        valid_entities = new_valid_entities

                # 仅保留包含实体的块（可选）
                if valid_entities:
                    data.append((chunk_text, valid_entities))
                else:  # 不包含实体的块
                    data_no_entity.append((chunk_text, valid_entities))
    random.shuffle(data)
    random.shuffle(data_no_entity)
    final_data = data + data_no_entity[:len(data)]  # 包含实体和不包含实体的样本数量一样

    return final_data, labels


def save_processed_data(data, labels, save_dir, train_ratio=0.9, val_ratio=0.1):
    """
    文件处理部分：数据集划分与持久化存储
    :param data: 处理后的数据列表
    :param labels: 标签集合
    :param save_dir: 保存目录
    :param train_ratio: 训练集比例
    :param val_ratio: 验证集比例
    """
    # 数据划分
    random.shuffle(data)
    total = len(data)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)

    train_data = data[:train_size]
    dev_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # 调试输出示例数据
    print("\n训练数据示例：")
    print(train_data[:2])

    # 确保保存目录存在
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 保存数据集
    with open(os.path.join(save_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join(save_dir, "dev.pkl"), "wb") as f:
        pickle.dump(dev_data, f)
    with open(os.path.join(save_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_data, f)

    # 保存标签文件
    with open(os.path.join(save_dir, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(labels)))


def main():
    # 一般采用第一种方法即可，除非存在一些实体类型需要足够的上下文才能判断出来，可自行修改第二种数据处理方法
    processed_data, label_set = process_data("input.jsonl")

    save_processed_data(
        data=processed_data,
        labels=label_set,
        save_dir="./processed_data",
        train_ratio=0.9,
        val_ratio=0.1
    )


if __name__ == '__main__':
    main()
