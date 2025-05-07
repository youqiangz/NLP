# -*- coding: utf-8 -*-

from typing import Dict, List, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from ..config.config import settings
from ..processing.preprocess import TextChunker
from ..processing.postprocess import Entity, merge_overlap_entities
from .model_loader import ModelLoader, categories_id2label


class NERPredictor:
    def __init__(self):
        """初始化模型和配置"""
        self.model = ModelLoader.get_model()
        self.tokenizer = ModelLoader.get_tokenizer()

        # 加载配置参数
        self.max_len = settings.model.max_len
        self.stride = settings.model.stride
        self.batch_size = settings.model.batch_size
        self.device = settings.model.device if torch.cuda.is_available() else "cpu"

        # 将模型移动到指定设备
        self._move_model_to_device()

    def _move_model_to_device(self):
        """将模型移动到指定设备（CPU/GPU）"""
        if self.device == "cuda":
            self.model.to("cuda")

    def _tokenize_and_chunk(self, text: str) -> List[Dict]:
        """
        分块文本并进行 tokenizer 编码
        :param text: 原始文本
        :return:
        """
        chunks = TextChunker.chunk_text(
            text,
            window_size=self.max_len,
            stride=self.stride
        )
        chunk_infos = []
        for chunk in chunks:
            encoding = self.tokenizer.encode(
                chunk.text,
                maxlen=self.max_len,
                return_offsets='transformers',
                return_dict=True
            )
            chunk_infos.append({
                'token_ids': encoding['input_ids'],
                'token_offsets': encoding['offset'],
                'chunk_start': chunk.start
            })
        return chunk_infos

    def _process_batch(self, batch: List[Dict]) -> List[Entity]:
        """
        处理一批 chunk 信息并返回实体列表
        :param batch: chunk 信息批次
        :return: 实体列表，每个元素为 (type, text, start, end, probability) 元组
        """
        tokens_batch = [torch.tensor(item['token_ids']) for item in batch]
        tokens_tensor = pad_sequence(tokens_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        tokens_tensor = tokens_tensor.to(self.device)

        max_seq_len = tokens_tensor.size(1)
        padded_offsets_batch = [
            item['token_offsets'] + [(0, 0)] * (max_seq_len - len(item['token_offsets']))
            for item in batch
        ]
        chunk_starts = [item['chunk_start'] for item in batch]

        with torch.no_grad():
            scores_batch = self.model.predict(tokens_tensor)

        res: List[Entity] = []
        for scores, token_offsets, chunk_start, tokens_seq in zip(
                scores_batch, padded_offsets_batch,
                chunk_starts, tokens_batch):
            scores_np = scores.cpu().numpy()
            for label_idx, start, end in zip(*np.where(scores_np > 0)):
                if 0 <= start < len(token_offsets) and 0 < end <= len(token_offsets) and start < end:
                    entity_text = self.tokenizer.decode(tokens_seq[start:end])
                    score_raw = scores[label_idx, start, end]
                    probability = float(torch.sigmoid(score_raw).cpu().numpy())

                    entity = Entity(
                        type=categories_id2label[label_idx],
                        text=entity_text,
                        start=token_offsets[start][0] + chunk_start,
                        end=token_offsets[end - 1][1] + chunk_start,
                        confidence=probability
                    )
                    res.append(entity)
                else:
                    print(f"跳过不合法的索引：start={start}, end={end}, offsets长度={len(token_offsets)}")
        return res

    def predict(self, text: str, threshold: float = 0.0) -> List[Dict[str, Union[str, int, float]]]:
        """
        对单个文本执行命名实体识别
        :param text: 输入文本
        :param threshold: 置信度阈值 (当前未使用，保留接口兼容性)
        :return: 实体列表，每个实体为字典格式:
            - type: 实体类型 (str)
            - text: 实体文本 (str)
            - start: 起始位置 (int)
            - end: 结束位置 (int)
            - confidence: 置信度 (float)
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        if not text.strip():
            raise ValueError("Input text cannot be empty or whitespace only")

        chunk_infos = self._tokenize_and_chunk(text)
        all_entities: List[Entity] = []

        for i in range(0, len(chunk_infos), self.batch_size):
            batch = chunk_infos[i:i + self.batch_size]
            batch_results = self._process_batch(batch)
            all_entities.extend(batch_results)
        merged_entities: List[Entity] = merge_overlap_entities(all_entities)
        return [e._asdict() for e in merged_entities]
