# -*- coding: utf-8 -*-

from typing import Dict

import torch

from bert4torch.layers import GlobalPointer
from bert4torch.models import BaseModel, build_transformer_model
from bert4torch.tokenizers import Tokenizer

from ..config.config import settings
from ..utils.helpers import load_label


labels: list[str] = load_label(settings.model.labels)
categories_label2id: Dict[str, int] = {label: index for index, label in enumerate(labels)}
categories_id2label: Dict[int, str] = {value: key for key, value in categories_label2id.items()}


class Model(BaseModel):
    def __init__(self):
        super().__init__()
        ner_vocab_size = len(categories_label2id)
        self.bert = build_transformer_model(config_path=settings.model.config_path, segment_vocab_size=0)
        self.global_pointer = GlobalPointer(hidden_size=768, heads=ner_vocab_size, head_size=settings.model.head_size)

    def forward(self, token_ids):
        sequence_output = self.bert([token_ids])  # [btz, seq_len, hdsz]
        logits = self.global_pointer(sequence_output, token_ids.gt(0).long())
        return logits


class ModelLoader:
    _model = None
    _tokenizer = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = Model()
            cls._model.load_weights(settings.model.path)
            cls._model.eval()
            if torch.cuda.is_available() and settings.model.device == "cuda":
                cls._model.to("cuda")
        return cls._model

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = Tokenizer(settings.model.tokenizer, do_lower_case=True)
        return cls._tokenizer
