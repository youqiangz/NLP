#! -*- coding:utf-8 -*-
'''
global_pointer用来做实体识别
- 数据集：http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
- 博客：https://kexue.fm/archives/8373
- [valid_f1]: 95.66

思路简介：
- bert出来的logits是[btz, seq_len, hdsz]
- 过global_point得到[btz, ner_vocab_size, seq_len, seq_len]
- 和同维的ner_labels去做MultilabelCategoricalCrossentropy的loss
'''
import os
import pickle
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.callbacks import Callback
from bert4torch.snippets import sequence_padding, ListDataset, seed_everything
from bert4torch.tokenizers import Tokenizer
from bert4torch.losses import MultilabelCategoricalCrossentropy
from bert4torch.layers import GlobalPointer

from NER.utils.utils import read_yaml, load_label

seed_everything(42)

#####################################
# 超参数设置
#####################################
maxlen = 512
batch_size = 16

labels = load_label("./data/merge/processed/labels.txt")
categories_label2id = {label: index for index, label in enumerate(labels)}
categories_id2label = {value: key for key, value in categories_label2id.items()}
ner_vocab_size = len(categories_label2id)
ner_head_size = 64

#####################################
# 加载预训练模型
#####################################
config_file = './pretrain/bert-base-chinese/config.json'
config = read_yaml(config_file)
config_path = os.path.join(config["pretrained_model"], "config.json")
checkpoint_path = os.path.join(config["pretrained_model"], "pytorch_model.bin")
dict_path = os.path.join(config["pretrained_model"], "vocab.txt")

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#####################################
# 加载数据集
#####################################

class MyDataset(ListDataset):
    @staticmethod
    def load_data(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data


tokenizer = Tokenizer(dict_path, do_lower_case=True)


def collate_fn(batch):
    batch_token_ids, batch_labels = [], []
    for i, (text, text_labels) in enumerate(batch):
        tokens = tokenizer.tokenize(text, maxlen=maxlen)
        mapping = tokenizer.rematch(text, tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        token_ids = tokenizer.tokens_to_ids(tokens)
        labels = np.zeros((len(categories_label2id), maxlen, maxlen))
        for start, end, label in text_labels:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                label = categories_label2id[label]
                labels[label, start, end] = 1

        batch_token_ids.append(token_ids)  # 前面已经限制了长度
        batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(sequence_padding(batch_labels, seq_dims=3), dtype=torch.long, device=device)
    return batch_token_ids, batch_labels


#####################################
# 创建dataloader
#####################################
train_dataloader = DataLoader(MyDataset(os.path.join(config["data"], "train.pkl")), batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
valid_dataloader = DataLoader(MyDataset(os.path.join(config["data"], "dev.pkl")), batch_size=batch_size,
                              collate_fn=collate_fn)


#####################################
# 定义bert上的模型结构
#####################################
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path,
                                            segment_vocab_size=0)
        self.global_pointer = GlobalPointer(hidden_size=768, heads=ner_vocab_size, head_size=ner_head_size)

    def forward(self, token_ids):
        sequence_output = self.bert([token_ids])  # [btz, seq_len, hdsz]
        logit = self.global_pointer(sequence_output, token_ids.gt(0).long())
        return logit


class MyLoss(MultilabelCategoricalCrossentropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pred, y_true):
        y_true = y_true.view(y_true.shape[0] * y_true.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        y_pred = y_pred.view(y_pred.shape[0] * y_pred.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        return super().forward(y_pred, y_true)


class Evaluator(Callback):
    # 评估与保存
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        f1, precision, recall = self.evaluate(valid_dataloader)
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('checkpoint/best_model_merge_0318.pt')
        print(f'[val] f1: {f1:.5f}, p: {precision:.5f} r: {recall:.5f} best_f1: {self.best_val_f1:.5f}')

    @staticmethod
    def evaluate(data, threshold=0):
        X, Y, Z = 0, 1e-10, 1e-10
        for x_true, label in data:
            scores = model.predict(x_true)
            for i, score in enumerate(scores):
                R = set()
                for l, start, end in zip(*np.where(score.cpu() > threshold)):
                    R.add((start, end, categories_id2label[l]))

                T = set()
                for l, start, end in zip(*np.where(label[i].cpu() > 0)):
                    T.add((start, end, categories_id2label[l]))
                X += len(R & T)
                Y += len(R)
                Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall


#####################################
# 实例化模型
#####################################

model = Model().to(device)
model.compile(loss=MyLoss(), optimizer=optim.Adam(model.parameters(), lr=2e-5))


#####################################
# 批量推理
#####################################
def batch_inference(text: str, threshold=0, window_size=512, stride=250, batch_size=8):
    """
    :param text:
    :param threshold:
    :param window_size:
    :param stride:
    :param batch_size:
    :return: 实体列表，每个实体为一个字典，包含 "type", "entity", "start", "end", "probability"
    """
    res = []
    chunk_infos = []

    for i in range(0, len(text), stride):
        chunk_text = text[i:i + window_size]
        encoding = tokenizer.encode(
            chunk_text,
            maxlen=window_size,
            return_offsets='transformers',
            return_dict=True
        )
        chunk_infos.append({
            'tokens': encoding['input_ids'],
            'offsets': encoding['offset'],
            'chunk_start': i
        })

    for j in range(0, len(chunk_infos), batch_size):
        batch = chunk_infos[j: j + batch_size]
        tokens_batch = [torch.tensor(item['tokens']) for item in batch]
        tokens_tensor = pad_sequence(tokens_batch, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        max_seq_len = tokens_tensor.size(1)

        # 对 offsets 进行 padding，缺失部分补充 (0, 0)
        padded_offsets_batch = [
            item['offsets'] + [(0, 0)] * (max_seq_len - len(item['offsets']))
            for item in batch
        ]
        chunk_starts = [item['chunk_start'] for item in batch]

        with torch.no_grad():
            scores_batch = model.predict(tokens_tensor)

        for idx, (scores, offsets, chunk_start, tokens_seq) in enumerate(zip(
                scores_batch, padded_offsets_batch, chunk_starts, tokens_batch)):
            # 假设 scores 的 shape 为 (num_labels, seq_len, seq_len)
            scores_np = scores.cpu().numpy()
            for l, start, end in zip(*np.where(scores_np > threshold)):
                if 0 <= start < len(offsets) and 0 < end <= len(offsets):
                    entity_text = tokenizer.decode(tokens_seq[start:end])
                    score_raw = scores[l, start, end]
                    probability = float(torch.sigmoid(score_raw).cpu().numpy())

                    entity_info = {
                        "type": categories_id2label[l],
                        "entity": entity_text,
                        "start": offsets[start][0] + chunk_start,
                        "end": offsets[end - 1][1] + chunk_start,
                        "probability": probability
                    }
                    res.append(entity_info)
                else:
                    print(f"跳过不合法的索引：start={start}, end={end}, offsets长度={len(offsets)}")
    return res


if __name__ == '__main__':
    evaluator = Evaluator()
    choice = 'infer'

    if choice == 'train':
        model.fit(train_dataloader, epochs=20, steps_per_epoch=None, callbacks=[evaluator])
    elif choice == 'eval':
        model.load_weights('best_model.pt')
        print(evaluator.evaluate(valid_dataloader))
    elif choice == 'infer':
        model.load_weights('./checkpoint/best_model.pt')
        model.eval()
        with open("./data/test.jsonl", 'r', encoding="utf-8") as f, open(
                "./data/results.jsonl", "w", encoding="utf-8") as out_f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = json.loads(line)
                results = batch_inference(line["text"])
                out_f.write(json.dumps({"id": line["id"], "results": results}, ensure_ascii=False) + "\n")
