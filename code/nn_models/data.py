"""
构建数据集
"""
import json
import torch
import torch.nn as nn
import nn_models.nn_config as nn_config
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import pickle
import dgl
import random
from functools import reduce


def tokenizer(s):
    """
    :param s: 摘要文本，切分成token
    :return:
    """
    tokens = []
    li = s.split()
    for word in li:
        if word.startswith('ENTITY'):
            tokens.append(word)
        else:
            _ = ''.join((char if char.isalpha() or char.isdigit() else " ") for char in word).split()
            for t in _:
                tokens.append(t.lower())  # 小写
    return tokens


class SampleGenerator:
    """
    针对一个sample : (m, e)
    生成可送入深度学习模型的数据
    """
    def __init__(self, mention_abstract_emb_fp, cell_max=5, row_max=5, col_max=30, abstract_max=100):
        self.cell_max = cell_max
        self.row_max = row_max
        self.col_max = col_max
        self.abstract_max = abstract_max
        with open(mention_abstract_emb_fp, 'rb') as f:
            self.mention_encode, self.abstract_encode = pickle.load(f)

    def process_sample(self, sample):
        """
        为一对m, e；构建符合深度学习模型的数据
        :param sample: {
            'cell_context': [...],
            'col_context': [...],
            'row_context': [...],
            'mention': '...',
            'abstract': '...',
        }
        :return:
        """
        cell_context = sample['cell_context']
        if len(cell_context) > self.cell_max:
            random.shuffle(cell_context)
            cell_context = cell_context[:self.cell_max]
        col_context = sample['col_context']
        if len(col_context) > self.col_max:
            random.shuffle(col_context)
            col_context = col_context[:self.col_max]
        row_context = sample['row_context']
        if len(row_context) > self.row_max:
            random.shuffle(row_context)
            row_context = row_context[:self.row_max]

        c_g, c_features = self.construct_graph(sample['mention'], cell_context, col_context)
        r_g, r_features = self.construct_graph(sample['mention'], cell_context, row_context)
        abstract = self.encode_abstract(sample['abstract'])
        return c_g, c_features, r_g, r_features, abstract

    def encode_abstract(self, abstract):
        """
        :param abstract:
        :return: [pad_seq_len, emb_dim]
        """
        res = []
        for t in tokenizer(abstract):
            if len(res) >= self.abstract_max:
                break
            if t in self.abstract_encode and self.abstract_encode[t] is not None:
                res.append(self.abstract_encode[t])

        for i in range(len(res), self.abstract_max):
            res.append(self.abstract_encode['<pad>'])
        return np.array(res)

    def construct_graph(self, m, cell_context, some_context):
        g = dgl.DGLGraph()
        # mention, cell mentions, col mentions
        g.add_nodes(num=1 + len(cell_context) + len(some_context))
        for i in range(1 + len(cell_context)):
            for j in range(1 + len(cell_context)):
                if i != j:  # 先不加自环
                    g.add_edge(i, j)
        for i in range(1 + len(cell_context), 1 + len(cell_context) + len(some_context)):
            g.add_edge(0, i)
            g.add_edge(i, 0)  # 无向图

        ndata = [self.get_mention_emb(m)]
        for i in range(len(cell_context)):  # 1, ..., 1+len(cell_context)
            ndata.append(self.get_mention_emb(cell_context[i]))
        for i in range(len(some_context)):  # 1+len(cell_context), 1+len(cell_context)+len(some_context)
            ndata.append(self.get_mention_emb(some_context[i]))

        ndata = np.array(ndata)
        features = torch.from_numpy(ndata).float()
        return g, features

    def get_mention_emb(self, m):
        """
        找不到的mention随机初始化
        :param m:
        :return:
        """
        if m in self.mention_encode and self.mention_encode[m] is not None:
            return self.mention_encode[m]
        else:
            return self.mention_encode['<unk>']


class AttentionSampleGenerator(SampleGenerator):
    def __init__(self, mention_abstract_emb_fp, cell_max=5, row_max=5, col_max=30, abstract_max=100):
        super().__init__(mention_abstract_emb_fp, cell_max, row_max, col_max, abstract_max)

    def process_sample(self, sample):
        """
        为一对m, e；构建符合深度学习模型的数据
        :param sample: {
            'cell_context': [...],
            'col_context': [...],
            'row_context': [...],
            'mention': '...',
            'abstract': '...',
        }
        :return:
        """
        cell_context = sample['cell_context']
        if len(cell_context) > self.cell_max:
            random.shuffle(cell_context)
            cell_context = cell_context[:self.cell_max]
        col_context = sample['col_context']
        if len(col_context) > self.col_max:
            random.shuffle(col_context)
            col_context = col_context[:self.col_max]
        row_context = sample['row_context']
        if len(row_context) > self.row_max:
            random.shuffle(row_context)
            row_context = row_context[:self.row_max]

        c_g, c_features = self.construct_graph(sample['mention'], cell_context, col_context)
        r_g, r_features = self.construct_graph(sample['mention'], cell_context, row_context)
        abstract = self.encode_abstract(sample['abstract'])
        entity_emb_repeat = np.array([sample['embedding'] for i in range(self.abstract_max)])
        mention_emb = self.get_mention_emb(sample['mention'])
        return c_g, c_features, r_g, r_features, abstract, entity_emb_repeat, mention_emb


class MentionEntityDataset(Dataset):
    """
    语义匹配数据集
    每条数据，包含，(mention, entity) 如果匹配，则标识1， 否则标识0
    是个二分类问题
    """
    def __init__(self, data, mention_abstract_emb_fp, cell_max=5, row_max=5, col_max=30, abstract_max=100):
        print('data', len(data))
        self.sample_generator = SampleGenerator(mention_abstract_emb_fp,
                                                cell_max, row_max, col_max, abstract_max)
        self.row_graphs = []
        self.row_features = []
        self.col_graphs = []
        self.col_features = []
        self.abstracts = []
        self.labels = []

        cnt = 0
        for item in tqdm(data):
            c_g, c_features, r_g, r_features, abstract = \
                self.sample_generator.process_sample(item)
            self.col_graphs.append(c_g)
            self.col_features.append(c_features)
            self.row_graphs.append(r_g)
            self.row_features.append(r_features)
            self.abstracts.append(abstract)
            self.labels.append(item['label'])

    def __getitem__(self, i):
        return self.row_graphs[i], self.row_features[i], \
               self.col_graphs[i], self.col_features[i], \
               self.abstracts[i], self.labels[i]

    def __len__(self):
        return len(self.abstracts)


class AttentionMentionEntityDataset(Dataset):
    """
    语义匹配数据集
    每条数据，包含，(mention, entity) 如果匹配，则标识1， 否则标识0
    是个二分类问题
    （添加了attention数据集）
    """
    def __init__(self, data, mention_abstract_emb_fp, cell_max=5, row_max=5, col_max=30, abstract_max=100):
        print('data', len(data))
        self.sample_generator = AttentionSampleGenerator(mention_abstract_emb_fp,
                                                         cell_max, row_max, col_max, abstract_max)
        self.row_graphs = []
        self.row_features = []
        self.col_graphs = []
        self.col_features = []
        self.abstracts = []
        self.entities_emb = []
        self.mention_emb = []
        self.labels = []

        cnt = 0
        for item in tqdm(data):
            c_g, c_features, r_g, r_features, abstract, entities_emb, mention_emb = \
                self.sample_generator.process_sample(item)
            self.col_graphs.append(c_g)
            self.col_features.append(c_features)
            self.row_graphs.append(r_g)
            self.row_features.append(r_features)
            self.abstracts.append(abstract)
            self.entities_emb.append(entities_emb)
            self.mention_emb.append(mention_emb)
            self.labels.append(item['label'])

    def __getitem__(self, i):
        return self.row_graphs[i], self.row_features[i], \
               self.col_graphs[i], self.col_features[i], \
               self.abstracts[i], self.entities_emb[i], \
               self.mention_emb[i], self.labels[i]

    def __len__(self):
        return len(self.abstracts)


if __name__ == '__main__':
    pass