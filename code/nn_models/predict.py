import torch
import dgl
from nn_models.model import Classifier, AttentionClassifier
from nn_models.data import *
import nn_models.nn_config as nn_config


class Predictor:
    def __init__(self, model_fp, mention_abstract_emb_fp):
        self.sample_generator = SampleGenerator(mention_abstract_emb_fp)
        self.model = Classifier(g_in_dim=nn_config.g_in_dim, g_hidden_dim=nn_config.g_hidden_dim,
                                l_in_dim=nn_config.l_in_dim, l_hidden_dim=nn_config.l_hidden_dim,
                                l_num_layers=nn_config.l_num_layers, l_dropout=nn_config.l_dropout).to(nn_config.device)
        print('load model...', model_fp)
        if nn_config.device == 'cuda':
            self.model.load_state_dict(torch.load(model_fp))
        else:
            self.model.load_state_dict(torch.load(model_fp, map_location='cpu'))
        self.model.eval()

    def collate_fn(self, samples):
        """
        转成batch数据
        :param samples:
        :return:
        """
        row_graphs, row_features, col_graphs, col_features, abstracts, labels \
            = [], [], [], [], [], []
        for item in samples:
            c_g, c_features, r_g, r_features, abstract = \
                self.sample_generator.process_sample(item)
            col_graphs.append(c_g)
            col_features.append(c_features)
            row_graphs.append(r_g)
            row_features.append(r_features)
            abstracts.append(abstract)
            labels.append(-1)

        batch_row_graphs = dgl.batch(row_graphs)
        # batch_row_features = torch.FloatTensor(torch.from_numpy(np.array(row_features)))
        batch_row_features = torch.FloatTensor(torch.cat(row_features))
        row_m_indices = [0]
        for g in row_graphs:
            row_m_indices.append(row_m_indices[-1] + len(g.nodes()))
        row_m_indices.pop()
        row_m_indices = torch.LongTensor(row_m_indices)

        batch_col_graphs = dgl.batch(col_graphs)
        # batch_col_features = torch.FloatTensor(torch.from_numpy(np.array(col_features)))
        batch_col_features = torch.FloatTensor(torch.cat(col_features))
        col_m_indices = [0]
        for g in col_graphs:
            col_m_indices.append(col_m_indices[-1] + len(g.nodes()))
        col_m_indices.pop()
        col_m_indices = torch.LongTensor(col_m_indices)

        batch_abstracts = torch.FloatTensor(abstracts)
        batch_labels = torch.LongTensor(labels)
        return batch_row_graphs, batch_row_features.to(nn_config.device), row_m_indices.to(nn_config.device), \
               batch_col_graphs, batch_col_features.to(nn_config.device), col_m_indices.to(nn_config.device), \
               batch_abstracts.to(nn_config.device), batch_labels.to(nn_config.device)

    def predict_one(self, sample):
        """
        :param sample: {
            'cell_context': [...],
            'col_context': [...],
            'row_context': [...],
            'mention': '...',
            'abstract': '...',
        }
        :return: 一个标量值
        """
        # 直接通过batch的操作进行封装，简单易行
        fake_sample = {'c_idx': 2, 'label': 1, 'mention': 'Los Angeles Memorial Sports Arena', 'cell_context': [], 'r_idx': 4, 'entity': 'Los_Angeles_Memorial_Sports_Arena', 'abstract': "Los Angeles Memorial Sports Arena\n\nThe Los Angeles Memorial Sports Arena was a multi-purpose  ENTITY/arena  at  ENTITY/Exposition_Park_(Los_Angeles) , in the  ENTITY/University_Park,_Los_Angeles  neighborhood of Los Angeles. It was located next to the  ENTITY/Los_Angeles_Memorial_Coliseum  and just south of the campus of the  ENTITY/University_of_Southern_California , which managed and operated both venues under a master lease agreement with the Los Angeles Memorial Coliseum Commission. The arena was demolished in 2016 and replaced with  ENTITY/Banc_of_California_Stadium , home of  ENTITY/Major_League_Soccer 's  ENTITY/Los_Angeles_FC  which opened in 2018.\n", 'col_context': ['Drake Fieldhouse', 'Cole Field House', 'Old Dominion University Fieldhouse', 'Memorial Coliseum', 'Ahearn Field House', 'Stokely Athletic Center', 'Reynolds Coliseum', 'McArthur Court', 'Memorial Gym', 'Carolina Coliseum'], 'name': '0.json', 'row_context': ['Los Angeles', 'California', 'University of Southern California']}
        samples = [sample, fake_sample]
        return self.predict_batch(samples)[0]

    def predict_batch(self, samples):
        """
        :param sample: [{
            'cell_context': [...],
            'col_context': [...],
            'row_context': [...],
            'mention': '...',
            'abstract': '...',
        },
        ...]
        :return: 一个list
        """
        assert len(samples) >= 2, 'batch size %d, should >= 2' % len(samples)
        with torch.no_grad():
            batch_row_graphs, batch_row_features, row_m_indices, \
            batch_col_graphs, batch_col_features, col_m_indices, \
            batch_abstracts, batch_labels = self.collate_fn(samples)
            pred = self.model(batch_row_graphs, batch_row_features, row_m_indices,
                              batch_col_graphs, batch_col_features, col_m_indices,
                              batch_abstracts)
            pred = torch.sigmoid(pred).to('cpu')
            pred = [p[0].item() for p in pred]
            return pred  # 最终预测结果，直接转到内存中


class AttentionPredictor:
    def __init__(self, model_fp, mention_abstract_emb_fp, has_lstm=True):
        self.sample_generator = AttentionSampleGenerator(mention_abstract_emb_fp)
        self.model = AttentionClassifier(g_in_dim=nn_config.g_in_dim,
                                         g_hidden_dim=nn_config.g_hidden_dim,
                                         g_attention_dim=nn_config.g_attention_dim,
                                         l_in_dim=nn_config.l_in_dim,
                                         l_hidden_dim=nn_config.l_hidden_dim,
                                         l_attention_dim=nn_config.l_attention_dim,
                                         l_num_layers=nn_config.l_num_layers,
                                         l_dropout=nn_config.l_dropout,
                                         has_lstm=has_lstm).to(nn_config.device)
        print('load model...', model_fp)
        if nn_config.device == 'cuda':
            self.model.load_state_dict(torch.load(model_fp))
        else:
            self.model.load_state_dict(torch.load(model_fp, map_location='cpu'))
        self.model.eval()

    def collate_fn(self, samples):
        """
        转成batch数据
        :param samples:
        :return:
        """
        row_graphs, row_features, col_graphs, col_features, \
        abstracts, entities_emb, mention_emb, labels = [], [], [], [], [], [], [], []
        for item in samples:
            c_g, c_features, r_g, r_features, abstract, entity_emb, m_emb = \
                self.sample_generator.process_sample(item)
            col_graphs.append(c_g)
            col_features.append(c_features)
            row_graphs.append(r_g)
            row_features.append(r_features)
            abstracts.append(abstract)
            entities_emb.append(entity_emb)
            mention_emb.append(m_emb)
            labels.append(-1)

        batch_row_graphs = dgl.batch(row_graphs)
        # batch_row_features = torch.FloatTensor(torch.from_numpy(np.array(row_features)))
        batch_row_features = torch.FloatTensor(torch.cat(row_features))
        row_m_indices = [0]
        for g in row_graphs:
            row_m_indices.append(row_m_indices[-1] + len(g.nodes()))
        row_m_indices.pop()
        row_m_indices = torch.LongTensor(row_m_indices)

        batch_col_graphs = dgl.batch(col_graphs)
        # batch_col_features = torch.FloatTensor(torch.from_numpy(np.array(col_features)))
        batch_col_features = torch.FloatTensor(torch.cat(col_features))
        col_m_indices = [0]
        for g in col_graphs:
            col_m_indices.append(col_m_indices[-1] + len(g.nodes()))
        col_m_indices.pop()
        col_m_indices = torch.LongTensor(col_m_indices)
        batch_abstracts = torch.FloatTensor(abstracts)
        batch_entities_emb = torch.FloatTensor(entities_emb)
        batch_mention_emb = torch.FloatTensor(mention_emb)
        batch_labels = torch.LongTensor(labels)
        return batch_row_graphs, batch_row_features.to(nn_config.device), row_m_indices.to(nn_config.device), \
               batch_col_graphs, batch_col_features.to(nn_config.device), col_m_indices.to(nn_config.device), \
               batch_abstracts.to(nn_config.device), batch_entities_emb.to(nn_config.device), \
               batch_mention_emb.to(nn_config.device), batch_labels.to(nn_config.device)

    def predict_one(self, sample):
        """
        :param sample: {
            'cell_context': [...],
            'col_context': [...],
            'row_context': [...],
            'mention': '...',
            'abstract': '...',
        }
        :return: 一个标量值
        """
        # 直接通过batch的操作进行封装，简单易行
        fake_sample = {'embedding': [0.0] * nn_config.entity_dim, 'c_idx': 2, 'label': 1, 'mention': 'Los Angeles Memorial Sports Arena', 'cell_context': [], 'r_idx': 4, 'entity': 'Los_Angeles_Memorial_Sports_Arena', 'abstract': "Los Angeles Memorial Sports Arena\n\nThe Los Angeles Memorial Sports Arena was a multi-purpose  ENTITY/arena  at  ENTITY/Exposition_Park_(Los_Angeles) , in the  ENTITY/University_Park,_Los_Angeles  neighborhood of Los Angeles. It was located next to the  ENTITY/Los_Angeles_Memorial_Coliseum  and just south of the campus of the  ENTITY/University_of_Southern_California , which managed and operated both venues under a master lease agreement with the Los Angeles Memorial Coliseum Commission. The arena was demolished in 2016 and replaced with  ENTITY/Banc_of_California_Stadium , home of  ENTITY/Major_League_Soccer 's  ENTITY/Los_Angeles_FC  which opened in 2018.\n", 'col_context': ['Drake Fieldhouse', 'Cole Field House', 'Old Dominion University Fieldhouse', 'Memorial Coliseum', 'Ahearn Field House', 'Stokely Athletic Center', 'Reynolds Coliseum', 'McArthur Court', 'Memorial Gym', 'Carolina Coliseum'], 'name': '0.json', 'row_context': ['Los Angeles', 'California', 'University of Southern California']}
        samples = [sample, fake_sample]
        return self.predict_batch(samples)[0]

    def predict_batch(self, samples):
        """
        :param sample: [{
            'cell_context': [...],
            'col_context': [...],
            'row_context': [...],
            'mention': '...',
            'abstract': '...',
        },
        ...]
        :return: 一个list
        """
        assert len(samples) >= 2, 'batch size %d, should >= 2' % len(samples)
        with torch.no_grad():
            batch_row_graphs, batch_row_features, row_m_indices, \
            batch_col_graphs, batch_col_features, col_m_indices, \
            batch_abstracts, batch_entities_emb, batch_mention_emb, \
            batch_labels = self.collate_fn(samples)
            pred = self.model(batch_row_graphs, batch_row_features, row_m_indices,
                              batch_col_graphs, batch_col_features, col_m_indices,
                              batch_abstracts, batch_entities_emb, batch_mention_emb)
            pred = torch.sigmoid(pred).to('cpu')
            pred = [p[0].item() for p in pred]
            return pred  # 最终预测结果，直接转到内存中
