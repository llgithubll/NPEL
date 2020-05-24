import os
import dgl
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
import random

from nn_models.model import *
from nn_models.data import *
import nn_models.nn_config as nn_config
from utils import create_dir


class Trainer:
    def __init__(self, batch_size, n_epoches, out_dir, train_data_fp, eval_size=0.2, topn_table=400):
        self.batch_size = batch_size
        self.n_epoches = n_epoches
        self.out_dir = out_dir
        create_dir(out_dir)
        self.device = nn_config.device

        self.model = Classifier(g_in_dim=nn_config.g_in_dim, g_hidden_dim=nn_config.g_hidden_dim,
                                l_in_dim=nn_config.l_in_dim, l_hidden_dim=nn_config.l_hidden_dim,
                                l_num_layers=nn_config.l_num_layers, l_dropout=nn_config.l_dropout).to(self.device)
        print(self.model)
        parameters_cnt = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_parameters_cnt = sum(p.numel() for p in self.model.parameters())
        print('The model has ', parameters_cnt, ' trainable parameters; total ', total_parameters_cnt, ' parameters')

        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion.to(self.device)

        with open(train_data_fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
            new_data = []
            table_name_set = set()
            for item in data:
                table_name_set.add(int(item['name'].split('.')[0]))
                if len(table_name_set) >= topn_table:  # 前400个表格，大概100000对(m,e)
                    break
                new_data.append(item)
            data = new_data
            print('total tables', topn_table, 'max table name', max(table_name_set))
            print('total (m,e) pairs', len(data))
            label = []
            for item in data:
                label.append(item['label'])

        train_data, val_data, train_label, val_label = train_test_split(data, label, test_size=eval_size, shuffle=True)
        train_dataset = MentionEntityDataset(data=train_data,
                                             mention_abstract_emb_fp=nn_config.mention_abstract_emb_fp)
        val_dataset = MentionEntityDataset(data=val_data,
                                           mention_abstract_emb_fp=nn_config.mention_abstract_emb_fp)
        print('train dataset', len(train_dataset), 'val dataset', len(val_dataset))
        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        row_graphs, row_features, \
        col_graphs, col_features, \
        abstracts, labels = map(list, zip(*batch))

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
        return batch_row_graphs, batch_row_features.to(self.device), row_m_indices.to(self.device), \
               batch_col_graphs, batch_col_features.to(self.device), col_m_indices.to(self.device), \
               batch_abstracts.to(self.device), batch_labels.to(self.device)

    def binary_accuracy(self, preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """
        # round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()  # convert into float for division
        acc = correct.sum() / len(correct)
        return acc

    def train_epoch(self):
        epoch_loss = 0.
        epoch_acc = 0.
        self.model.train()

        for i, batch in enumerate(self.train_loader):
            batch_row_graphs, batch_row_features, row_m_indices, \
            batch_col_graphs, batch_col_features, col_m_indices, \
            batch_abstracts, batch_labels = batch
            self.optimizer.zero_grad()
            pred = self.model(batch_row_graphs, batch_row_features, row_m_indices,
                              batch_col_graphs, batch_col_features, col_m_indices,
                              batch_abstracts)
            pred = pred.squeeze(1)
            loss = self.criterion(pred, batch_labels.float())
            acc = self.binary_accuracy(pred, batch_labels.float())

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / len(self.train_loader), epoch_acc / len(self.train_loader)

    def eval_epoch(self):
        epoch_loss = 0.
        epoch_acc = 0.
        self.model.eval()

        for i, batch in enumerate(self.val_loader):
            batch_row_graphs, batch_row_features, row_m_indices, \
            batch_col_graphs, batch_col_features, col_m_indices, \
            batch_abstracts, batch_labels = batch
            pred = self.model(batch_row_graphs, batch_row_features, row_m_indices,
                              batch_col_graphs, batch_col_features, col_m_indices,
                              batch_abstracts)
            pred = pred.squeeze(1)
            loss = self.criterion(pred, batch_labels.float())
            acc = self.binary_accuracy(pred, batch_labels.float())

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / len(self.val_loader), epoch_acc / len(self.val_loader)

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def run(self):
        best_valid_loss = float('inf')
        for epoch in range(self.n_epoches):
            start_time = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.eval_epoch()
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.out_dir, 'best_model'))
                print('\nSaving in', self.out_dir)

            print('Epoch: ', epoch + 1, ' | Epoch Time: ', epoch_mins, 'm ', epoch_secs, 's')
            print('\tTrain Loss: {%.3f} | Train Acc: {%.2f}' % (train_loss, train_acc))
            print('\t Val. Loss: {%.3f} |  Val. Acc: {%.2f}' % (val_loss, val_acc))


class AttentionTrainer:
    def __init__(self, batch_size, n_epoches, out_dir, train_data_fp, eval_size=0.2, topn_table=400, has_lstm=True):
        print('Attention Trainer')
        self.batch_size = batch_size
        self.n_epoches = n_epoches
        self.out_dir = out_dir
        create_dir(out_dir)
        self.device = nn_config.device

        self.model = AttentionClassifier(g_in_dim=nn_config.g_in_dim,
                                         g_hidden_dim=nn_config.g_hidden_dim,
                                         g_attention_dim=nn_config.g_attention_dim,
                                         l_in_dim=nn_config.l_in_dim,
                                         l_hidden_dim=nn_config.l_hidden_dim,
                                         l_attention_dim=nn_config.l_attention_dim,
                                         l_num_layers=nn_config.l_num_layers,
                                         l_dropout=nn_config.l_dropout,
                                         has_lstm=has_lstm).to(self.device)

        print(self.model)
        parameters_cnt = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_parameters_cnt = sum(p.numel() for p in self.model.parameters())
        print('The model has ', parameters_cnt, ' trainable parameters; total ', total_parameters_cnt, ' parameters')

        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion.to(self.device)

        with open(train_data_fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
            new_data = []
            table_name_set = set()
            for item in data:
                table_name_set.add(int(item['name'].split('.')[0]))
                if len(table_name_set) >= topn_table:  # 前400个表格，大概100000对(m,e)
                    break
                new_data.append(item)
            data = new_data
            print('total tables', topn_table, 'max table name', max(table_name_set))
            print('total (m,e) pairs', len(data))
            label = []
            for item in data:
                label.append(item['label'])

        train_data, val_data, train_label, val_label = train_test_split(data, label, test_size=eval_size, shuffle=True)
        train_dataset = AttentionMentionEntityDataset(data=train_data,
                                                      mention_abstract_emb_fp=nn_config.mention_abstract_emb_fp)
        val_dataset = AttentionMentionEntityDataset(data=val_data,
                                                    mention_abstract_emb_fp=nn_config.mention_abstract_emb_fp)
        print('train dataset', len(train_dataset), 'val dataset', len(val_dataset))
        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        row_graphs, row_features, \
        col_graphs, col_features, \
        abstracts, entities_emb, \
        mention_emb, labels = map(list, zip(*batch))

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

        return batch_row_graphs, batch_row_features.to(self.device), row_m_indices.to(self.device), \
               batch_col_graphs, batch_col_features.to(self.device), col_m_indices.to(self.device), \
               batch_abstracts.to(self.device), batch_entities_emb.to(self.device), \
               batch_mention_emb.to(self.device), batch_labels.to(self.device)

    def binary_accuracy(self, preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """
        # round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()  # convert into float for division
        acc = correct.sum() / len(correct)
        return acc

    def train_epoch(self):
        epoch_loss = 0.
        epoch_acc = 0.
        self.model.train()

        for i, batch in enumerate(self.train_loader):
            batch_row_graphs, batch_row_features, row_m_indices, \
            batch_col_graphs, batch_col_features, col_m_indices, \
            batch_abstracts, batch_entities_emb, batch_mention_emb, batch_labels = batch
            self.optimizer.zero_grad()
            pred = self.model(batch_row_graphs, batch_row_features, row_m_indices,
                              batch_col_graphs, batch_col_features, col_m_indices,
                              batch_abstracts, batch_entities_emb, batch_mention_emb)
            pred = pred.squeeze(1)
            loss = self.criterion(pred, batch_labels.float())
            acc = self.binary_accuracy(pred, batch_labels.float())

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / len(self.train_loader), epoch_acc / len(self.train_loader)

    def eval_epoch(self):
        epoch_loss = 0.
        epoch_acc = 0.
        self.model.eval()

        for i, batch in enumerate(self.val_loader):
            batch_row_graphs, batch_row_features, row_m_indices, \
            batch_col_graphs, batch_col_features, col_m_indices, \
            batch_abstracts, batch_entities_emb, batch_mention_emb, batch_labels = batch
            pred = self.model(batch_row_graphs, batch_row_features, row_m_indices,
                              batch_col_graphs, batch_col_features, col_m_indices,
                              batch_abstracts, batch_entities_emb, batch_mention_emb)
            pred = pred.squeeze(1)
            loss = self.criterion(pred, batch_labels.float())
            acc = self.binary_accuracy(pred, batch_labels.float())

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / len(self.val_loader), epoch_acc / len(self.val_loader)

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def run(self):
        best_valid_loss = float('inf')
        for epoch in range(self.n_epoches):
            start_time = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.eval_epoch()
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.out_dir, 'best_model'))
                print('\nSaving in', self.out_dir)

            print('Epoch: ', epoch + 1, ' | Epoch Time: ', epoch_mins, 'm ', epoch_secs, 's')
            print('\tTrain Loss: {%.3f} | Train Acc: {%.2f}' % (train_loss, train_acc))
            print('\t Val. Loss: {%.3f} |  Val. Acc: {%.2f}' % (val_loss, val_acc))


if __name__ == '__main__':
    pass
