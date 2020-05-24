import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import dgl
from dgl.data import MiniGCDataset
import networkx as nx
import dgl.function as fn

import nn_models.nn_config as nn_config

# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')


def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}


class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        # feature: [node_size, in_feats]
        # g: [node_size, ]
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        res = g.ndata.pop('h')
        # res: [node_size, out_feats]
        return res


class AttentionNodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, attn_encode_dim, attention_dim, activation):
        super(AttentionNodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.w_h = torch.zeros(out_feats, attention_dim).requires_grad_(True).to(nn_config.device)
        self.w_e = torch.zeros(attn_encode_dim, attention_dim).requires_grad_(True).to(nn_config.device)
        self.u = torch.zeros(attention_dim).requires_grad_(True).to(nn_config.device)

    def attention_net(self, h, entities_encode):
        # h: [node_size, out_feats]
        # entities_encode: [node_size, e_encode_dim]

        # [node_size, e_encode_dim] * [e_encode_dim, attn_dim]
        #  + [node_size, out_feats] * [out_feats, attn_dim]
        #  = [node_size, attn_dim]
        attn_tanh = torch.tanh(torch.mm(entities_encode, self.w_e) + torch.mm(h, self.w_h))
        # attn_tanh = [node_size, attn_dim]
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u, [-1, 1]))
        # attn_hidden_layer: [node_size, 1]
        exp = torch.exp(attn_hidden_layer)
        # exp: [node_size, 1]
        alpha = exp / torch.sum(exp)
        # alpha: [node_size, 1]
        h = h * alpha  # 每个节点的隐藏层状态，乘以新权重
        # h: [node_size, out_feats]
        return h

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        h = self.attention_net(h, node.data['a'])
        return {'h': h}


class AGCN(nn.Module):
    """
    GCN, attented by entity encode
    """

    def __init__(self, in_feats, out_feats, attn_encode_dim, attention_dim, activation):
        super(AGCN, self).__init__()
        self.apply_mod = AttentionNodeApplyModule(in_feats, out_feats, attn_encode_dim, attention_dim, activation)

    def forward(self, g, feature, attn_encode):
        # Initialize the node features with h.
        g.ndata['h'] = feature

        g.ndata['a'] = attn_encode  # [node_size, attn_encode_dim]
        # 引入用来进行attention的向量，曾经是实体编码，现在是mention自己编码

        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        h = g.ndata.pop('h')
        return h


class BiLSTMEncoder(nn.Module):
    """
    使用BiLSTM对序列进行编码
    """
    def __init__(self, input_size, hidden_size=100,
                 num_layers=1, dropout=0.):
        super(BiLSTMEncoder, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            bias=True, dropout=dropout, bidirectional=True)

    def forward(self, text_emb):
        """
        :param text_emb: [batch_size, pad_seq_len, emb_dim]
        :return: [batch_size, hid dim * num direct]
        """
        text_emb = text_emb.permute(1, 0, 2)
        # [pat_seq_len, batch_size, emb_dim]

        output, (hidden, cell) = self.lstm(text_emb)

        # output = [sent len, batch size, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # hidden[-2, :, :] --> forward_layer_n
        # hidden[-1, :, :] --> backward_layer_n
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # hidden = [1, batch size, hid dim * num directions]  ?
        hidden = hidden.squeeze(0)
        # hidden: [batch_size, hid dim * num direct]
        return hidden


class AttentionBiLSTMEncoder(nn.Module):
    """
    LSTM中加入e作为attention
    """
    def __init__(self, input_size, hidden_size=100, attention_size=128, num_layers=1, dropout=0.):
        super(AttentionBiLSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            bias=True, dropout=dropout, bidirectional=True)

        self.w_h = torch.zeros(hidden_size * 2, attention_size).requires_grad_(True).to(nn_config.device)
        self.w_e = torch.zeros(input_size, attention_size).requires_grad_(True).to(nn_config.device)
        # hidden_size * 2: 双向
        self.u = torch.zeros(attention_size).requires_grad_(True).to(nn_config.device)

    def attention_net(self, lstm_output, entities_emb):
        # lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        """
        print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)
        print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)
        print(attn_tanh.size())  (squence_length * batch_size, attention_size)
        print(attn_hidden_layer.size())  (squence_length * batch_size, 1)
        print(exps.size())  (batch_size, squence_length)
        print(alphas.size()) (batch_size, squence_length)
        print(alphas_reshape.size()) = (batch_size, squence_length, 1)
        print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)
        print(attn_output.size()) = (batch_size, hidden_size*layer_size)
        """
        entities_reshape = torch.Tensor.reshape(entities_emb, [-1, self.input_size])
        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * 2])
        # M = tanh(H)
        # attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_h))
        # [seq_len * b_size, attn_size] = [seq_len * b_size, hid_dim * 2] * [hid_dim * 2, attn_size]
        attn_tanh = torch.tanh(torch.mm(entities_reshape, self.w_e) + torch.mm(output_reshape, self.w_h))
        # attn_tanh: [seq_len * b_size, attn_size]

        # alpha = softmax(omega.T*M)
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u, [-1, 1]))
        # attn_hidden_layer: [seq_len * b_size, 1]

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, lstm_output.size()[0]])
        # exps: [b_size, seq_len]
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # alphas: [b_size, seq_len]
        alphas_reshape = torch.Tensor.reshape(alphas, [-1, lstm_output.size()[0], 1])
        # alphas_reshape: [b_size, seq_len, 1]
        state = lstm_output.permute(1, 0, 2)
        # state: [b_size, seq_len, hid dim * num direct]
        # r = H*alpha.T
        s = state * alphas_reshape
        # s: [b_size, seq_len, hid dim * num direct]
        attn_output = torch.sum(s, 1)
        return attn_output

    def forward(self, text_emb, entities_emb):
        """
        :param text_emb: [batch_size, pad_seq_len, emb_dim]
        :param entities_emb: [batch_size, pad_seq_len, emb_dim]
        :return: [batch_size, hid dim * num direct]
        """
        text_emb = text_emb.permute(1, 0, 2)
        # [pat_seq_len, batch_size, emb_dim]
        entities_emb = entities_emb.permute(1, 0, 2)
        # [pat_seq_len, batch_size, emb_dim]

        output, (hidden, cell) = self.lstm(text_emb)

        # # output = [sent len, batch size, hid dim * num directions]
        # # hidden = [num layers * num directions, batch size, hid dim]
        # # cell = [num layers * num directions, batch size, hid dim]
        #
        # # hidden[-2, :, :] --> forward_layer_n
        # # hidden[-1, :, :] --> backward_layer_n
        # hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # # hidden = [1, batch size, hid dim * num directions]  ?
        # hidden = hidden.squeeze(0)
        # # hidden: [batch_size, hid dim * num direct]
        attention_output = self.attention_net(output, entities_emb)

        # attention_output: [batch_size, hid dim * num direct]
        return attention_output


class AttentionLSTMEncoder(nn.Module):
    """
    LSTM中加入e作为attention
    """
    def __init__(self, input_size, hidden_size=100, attention_size=128, num_layers=1, dropout=0.):
        super(AttentionLSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            bias=True, dropout=dropout, bidirectional=True)

        self.w_h = torch.zeros(hidden_size, attention_size).requires_grad_(True).to(nn_config.device)
        self.w_e = torch.zeros(input_size, attention_size).requires_grad_(True).to(nn_config.device)
        self.u = torch.zeros(attention_size).requires_grad_(True).to(nn_config.device)

    def attention_net(self, lstm_output, entities_emb):
        # lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        """
        print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)
        print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)
        print(attn_tanh.size())  (squence_length * batch_size, attention_size)
        print(attn_hidden_layer.size())  (squence_length * batch_size, 1)
        print(exps.size())  (batch_size, squence_length)
        print(alphas.size()) (batch_size, squence_length)
        print(alphas_reshape.size()) = (batch_size, squence_length, 1)
        print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)
        print(attn_output.size()) = (batch_size, hidden_size*layer_size)
        """
        entities_reshape = torch.Tensor.reshape(entities_emb, [-1, self.input_size])
        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size])
        attn_tanh = torch.tanh(torch.mm(entities_reshape, self.w_e) + torch.mm(output_reshape, self.w_h))
        # attn_tanh: [seq_len * b_size, attn_size]

        # alpha = softmax(omega.T*M)
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u, [-1, 1]))
        # attn_hidden_layer: [seq_len * b_size, 1]

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, lstm_output.size()[0]])
        # exps: [b_size, seq_len]
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # alphas: [b_size, seq_len]
        alphas_reshape = torch.Tensor.reshape(alphas, [-1, lstm_output.size()[0], 1])
        # alphas_reshape: [b_size, seq_len, 1]
        state = lstm_output.permute(1, 0, 2)
        # state: [b_size, seq_len, hid dim * num direct]
        # r = H*alpha.T
        s = state * alphas_reshape
        # s: [b_size, seq_len, hid dim * num direct]
        attn_output = torch.sum(s, 1)
        return attn_output

    def forward(self, text_emb, entities_emb):
        """
        :param text_emb: [batch_size, pad_seq_len, emb_dim]
        :param entities_emb: [batch_size, pad_seq_len, emb_dim]
        :return: [batch_size, hid dim * num direct]
        """
        text_emb = text_emb.permute(1, 0, 2)
        # [pat_seq_len, batch_size, emb_dim]
        entities_emb = entities_emb.permute(1, 0, 2)
        # [pat_seq_len, batch_size, emb_dim]

        output, (hidden, cell) = self.lstm(text_emb)

        # # output = [sent len, batch size, hid dim * num directions]
        # # hidden = [num layers * num directions, batch size, hid dim]
        # # cell = [num layers * num directions, batch size, hid dim]
        #
        # # hidden[-2, :, :] --> forward_layer_n
        # # hidden[-1, :, :] --> backward_layer_n
        # hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # # hidden = [1, batch size, hid dim * num directions]  ?
        # hidden = hidden.squeeze(0)
        # # hidden: [batch_size, hid dim * num direct]
        attention_output = self.attention_net(output, entities_emb)

        # attention_output: [batch_size, hid dim * num direct]
        return attention_output


if __name__ == '__main__':
    pass
