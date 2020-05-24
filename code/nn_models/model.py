from nn_models.layers import *


class Classifier(nn.Module):
    """
    GCN: mention上下文编码
    LSTM: 实体编码
    """
    def __init__(self, g_in_dim, g_hidden_dim,
                 l_in_dim, l_hidden_dim, l_num_layers, l_dropout):
        """
        :param g_in_dim: 图卷积的输入维度
        :param g_hidden_dim: 图卷积的隐藏层维度
        :param l_hidden_dim: LSTM的隐藏层维度
        """
        super(Classifier, self).__init__()

        self.row_gcn_layers = nn.ModuleList([
            GCN(g_in_dim, g_hidden_dim, F.relu),
            GCN(g_hidden_dim, g_hidden_dim, F.relu)
        ])

        self.col_gcn_layers = nn.ModuleList([
            GCN(g_in_dim, g_hidden_dim, F.relu),
            GCN(g_hidden_dim, g_hidden_dim, F.relu)
        ])

        self.lstm = BiLSTMEncoder(input_size=l_in_dim,
                                  hidden_size=l_hidden_dim,
                                  num_layers=l_num_layers,
                                  dropout=l_dropout)

        # gcn_row, gcn_col, lstm 编码拼接，作二分类
        self.linear = nn.Linear(g_hidden_dim + g_hidden_dim + l_hidden_dim * 2,
                                1)  # output_dim=1表示是正例的概率

    def forward(self, row_batch_graph, row_batch_features, row_m_indices,
                col_batch_graph, col_batch_features, col_m_indices,
                batch_abstract):
        """
        :param row_batch_graph: mention行上下文图
        :param row_batch_features: 每个mention的特征向量
        :param row_m_indices: 每个mention在batch图中的位置
        :param col_batch_graph: mention列上下文图
        :param col_batch_features: 每个mention的特征向量
        :param col_m_indices: 每个mention在batch图中的位置
        :param batch_abstract: 实体对应摘要 [batch_size, pad_seq_len, emb_dim]
        :return:
        """
        # r_h = row_batch_graph.in_degrees().view(-1, 1).float()
        r_h = row_batch_features
        # r_h: [batch_nodes_size, g_in_dim]
        for r_conv in self.row_gcn_layers:
            r_h = r_conv(row_batch_graph, r_h)
        row_batch_graph.ndata['h'] = r_h
        # r_h: [batch_nodes_size, g_hidden_dim]
        r_h = r_h[row_m_indices]
        # r_h: [batch_size, g_hidden_dim]

        # c_h = col_batch_graph.in_degrees().view(-1, 1).float()
        c_h = col_batch_features
        # c_h: [batch_nodes_size, g_in_dim]
        for c_conv in self.col_gcn_layers:
            c_h = c_conv(col_batch_graph, c_h)
        col_batch_graph.ndata['h'] = c_h
        # c_h: [batch_nodes_size, g_hidden_dim]
        c_h = c_h[col_m_indices]
        # c_h: [batch_size, g_hidden_dim]

        # batch_abstract: [batch_size, pad_seq_len, emb_dim]
        entity_encode = self.lstm(batch_abstract)
        # entity_encode: [batch_size, l_hidden_dim * 2]

        cat = torch.cat([r_h, c_h, entity_encode], dim=1)
        # cat: [batch_size, g_hidden_dim + g_hidden_dim + l_hidden_dim * 2]

        return self.linear(cat)  # [batch_size, 1]  # 最终匹配得分


class AttentionClassifier(nn.Module):
    """
    Attention GCN: mention上下文编码
    Attention LSTM: 实体编码
    """
    def __init__(self, g_in_dim, g_hidden_dim, g_attention_dim,
                 l_in_dim, l_hidden_dim, l_attention_dim, l_num_layers, l_dropout,
                 has_lstm=True):
        """
        :param g_in_dim: 图卷积的输入维度
        :param g_hidden_dim: 图卷积的隐藏层维度
        :param l_hidden_dim: LSTM的隐藏层维度
        """
        super(AttentionClassifier, self).__init__()
        self.has_lstm = has_lstm

        self.lstm = AttentionLSTMEncoder(input_size=l_in_dim,
                                         hidden_size=l_hidden_dim,
                                         attention_size=l_attention_dim,
                                         num_layers=l_num_layers,
                                         dropout=l_dropout)

        self.row_gcn_layers = nn.ModuleList([
            AGCN(g_in_dim, g_hidden_dim, g_in_dim, g_attention_dim, F.relu),
            AGCN(g_hidden_dim, g_hidden_dim, g_in_dim, g_attention_dim, F.relu)
        ])

        self.col_gcn_layers = nn.ModuleList([
            AGCN(g_in_dim, g_hidden_dim, g_in_dim, g_attention_dim, F.relu),
            AGCN(g_hidden_dim, g_hidden_dim, g_in_dim, g_attention_dim, F.relu)
        ])

        # gcn_row, gcn_col, lstm 编码拼接，作二分类
        self.linear = nn.Linear(g_hidden_dim + g_hidden_dim + l_hidden_dim,
                                1)  # output_dim=1表示是正例的概率
        if not has_lstm:
            self.entity_linear = nn.Linear(l_in_dim, l_hidden_dim)  # 为了送入gcn

    def forward(self, row_batch_graph, row_batch_features, row_m_indices,
                col_batch_graph, col_batch_features, col_m_indices,
                batch_abstract, batch_entities, batch_mention):
        """
        :param row_batch_graph: mention行上下文图
        :param row_batch_features: 每个mention的特征向量
        :param row_m_indices: 每个mention在batch图中的位置
        :param col_batch_graph: mention列上下文图
        :param col_batch_features: 每个mention的特征向量
        :param col_m_indices: 每个mention在batch图中的位置
        :param batch_abstract: 实体对应摘要 [batch_size, pad_seq_len, emb_dim]
        :param batch_entities: 预训练的词向量 [batch_size, pad_seq_len, emb_dim]
        :param batch_mention: mention的预训练向量 [batch_size, emb_dim]
        :return:
        """
        if self.has_lstm:
            # batch_abstract: [batch_size, pad_seq_len, emb_dim]
            entity_encode = self.lstm(batch_abstract, batch_entities)
        else:
            batch_entities = batch_entities.permute(0, 2, 1)
            entity_encode = self.entity_linear(batch_entities[:, :, 0])
        # 将 batch_mention(即attn_encode)扩展为[batch_nodes_size, g_in_dim]
        row_nodes_attn_encode = []
        for i, n in enumerate(row_batch_graph.batch_num_nodes):
            row_nodes_attn_encode.append(batch_mention[i].repeat(n, 1))
        row_nodes_attn_encode = torch.cat(row_nodes_attn_encode)
        # r_h = row_batch_graph.in_degrees().view(-1, 1).float()
        r_h = row_batch_features
        # r_h: [batch_nodes_size, g_in_dim]
        for r_conv in self.row_gcn_layers:
            r_h = r_conv(row_batch_graph, r_h, row_nodes_attn_encode)
        row_batch_graph.ndata['h'] = r_h
        # r_h: [batch_nodes_size, g_hidden_dim]
        r_h = r_h[row_m_indices]
        # r_h: [batch_size, g_hidden_dim]

        col_nodes_attn_encode = []
        for i, n in enumerate(col_batch_graph.batch_num_nodes):
            col_nodes_attn_encode.append(batch_mention[i].repeat(n, 1))
        col_nodes_attn_encode = torch.cat(col_nodes_attn_encode)

        # c_h = col_batch_graph.in_degrees().view(-1, 1).float()
        c_h = col_batch_features
        # c_h: [batch_nodes_size, g_in_dim]
        for c_conv in self.col_gcn_layers:
            c_h = c_conv(col_batch_graph, c_h, col_nodes_attn_encode)
        col_batch_graph.ndata['h'] = c_h
        # c_h: [batch_nodes_size, g_hidden_dim]
        c_h = c_h[col_m_indices]
        # c_h: [batch_size, g_hidden_dim]

        cat = torch.cat([r_h, c_h, entity_encode], dim=1)
        res = self.linear(cat)
        # [batch_size, 1]
        return res  # [batch_size, 1]  # 最终匹配得分


if __name__ == '__main__':
    pass





