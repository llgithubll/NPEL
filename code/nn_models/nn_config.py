# -----------------------------------------------
# 深度学习配置
import torch
import os
import os.path
import config


# GPU设置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(1)  # 临时： GPU0已满，使用GPU1
    print(device, torch.cuda.get_device_name(), torch.cuda.current_device())
else:
    print(device)


# 种子设置
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)


# 参数设置
entity_dim = 100  # 实体的向量维度
g_in_dim = 100  # 图卷积输入维度
g_hidden_dim = 256  # 图卷积隐藏层状态维度
g_attention_dim = 128  # 图卷积中注意力向量的维度
l_in_dim = 100  # LSTM输入token的维度
l_hidden_dim = 256  # LSTM中token隐藏层维度
l_num_layers = 2  # LSTM的层数
l_dropout = 0.2  # 当LSTM有多层时，层与层之间的dropout
l_attention_dim = 128  # attention_vector的维度（u的大小）

# 路径
mention_abstract_emb_fp = config.TabEL_mention_abstract_emb_fp
