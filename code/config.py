import os
import os.path

# -----------------------------------------------
candidate_topn = 30

# 保存文件中的分隔符
sep = '---ll+++'  # 应对复杂的字符串形式

# -----------------------------------------------
# Wikipedia数据存放地址
enwiki_dir = '/data/enwiki_dir'

enwiki_with_links_dir = os.path.join(enwiki_dir, 'enwiki_20191020_with_links')
enwiki_title_fp = os.path.join(enwiki_dir, 'enwiki-20191020-all-titles-in-ns0')
enwiki_abstract_dump_fp = os.path.join(enwiki_dir, 'enwiki-20191020-abstract.xml')
enwiki_abstract_fp = os.path.join(enwiki_dir, 'enwiki-20191020-extract-abstract.xml')
enwiki_abstract_dict_fp = os.path.join(enwiki_dir, 'enwiki-20191020-abstract-dict.json')
enwiki_entities_fp = os.path.join(enwiki_dir, 'enwiki-20191020-all-entities.json')
enwiki_in_links_fp = os.path.join(enwiki_dir, 'enwiki-20191020-in_links.json')
enwiki_out_links_fp = os.path.join(enwiki_dir, 'enwiki-20191020-out_links.json')

# -----------------------------------------------
table_data_dir = '/data/table/data'

embedding_fp = os.path.join(enwiki_dir, 'enwiki_20191020_100d.txt')

# 表格数据地址
Limaye_dir = os.path.join(table_data_dir, 'Limaye')
Limaye_mention_abstract_emb_fp = os.path.join(Limaye_dir, 'train_embeddings.pkl')

TabEL_dir = os.path.join(table_data_dir, 'TabEL')  # train_data 在TabEL中
TabEL_mention_abstract_emb_fp = os.path.join(TabEL_dir, 'train_embeddings.pkl')

wiki_links_dir = os.path.join(table_data_dir, 'wiki_links')
wiki_links_mention_abstract_emb_fp = os.path.join(wiki_links_dir, 'train_embeddings.pkl')


# -----------------------------------------------
# m e1 e1_cnt e2 e2_cnt
enwiki_m_e_freq_fp = os.path.join(table_data_dir, 'wiki_m_e_freq_20191020.txt')
# 对mention进行了清洗
enwiki_clean_m_e_freq_fp = os.path.join(table_data_dir, 'wiki_clean_m_e_freq_20191020.txt')

model_dir = os.path.join(table_data_dir, 'model')

# 深度学习模型
# gcn + lstm
base_gcn_lstm_model = os.path.join(model_dir, 'base_gcn_lstm_model')

# e对lstm做attention，结果作为实体编码，对gcn做attention
attention_gcn_lstm_model = os.path.join(model_dir, 'attention_gcn_lstm_model')

# e作为实体编码对gcn做attention
attention_gcn_model = os.path.join(model_dir, 'attention_gcn_model')  # 没有LSTM

# e对lstm做attention，m对gcn做attention
attention_mgcn_elstm_model = os.path.join(model_dir, 'attention_mgcn_elstm_model')


# -----------------------------------------------
# 其余相关文件路径，根据实际情况配置
