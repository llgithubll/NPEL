import numpy as np
import torch
import torch.nn as nn
from functools import reduce
import shutil
import random
import pickle
from tqdm import tqdm
import json
from wiki import *
import config
from utils import *
from collections import Counter
from nn_models.data import tokenizer


class WikiCandsGen:
    def __init__(self, m_e_freq_fp, clean_m, title_fp=None, freq_topn=30):
        if clean_m:
            assert 'clean' in m_e_freq_fp
        else:
            assert 'clean' not in m_e_freq_fp

        self.clean_m = clean_m
        self.freq_topn = freq_topn
        if title_fp:
            self.m2es = load_wiki_m_e_freq_aux(m_e_freq_fp, title_fp, freq_topn)
        else:
            self.m2es = load_wiki_m_e_freq(m_e_freq_fp, freq_topn)

    def _limaye_table_context(self, table, c_idx, r_idx):
        col_context = table[c_idx]
        row_context = [col[r_idx] for col in table]
        return col_context, row_context

    def liamye_converage(self):
        """
        生成候选实体，并统计候选实体中能找到正确链接实体的覆盖率
        limaye数据集一个cell只有一个mention
        :return:
        """
        print_line()
        print('limaye')
        tables_dir = os.path.join(config.Limaye_dir, 'selectedTables_json')
        entities_dir = os.path.join(config.Limaye_dir, 'selectedTablesAnnotationsRelabel_json')
        cand_dir = os.path.join(config.Limaye_dir, 'selectedTablesContext_and_Candidates_json')
        create_dir(cand_dir)

        table_names = os.listdir(tables_dir)
        assert table_names == os.listdir(entities_dir)

        table_mention_cnt = 0  # 在表格中需要链接的mention
        table_mentions = set()
        wiki_mention_cnt = 0  # 在wiki能找到的mention
        wiki_mentions = set()
        hit_entity_cnt = 0
        hit_mentions = set()  # 在wiki中找到并且包含标注候选实体的mention
        candidate_entities = list()
        mention2cand_e = {}

        non_find_mentions = []
        non_find_mention_set = set()

        for i, name in enumerate(table_names):
            t_fp = os.path.join(tables_dir, name)
            e_fp = os.path.join(entities_dir, name)
            c_fp = os.path.join(cand_dir, name)

            with open(t_fp, 'r', encoding='utf-8') as t_f, \
                    open(e_fp, 'r', encoding='utf-8') as e_f, \
                    open(c_fp, 'w', encoding='utf-8') as c_f:
                table = json.load(t_f)['cols']
                entity = json.load(e_f)['entity']
                candidates = []
                assert len(table) == len(entity)
                for c_idx, (col_t, col_e) in enumerate(zip(table, entity)):
                    cand_col = []
                    for r_idx, (m, e) in enumerate(zip(col_t, col_e)):
                        e = e[1]  # relabel entity是个四元组（e, new_e, type, is_relabel)
                        cand_es = []

                        if e:
                            if self.clean_m:
                                m = clean_mention(m)
                            table_mentions.add(m)
                            table_mention_cnt += 1
                            if m in self.m2es:
                                wiki_mentions.add(m)
                                wiki_mention_cnt += 1
                                if e in self.m2es[m]:
                                    cand_es = list(self.m2es[m])
                                    mention2cand_e[m] = cand_es
                                    candidate_entities.extend(cand_es)
                                    hit_mentions.add(m)
                                    hit_entity_cnt += 1
                            else:
                                non_find_mentions.append((name, c_idx, r_idx, m))
                                non_find_mention_set.add(m)
                        # cand_col.append((m, cand_es))
                        col_c, row_c = self._limaye_table_context(table, c_idx, r_idx)
                        cand_col.append({
                            'mention': m,
                            'entity': e,
                            'candidates': cand_es,
                            'col_idx': c_idx,
                            'row_idx': r_idx,
                            'col': col_c,
                            'row': row_c,
                        })
                    candidates.append(cand_col)
                json.dump(candidates, c_f, ensure_ascii=False, indent=4)

        candidate_entities = list(set(candidate_entities))
        print('freq_topn', self.freq_topn)
        print('在表格中需要链接的mention', table_mention_cnt, '去重', len(table_mentions))
        print('候选实体一共有', len(candidate_entities))

        clean_str = 'clean_' if self.clean_m else ''
        with open(os.path.join(config.Limaye_dir, 'wiki_non_find_' + clean_str + 'mention_location.json'), 'w',
                  encoding='utf-8') as dest_f:
            json.dump(non_find_mentions, dest_f, ensure_ascii=False, indent=4)
        with open(os.path.join(config.Limaye_dir, 'wiki_non_find_' + clean_str + 'mention.json'), 'w',
                  encoding='utf-8') as dest_f:
            json.dump(list(non_find_mention_set), dest_f, ensure_ascii=False, indent=4)
        with open(os.path.join(config.Limaye_dir, 'selected_' + clean_str + 'candidate_entities.json'), 'w',
                  encoding='utf-8') as dest_f:
            json.dump(candidate_entities, dest_f, ensure_ascii=False, indent=4)
        with open(os.path.join(config.Limaye_dir, 'selected_' + clean_str + 'mention2candidate_entities.json'), 'w',
                  encoding='utf-8') as dest_f:
            json.dump(mention2cand_e, dest_f, ensure_ascii=False, indent=4)

    def _TabEL_liked_converage(self, main_dir, tables_dir, cand_dir):
        create_dir(cand_dir)

        table_names = os.listdir(tables_dir)
        table_mention_cnt = 0
        table_mentions = set()
        wiki_mention_cnt = 0
        wiki_mentions = set()
        hit_entity_cnt = 0
        hit_mentions = set()
        candidate_entities = []
        mention2cand_e = {}

        non_find_mentions = []
        non_find_mention_set = set()

        for i, name in enumerate(table_names):
            t_fp = os.path.join(tables_dir, name)
            c_fp = os.path.join(cand_dir, name)

            with open(t_fp, 'r', encoding='utf-8') as t_f, \
                    open(c_fp, 'w', encoding='utf-8') as c_f:
                table = json.load(t_f)
                table_data = table['tableData']
                new_table_data = []
                for row in table_data:
                    new_row = []
                    for cell in row:
                        new_surfaceLinks = []
                        for elem in cell['surfaceLinks']:
                            cand_es = []
                            m = elem['surface']
                            e = elem['target']['title']
                            if self.clean_m:
                                m = clean_mention(m)
                            table_mentions.add(m)
                            table_mention_cnt += 1
                            if m in self.m2es:
                                wiki_mentions.add(m)
                                wiki_mention_cnt += 1
                                if e in self.m2es[m]:
                                    cand_es = list(self.m2es[m])
                                    mention2cand_e[m] = cand_es
                                    candidate_entities.extend(cand_es)
                                    hit_mentions.add(m)
                                    hit_entity_cnt += 1
                            else:
                                non_find_mention_set.add(m)
                            elem['candidates'] = cand_es
                            new_surfaceLinks.append(elem)
                        cell['surfaceLinks'] = new_surfaceLinks
                        new_row.append(cell)
                    new_table_data.append(new_row)
                table['tableData'] = new_table_data
                json.dump(table, c_f, ensure_ascii=False, indent=4)

        candidate_entities = list(set(candidate_entities))
        print('freq_topn', self.freq_topn)
        print('候选实体一共有', len(candidate_entities))

        clean_str = 'clean_' if self.clean_m else ''
        with open(os.path.join(main_dir, 'wiki_non_find_' + clean_str + 'mention.json'), 'w',
                  encoding='utf-8') as dest_f:
            json.dump(list(non_find_mention_set), dest_f, ensure_ascii=False, indent=4)
        with open(os.path.join(main_dir, 'selected_' + clean_str + 'candidate_entities.json'), 'w',
                  encoding='utf-8') as dest_f:
            json.dump(candidate_entities, dest_f, ensure_ascii=False, indent=4)
        with open(os.path.join(main_dir, 'selected_' + clean_str + 'mention2candidate_entities.json'), 'w',
                  encoding='utf-8') as dest_f:
            json.dump(mention2cand_e, dest_f, ensure_ascii=False, indent=4)

    def TabEL_converage(self):
        """
        生成候选实体文件，并统计候选实体中能找到正确链接实体的覆盖率
        :return:
        """
        print_line()
        print('TabEL')
        tables_dir = os.path.join(config.TabEL_dir, 'selectedTables_json')
        cand_dir = os.path.join(config.TabEL_dir, 'selectedTables_and_Candidates_json')
        self._TabEL_liked_converage(config.TabEL_dir, tables_dir, cand_dir)

    def wiki_links_converage(self):
        print_line()
        print('wiki_links')
        tables_dir = os.path.join(config.wiki_links_dir, 'selectedTables_json')
        cand_dir = os.path.join(config.wiki_links_dir, 'selectedTables_and_Candidates_json')
        self._TabEL_liked_converage(config.wiki_links_dir, tables_dir, cand_dir)


@timethis
def gen_abstract(entities_redirect_fp, abstract_dict_fp, dest_fp):
    """
    主要是为候选实体抽取出对应的abstract
    :param entities_redirect_fp:  经过处理之后的实体文件，实体名称会更新
    :param abstract_dict_fp: 包含所有实体摘要的字典文件
    :param dest_fp:
    :return:
    """
    with open(entities_redirect_fp, 'r', encoding='utf-8') as f:
        entities_abstract = json.load(f)
        print('entities', len(entities_abstract))

    with open(abstract_dict_fp, 'r', encoding='utf-8') as f:
        abstract = json.load(f)
        print('abstract', len(abstract))

    for e in entities_abstract:
        if e in abstract:
            entities_abstract[e]['abstract'] = abstract[e]
        elif entities_abstract[e]['title'] in abstract:
            entities_abstract[e]['abstract'] = abstract[entities_abstract[e]['title']]
        else:
            entities_abstract[e]['abstract'] = ''

    has_abstract = sum([bool(_['abstract']) for _ in entities_abstract.values()])
    print('unkonw entity', sum([bool(_['status'] == 'unknown') for _ in entities_abstract.values()]))
    print('has_abstract/total ' + str(has_abstract) + '/' + str(len(entities_abstract)) + '=' + str(
        has_abstract / len(entities_abstract)))
    with open(dest_fp, 'w', encoding='utf-8') as dest_f:
        json.dump(entities_abstract, dest_f, ensure_ascii=False, indent=4)


@timethis
def gen_embedding(emb_dim, entities_redirect_fp, emb_fp, dest_fp):
    """
    为候选实体抽取embedding
    :param emb_dim: 词向量维度
    :param entities_redirect_fp: 经过处理后的实体文件，实体名称会更新
    :param emb_fp: 包含所有实体和词汇的预训练词向量
    :param dest_fp:
    :return:
    """
    entitiy_embedding = {}
    with open(entities_redirect_fp, 'r', encoding='utf-8') as f:
        entities = json.load(f)
        for e, item in entities.items():
            if e:
                entitiy_embedding[e] = []
                entitiy_embedding[e.lower()] = []
                if item['title']:
                    entitiy_embedding[item['title']] = []
                    entitiy_embedding[item['title'].lower()] = []

        print('entities', len(entities))
        print('entity embedding(+title)', len(entitiy_embedding))

    with open(emb_fp, 'r', encoding='utf-8') as emb_f:
        line_cnt = 0
        for line in tqdm(emb_f):
            values = line.strip().split()
            if len(values) <= 2:  # 表示数目，
                print(line.strip())
                continue
            line_cnt += 1
            e = ' '.join(values[:-emb_dim])
            if e.startswith('ENTITY/'):  # 看成实体，实体不变大小写
                e = e[7:].replace(' ', '_')
                if e in entitiy_embedding:
                    entitiy_embedding[e] = [float(_) for _ in values[-emb_dim:]]
            else:  # 看成词
                if e in entitiy_embedding:
                    entitiy_embedding[e] = [float(_) for _ in values[-emb_dim:]]
                elif e.lower() in entitiy_embedding:
                    entitiy_embedding[e.lower()] = [float(_) for _ in values[-emb_dim:]]
        print(line_cnt, emb_dim)

    for e in entities:
        if e:
            if entitiy_embedding[e]:
                entities[e]['embedding'] = entitiy_embedding[e]
            elif entitiy_embedding[e.lower()]:
                entities[e]['embedding'] = entitiy_embedding[e.lower()]
            elif entities[e]['title'] and entitiy_embedding[entities[e]['title']]:
                entities[e]['embedding'] = entitiy_embedding[entities[e]['title']]
            elif entities[e]['title'] and entities[e]['title'].lower() and entitiy_embedding[
                entities[e]['title'].lower()]:
                entities[e]['embedding'] = entitiy_embedding[entities[e]['title'].lower()]
            else:
                entities[e]['embedding'] = []

    has_emb = sum([bool(_['embedding']) for _ in entities.values()])
    print('unkonw entity', sum([bool(_['status'] == 'unknown') for _ in entities.values()]))
    print('has_embedding/total ' + str(has_emb) + '/' + str(len(entities)) + '=' + str(has_emb / len(entities)))
    with open(dest_fp, 'w', encoding='utf-8') as dest_f:
        json.dump(entities, dest_f, ensure_ascii=False, indent=4)
    print()


class TabEL_Liked_TrainingDataGen:
    def __init__(self, table_dir, cand_abstract_dict_fp, cand_embedding_dict_fp, mention2e_fp, neg_cnt, dest_fp, clean):
        """
        :param table_dir: 整理出来的训练数据表格文件夹
        :param cand_abstract_dict_fp: 从以上表格中抽的候选实体，及摘要
        :param cand_embedding_dict_fp: 词向量的文件
        :param mention2e_fp: 整理的表格中的mention对应的候选实体
        :param neg_cnt: 负例个数
        :param dest_fp: 目标文件
        """
        table_fps = []
        for name in os.listdir(table_dir):
            table_fps.append(os.path.join(table_dir, name))

        with open(cand_abstract_dict_fp, 'r', encoding='utf-8') as f:
            abstract = json.load(f)
            candidate_entities = list(abstract.keys())
        with open(cand_embedding_dict_fp, 'r', encoding='utf-8') as f:
            embedding = json.load(f)
        with open(mention2e_fp, 'r', encoding='utf-8') as f:
            self.m2es = json.load(f)

        row_context_len = []
        col_context_len = []
        cell_context_len = []
        res = {}
        for t_fp in tqdm(table_fps):
            name = os.path.basename(t_fp)
            with open(t_fp, 'r', encoding='utf-8') as f:
                table = json.load(f)

            table_mentions = []
            for row in table['tableData']:
                row_mentions = []
                for cell in row:
                    cell_mentions = []
                    for e in cell['surfaceLinks']:
                        if e['surface']:
                            cell_mentions.append(e['surface'])
                    row_mentions.append(cell_mentions)
                table_mentions.append(row_mentions)

            for r_idx, row in enumerate(table['tableData']):
                for c_idx, cell in enumerate(row):
                    for e in cell['surfaceLinks']:
                        m = e['surface']
                        if clean:
                            m = clean_mention(m)
                        if m not in self.m2es:
                            continue
                        pos_e = e['target']['title']
                        # 实体要有摘要，要有embedding
                        if pos_e not in abstract or abstract[pos_e]['abstract'] == '' or \
                                pos_e not in embedding or embedding[pos_e]['embedding'] == []:
                            continue
                        cand_e = [_ for _ in self.m2es[m]
                                  if _ and _ != pos_e and abstract[_]['abstract'] and embedding[_]['embedding']]
                        if len(cand_e) >= neg_cnt:  # 从候选实体中挑选负例
                            random.shuffle(cand_e)
                            neg_es = cand_e[:neg_cnt]
                        else:
                            cand_e = set(cand_e)
                            while len(cand_e) < neg_cnt:
                                _ = random.choice(candidate_entities)
                                if _ and _ != pos_e and \
                                        abstract[_]['title'] != pos_e and \
                                        abstract[_]['abstract'] and \
                                        embedding[_]['embedding']:
                                    cand_e.add(_)
                            neg_es = list(cand_e)
                        row_context, col_context, cell_context = self.mention_context(table_mentions, m, r_idx, c_idx)

                        if col_context:
                            if not row_context:
                                row_context = {m}

                            res[str((name, r_idx, c_idx, m))] = {  # load后，需要eval(key)，得到原始key
                                'row_context': list(row_context),
                                'col_context': list(col_context),
                                'cell_context': list(cell_context),
                                'pos_entity': pos_e,
                                'neg_entities': neg_es
                            }
                            row_context_len.append(len(row_context))
                            col_context_len.append(len(col_context))
                            cell_context_len.append(len(cell_context))

        row_context_len = list(Counter(row_context_len).items())
        row_context_len.sort()
        col_context_len = list(Counter(col_context_len).items())
        col_context_len.sort()
        cell_context_len = list(Counter(cell_context_len).items())
        cell_context_len.sort()

        print_line('row_context_len')
        pprint(row_context_len)

        print_line('col_context_len')
        pprint(col_context_len)

        print_line('cell_context_len')
        pprint(cell_context_len)

        print('train data mentions', len(res))
        with open(dest_fp, 'w', encoding='utf-8') as dest_f:
            json.dump(res, dest_f, ensure_ascii=False, indent=4)

    def mention_context(self, mentions_matrix, curr_mention, r_idx, c_idx):
        row_c = set()
        col_c = set()
        cell_c = set()  # 同一个cell中的其他mention

        for c_i, cell in enumerate(mentions_matrix[r_idx]):
            for m in cell:
                if m and m != curr_mention and m in self.m2es:
                    if c_i == c_idx:
                        cell_c.add(m)
                    else:
                        row_c.add(m)
        for r_i, row in enumerate(mentions_matrix):
            for m in row[c_idx]:
                if m and m != curr_mention and m in self.m2es:
                    if r_i == r_idx:
                        cell_c.add(m)
                    else:
                        col_c.add(m)
        return row_c, col_c, cell_c


class Limaye_TrainingDataGen:
    """
    类似于TabEL_Liked_TrainingDataGen
    构造出模型可以接受的数据格式
    Limaye和TabEL_liked数据格式不同
    """

    def __init__(self, table_dir, entity_dir,
                 cand_abstract_dict_fp, cand_embedding_dict_fp, mention2e_fp, dest_fp, clean):
        """
        :param table_dir: 整理出来的训练数据表格文件夹
        :param entity_dir: 对应的链接实体在单独的一个文件夹
        :param cand_abstract_dict_fp: 从以上表格中抽的候选实体，及摘要
        :param cand_embedding_dict_fp: 词向量的文件
        :param mention2e_fp: 整理的表格中的mention对应的候选实体
        :param dest_fp: 目标文件
        """
        table_names = os.listdir(table_dir)
        assert table_names == os.listdir(entity_dir)

        with open(cand_abstract_dict_fp, 'r', encoding='utf-8') as f:
            abstract = json.load(f)
        with open(cand_embedding_dict_fp, 'r', encoding='utf-8') as f:
            embedding = json.load(f)
        with open(mention2e_fp, 'r', encoding='utf-8') as f:
            self.m2es = json.load(f)

        row_context_len = []
        col_context_len = []
        cell_context_len = []
        res = {}
        for name in tqdm(table_names):
            t_fp = os.path.join(table_dir, name)
            e_fp = os.path.join(entity_dir, name)
            with open(t_fp, 'r', encoding='utf-8') as t_f, \
                    open(e_fp, 'r', encoding='utf-8') as e_f:
                table = json.load(t_f)['cols']
                entity = json.load(e_f)['entity']

            for c_idx, col in enumerate(entity):
                for r_idx, cell in enumerate(col):
                    m = table[c_idx][r_idx]
                    if clean:
                        m = clean_mention(m)
                    if m not in self.m2es:
                        continue
                    e = cell[1]
                    if not e:
                        continue
                    if e not in abstract or abstract[e]['abstract'] == '' or \
                            e not in embedding or embedding[e]['embedding'] == []:
                        continue
                    row_context, col_context, cell_context = self.mention_context(table, m, r_idx, c_idx)
                    if col_context:
                        if not row_context:
                            row_context = {m}

                        res[str((name, r_idx, c_idx, m))] = {  # load后，需要eval(key)，得到原始key
                            'row_context': list(row_context),
                            'col_context': list(col_context),
                            'cell_context': list(cell_context),
                            'pos_entity': e,
                            'neg_entities': []
                        }
                        row_context_len.append(len(row_context))
                        col_context_len.append(len(col_context))
                        cell_context_len.append(len(cell_context))

        row_context_len = list(Counter(row_context_len).items())
        row_context_len.sort()
        col_context_len = list(Counter(col_context_len).items())
        col_context_len.sort()
        cell_context_len = list(Counter(cell_context_len).items())
        cell_context_len.sort()

        print_line('row_context_len')
        pprint(row_context_len)

        print_line('col_context_len')
        pprint(col_context_len)

        print_line('cell_context_len')
        pprint(cell_context_len)

        print('train data mentions', len(res))
        with open(dest_fp, 'w', encoding='utf-8') as dest_f:
            json.dump(res, dest_f, ensure_ascii=False, indent=4)

    def mention_context(self, table, curr_mention, r_idx, c_idx):
        row_c = set()
        col_c = set()
        cell_c = set()

        for m in table[c_idx]:
            if m != curr_mention and m in self.m2es:
                col_c.add(m)
        for col in table:
            m = col[r_idx]
            if m != curr_mention and m in self.m2es:
                row_c.add(m)
        return row_c, col_c, cell_c


def mention_entity_pair(train_mention_fp, abstract_fp, embedding_fp, dest_fp):
    """
    实际上只是对训练数据的展开
    :param train_mention_fp:一个mention对应的上下文，正确entity，错误entities
    :param abstract_fp:所有候选实体的摘要
    :param embedding_fp:所有候选实体的embedding_fp
    :return:
    """
    with open(train_mention_fp, 'r', encoding='utf-8') as f:
        _ = json.load(f)
        mentions = {}
        for k, v in _.items():
            mentions[eval(k)] = v
    with open(abstract_fp, 'r', encoding='utf-8') as f:
        abstract = json.load(f)
    with open(embedding_fp, 'r', encoding='utf-8') as f:
        embedding = json.load(f)

    res = []
    for k, v in tqdm(mentions.items()):
        res.append({
            'name': k[0],
            'r_idx': k[1],
            'c_idx': k[2],
            'mention': k[3],
            'entity': v['pos_entity'],
            'cell_context': v['cell_context'],
            'col_context': v['col_context'],
            'row_context': v['row_context'],
            'abstract': abstract[v['pos_entity']]['abstract'],
            'embedding': embedding[v['pos_entity']]['embedding'],  # 实体embedding
            'label': 1
        })
        for n_e in v['neg_entities']:
            res.append({
                'name': k[0],
                'r_idx': k[1],
                'c_idx': k[2],
                'mention': k[3],
                'entity': n_e,
                'cell_context': v['cell_context'],
                'col_context': v['col_context'],
                'row_context': v['row_context'],
                'abstract': abstract[n_e]['abstract'],
                'embedding': embedding[n_e]['embedding'],  # 实体embedding
                'label': 0
            })

    print('total', len(res), '(m,e) pair')
    names = [_['name'].split('.')[0] for _ in res]
    if all([name.isdigit() for name in names]):
        res.sort(key=lambda x: int(x['name'].split('.')[0]))
    with open(dest_fp, 'w', encoding='utf-8') as dest_f:
        json.dump(res, dest_f, ensure_ascii=False, indent=4)


def dump_mention_abstract_emb(emb_fp, train_fp, dest_fp, emb_dim):
    """
    对mention和abstract进行编码

    加载词向量文件，词向量文件格式如下（对应到本环境中，word其实是char) 首行是统计信息，其余是词向量
    word_count emb_dim
    word1 0.1 0.2 0.3 ...
    word2 0.1 0.2 0.3 ...
    ...
    :param emb_fp: 文件路径，10G大小, 太大了，只取数据集上出现了的实体和词。
    :param train_fp: 训练数据文件路径
    :param dest_fp: 保存结果，直接pickle.dump
    :param emb_dim: 词向量维度
    :return: word_to_idx, embeddings: [[0.1 0.2 0.3 ...], [0.1, 0.2, 0.3 ...], ...]
    """
    with open(train_fp, 'r', encoding='utf-8') as f:
        train = json.load(f)
    print('train', len(train))

    mention_encode = {}  # mention的编码
    abstract_encode = {}  # abstract中的词和实体的编码
    for i, item in tqdm(enumerate(train)):
        mention_encode[item['mention']] = None
        for _ in item['col_context']:
            mention_encode[_] = None
        for _ in item['row_context']:
            mention_encode[_] = None
        for _ in item['cell_context']:
            mention_encode[_] = None

        for t in tokenizer(item['abstract']):
            abstract_encode[t] = None

    print('mention_encode', len(mention_encode))
    print('abstract tokens', len(abstract_encode))

    word_to_idx = {}
    embeddings = []

    entity_cnt = 0
    with open(emb_fp, 'r', encoding='utf-8') as f:
        cnt = 0
        for idx, line in tqdm(enumerate(f)):
            line = line.strip().split()
            if len(line) <= 2:  # 不是embedding，是数量
                print(line)
                continue
            cnt += 1
            word = ' '.join(line[:-emb_dim])
            word = word.replace(' ', '_')

            if word.startswith('ENTITY/'):
                entity_cnt += 1
            embed = np.array([float(_) for _ in line[-emb_dim:]])

            word_to_idx[word] = idx
            embeddings.append(embed)
        print('corpus embedding: (word_count ', cnt, '), (dimention ', emb_dim, ')')
    print('total', len(word_to_idx), 'entity', entity_cnt, 'dump...')

    for k in mention_encode:
        e_k = 'ENTITY/' + k.replace(' ', '_')
        if e_k in word_to_idx:
            mention_encode[k] = embeddings[word_to_idx[e_k]]
        elif k.lower() in word_to_idx:
            mention_encode[k] = embeddings[word_to_idx[k.lower()]]
        else:
            tokens = ''.join((char if char.isalpha() or char.isdigit() else " ") for char in k).split()
            token_embs = []
            for t in tokens:
                e_t = 'ENTITY/' + t
                if e_t in word_to_idx:
                    token_embs.append(embeddings[word_to_idx[e_t]])
                elif t.lower() in word_to_idx:
                    token_embs.append(embeddings[word_to_idx[t.lower()]])
            if len(token_embs) == 1:
                mention_encode[k] = token_embs[0]
            elif len(token_embs) > 1:
                token_embs = np.array(token_embs)
                reduce_emb = reduce(lambda x, y: x + y, token_embs) / len(token_embs)
                mention_encode[k] = reduce_emb

    for t in abstract_encode:
        if t in word_to_idx:
            abstract_encode[t] = embeddings[word_to_idx[t]]

    print('mention_encode(without pad, unk)', len(mention_encode))
    print('abstract tokens(without pad, unk)', len(abstract_encode))

    mention_empty_cnt = 0
    for k, v in mention_encode.items():
        if v is None:
            mention_empty_cnt += 1
    abstract_empty_cnt = 0
    for k, v in abstract_encode.items():
        if v is None:
            abstract_empty_cnt += 1

    print('empty mention rate(without pad, unk)', mention_empty_cnt, '/', len(mention_encode),
          mention_empty_cnt / len(mention_encode))
    print('empty abstract rate(without pad, unk)', abstract_empty_cnt, '/', len(abstract_encode),
          abstract_empty_cnt / len(abstract_encode))

    pad_emb = torch.zeros((1, emb_dim))
    nn.init.xavier_normal_(pad_emb)
    pad_emb = np.array(pad_emb.tolist()[0])
    unk_emb = torch.zeros((1, emb_dim))
    nn.init.xavier_normal_(unk_emb)
    unk_emb = np.array(unk_emb.tolist()[0])

    mention_encode['<unk>'], mention_encode['<pad>'] = unk_emb, pad_emb
    abstract_encode['<unk>'], abstract_encode['<pad>'] = unk_emb, pad_emb

    with open(dest_fp, 'wb') as dest_f:
        pickle.dump((mention_encode, abstract_encode), dest_f)


def rebuild_TabEL_liked_tables(table_names: set, train_data_m_e_fp, m2e_fp, candidate_abstract_fp,
                               candidate_embedding_fp,
                               dest_dir,
                               m_cnt_lower, m_cnt_upper):
    """
    重建表格，包含表格原始的状态
    每个mention和对应的候选实体
    :param table_names, 用来构建数据集的表格的文件名集合
    :param train_data_m_e_fp, 从m, e 数据对中重建表格，这里面的数据，已经经过处理了（处理方式data_generate.py/TabEL_Liked_TrainingDataGen)
    :param m2e_fp, 先验统计，用来生成候选实体
    :param candidate_abstract_fp, 数据集覆盖的所有实体的摘要
    :param candidate_embedding_fp, 数据集覆盖的所有实体的embediding
    :param dest_dir, 将重构表格写入对应文件路径
    :param m_cnt_lower, mention数量下限
    :param m_cnt_upper, mention数量上限
    """
    if os.path.exists(dest_dir):
        print('clear', dest_dir)
        shutil.rmtree(dest_dir)
    create_dir(dest_dir)

    tables_mention = defaultdict(list)
    with open(train_data_m_e_fp, 'r', encoding='utf-8') as f:
        train_data_m_e = json.load(f)
    with open(m2e_fp, 'r', encoding='utf-8') as f:
        m2es = json.load(f)
    with open(candidate_abstract_fp, 'r', encoding='utf-8') as f:
        abstract = json.load(f)
    with open(candidate_embedding_fp, 'r', encoding='utf-8') as f:
        embedding = json.load(f)

    mention_cnt = 0
    hit_cnt = 0
    for item in train_data_m_e:
        if item['name'] in table_names and item['label'] == 1:
            mention_cnt += 1
            item['candidates_abstract'] = {}
            item['candidates_embedding'] = {}
            for e in m2es[item['mention']]:
                if e and abstract[e]['abstract'] and embedding[e]['embedding']:
                    item['candidates_abstract'][e] = abstract[e]['abstract']
                    item['candidates_embedding'][e] = embedding[e]['embedding']
            if item['entity'] in item['candidates_abstract']:
                hit_cnt += 1
                tables_mention[item['name']].append(item)

    print('mention cnt: [%d,%d]' % (m_cnt_lower, m_cnt_upper))
    print('hit rate', hit_cnt, '/', mention_cnt, '=', hit_cnt / mention_cnt)
    print('tables', len(tables_mention))
    print('max mention cnt:', max([len(v) for v in tables_mention.values()]))
    save_cnt = 0
    for name in tables_mention:
        if m_cnt_lower <= len(tables_mention[name]) <= m_cnt_upper:
            save_cnt += 1
            with open(os.path.join(dest_dir, name), 'w', encoding='utf-8') as dest_f:
                json.dump(tables_mention[name], dest_f, ensure_ascii=False, indent=4)
    print('save tables', save_cnt)
    print()


if __name__ == '__main__':
    pass