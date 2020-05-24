import shutil
import numpy as np
import time
import config
import os
import os.path
import json
import copy

from multiprocessing import Pool

from scipy import spatial
from strsimpy.jaro_winkler import JaroWinkler
from utils import timethis, print_line, create_dir, sigmoid, check_result

from collections import defaultdict
from queue import PriorityQueue

from nn_models.predict import Predictor, AttentionPredictor
from wiki import load_prior


class JointDisambiguation:
    def __init__(self, table_fp, dest_fp, m_e_pred, part_prior,
                 alpha=0.5, beta=0.5, verbose=1):
        """
        :param table_fp:
        :param dest_fp:
        :param m_e_pred: 深度学习模型预测概率
        :param part_prior: 包含表格内部所有需要的先验的概率
        :param alpha: m,e中计算先验的权重
        :param beta: 方差影响的权重
        :param verbose:
        """
        self.prior = part_prior
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose
        self.m_e_pred = m_e_pred

        # cache
        self.col_entities = {}
        self.distance_cache = {}  # 保存一些计算过的距离
        self.col_item = defaultdict(list)  # 列索引，记录每列的mention（只记录在table中下标）
        self.row_item = defaultdict(list)  # 行索引
        self.cell_item = defaultdict(set)  # 单元格索引，一个单元格可能有多个mention
        self.mention_pair_conf = {}  # mention_pair conf

        with open(table_fp, 'r', encoding='utf-8') as f:
            table = json.load(f)
        print(os.path.basename(table_fp), 'mentions', len(table))

        for i in range(len(table)):
            r, c = table[i]['r_idx'], table[i]['c_idx']
            self.row_item[r].append(i)
            self.col_item[c].append(i)
            self.cell_item[(r, c)].add(i)

        for item in table:
            self.col_entities[item['c_idx']] = {}
        self.subject_c_idx = subject_col_select(table)

        if len(table) == 1:  # 不需要联合消歧
            item = table[0]
            disam_e = None
            max_v = -1
            for c_e, c_e_a in item['candidates_abstract'].items():
                v = self._m_e_sim(item['r_idx'], item['c_idx'], item['mention'], c_e)
                if v > max_v:
                    max_v = v
                    disam_e = c_e
            table[0]['disambiguation'] = disam_e
            with open(dest_fp, 'w', encoding='utf-8') as dest_f:
                json.dump(table, dest_f, ensure_ascii=False, indent=4)
        else:  # 联合消歧
            table_disam = self.pair_linking_collective_disambiguate(table)
            with open(dest_fp, 'w', encoding='utf-8') as dest_f:
                json.dump(table_disam, dest_f, ensure_ascii=False, indent=4)

    def pair_linking_collective_disambiguate(self, table):
        """
        对table进行pair-linking
        :param table: 一个list，每个元素是需要链接的mention
        :return:
        """
        mention_cnt = len(table)
        for item in table:
            item['disambiguation'] = ''
            item['candidates'] = set(item['candidates_abstract'].keys())

        linked = [False] * mention_cnt
        iter_cnt = 0

        while True:
            t0 = time.time()
            if all(linked):  # 全部消歧
                break

            mention_pairs_confidence = []
            for i in range(mention_cnt):
                if linked[i]:
                    continue
                for c_i in self.row_item[table[i]['r_idx']]:  # 当前行的所有元素在table中下标
                    if c_i == i:
                        continue
                    key = self._mention_pair_key(table[i], table[c_i])
                    key1 = self._mention_pair_key(table[c_i], table[i])
                    if key in self.mention_pair_conf:
                        conf, e0, e1 = self.mention_pair_conf[key]
                    elif key1 in self.mention_pair_conf:
                        conf, e1, e0 = self.mention_pair_conf[key1]
                    else:
                        conf, e0, e1 = self.confidence_between_mention(table[i], table[c_i])
                        self.mention_pair_conf[key] = (conf, e0, e1)
                    mention_pairs_confidence.append((conf, i, e0, c_i, e1))

                for r_i in self.col_item[table[i]['c_idx']]:  # 当前列的所有元素在table中下标
                    if r_i == i:
                        continue
                    key = self._mention_pair_key(table[i], table[r_i])
                    key1 = self._mention_pair_key(table[r_i], table[i])
                    if key in self.mention_pair_conf:
                        conf, e0, e1 = self.mention_pair_conf[key]
                    elif key1 in self.mention_pair_conf:
                        conf, e1, e0 = self.mention_pair_conf[key1]
                    else:
                        conf, e0, e1 = self.confidence_between_mention(table[i], table[r_i])
                        self.mention_pair_conf[key] = (conf, e0, e1)
                    # if e0 is None:  # table[r_i]所在cell有多个mention
                    #     continue
                    mention_pairs_confidence.append((conf, i, e0, r_i, e1))

            mention_pairs_confidence.sort(reverse=True)
            conf, i, e0, j, e1 = mention_pairs_confidence[0]

            if not linked[i]:
                linked[i] = True
                table[i]['disambiguation'] = e0
                table[i]['candidates'] = {e0}
                self.col_entities[table[i]['c_idx']][e0] = (table[i]['r_idx'],  # 行，下标
                                                            table[i]['candidates_embedding'][e0])  # 实体emb
                for c_i in self.row_item[table[i]['r_idx']]:  # 当前行的所有元素在table中下标
                    if c_i == i or linked[c_i]:
                        continue
                    key = self._mention_pair_key(table[i], table[c_i])
                    self.mention_pair_conf[key] = self.confidence_between_mention(table[i], table[c_i])
                for r_i in self.col_item[table[i]['c_idx']]:  # 当前列所有元素在table中下标
                    if r_i == i or linked[r_i]:
                        continue
                    key = self._mention_pair_key(table[i], table[r_i])
                    self.mention_pair_conf[key] = self.confidence_between_mention(table[i], table[r_i])

            if not linked[j]:
                linked[j] = True
                table[j]['disambiguation'] = e1
                table[j]['candidates'] = {e1}
                self.col_entities[table[j]['c_idx']][e1] = (table[j]['r_idx'],
                                                            table[j]['candidates_embedding'][e1])
                for c_j in self.row_item[table[j]['r_idx']]:  # 当前行的所有元素在table中下标
                    if c_j == j or linked[c_j]:
                        continue
                    key = self._mention_pair_key(table[j], table[c_j])
                    self.mention_pair_conf[key] = self.confidence_between_mention(table[j], table[c_j])
                for r_j in self.col_item[table[j]['c_idx']]:  # 当前列所有元素在table中下标
                    if r_j == j or linked[r_j]:
                        continue
                    key = self._mention_pair_key(table[j], table[r_j])
                    self.mention_pair_conf[key] = self.confidence_between_mention(table[j], table[r_j])

            iter_cnt += 1
            if self.verbose >= 2:
                print('\titer', iter_cnt, 'done', '%d/%d' % (sum(linked), len(linked)),
                      'time %.3fs' % (time.time() - t0))

        for item in table:
            item.pop('candidates')  # candidates完成使命，这里已不需要
        return table  # 返回完成链接了的mention

    def _mention_pair_key(self, item0, item1):
        r0, c0 = item0['r_idx'], item0['c_idx']
        r1, c1 = item1['r_idx'], item1['c_idx']
        key = (r0, c0, item0['mention'], r1, c1, item1['mention'])
        return key

    def confidence_between_mention(self, item0, item1):
        r0, c0 = item0['r_idx'], item0['c_idx']
        r1, c1 = item1['r_idx'], item1['c_idx']
        max_conf, max_c1, max_c2 = float('-inf'), None, None
        if (r0, c0) == (r1, c1):  # 同cell的元素，只计算距离
            for c1 in item0['candidates']:
                for c2 in item1['candidates']:
                    distance = self._d_div_3(item0, item1, c1, c2)
                    sim = 1 - distance
                    conf = sim
                    if conf > max_conf:
                        max_conf, max_c1, max_c2 = conf, c1, c2
            return max_conf, max_c1, max_c2
        else:
            line_in_col_sub = set([_[0] for _ in self.col_entities[self.subject_c_idx].values()])
            emb_in_col_sub = {}
            for _ in self.col_entities[self.subject_c_idx].values():
                emb_in_col_sub[_[0]] = _[1]  # 每一行只留下一个emb留作计算
            c_idx1, c_idx2 = item0['c_idx'], item1['c_idx']
            line_in_col1 = set([_[0] for _ in self.col_entities[c_idx1].values()])
            line_in_col2 = set([_[0] for _ in self.col_entities[c_idx2].values()])
            row_set1 = line_in_col1 & line_in_col_sub
            row_set2 = line_in_col2 & line_in_col_sub

            max_conf, max_c1, max_c2 = float('-inf'), None, None
            for c1 in item0['candidates']:
                for c2 in item1['candidates']:
                    distance = self._d_div_3(item0, item1, c1, c2)
                    sim = 1 - distance
                    coherence_diff = self._delta_var(item0, item1, c1, c2, row_set1, row_set2, emb_in_col_sub)
                    conf = sim + self.beta * coherence_diff
                    if conf > max_conf:
                        max_conf, max_c1, max_c2 = conf, c1, c2
            return max_conf, max_c1, max_c2

    def _d_div_3(self, m1, m2, c1, c2):
        key = (m1['r_idx'], m1['c_idx'], m1['mention'], c1,
               m2['r_idx'], m2['c_idx'], m2['mention'], c2)
        if key in self.distance_cache:
            return self.distance_cache[key]

        d = 1 - (self._m_e_sim(m1['r_idx'], m1['c_idx'], m1['mention'], c1) +
                 self._m_e_sim(m2['r_idx'], m2['c_idx'], m2['mention'], c2) +
                 self._e_e_sim(m1['candidates_embedding'][c1], m2['candidates_embedding'][c2])) / 3
        self.distance_cache[key] = d
        return d

    def _m_e_sim(self, r, c, m, e):
        pred = self.m_e_pred[(r, c, m, e)]
        proi = self.prior[m][e]
        res = (1 - self.alpha) * pred + self.alpha * proi
        return res

    def _e_e_sim(self, e1_emb, e2_emb):
        """
        两个实体embedding的余弦相似度
        :param e1_emb:
        :param e2_emb:
        :return:
        """
        return 1. - spatial.distance.cosine(e1_emb, e2_emb)

    def _delta_var(self, item0, item1, e0, e1, row_set0, row_set1, emb_in_col_sub):
        """
        一对mention: item0, item1及对应候选实体e0, e1
        :return:
        """
        col0_emb = [_[1] for _ in self.col_entities[item0['c_idx']].values()]
        col1_emb = [_[1] for _ in self.col_entities[item1['c_idx']].values()]
        e0_emb = item0['candidates_embedding'][e0]
        e1_emb = item1['candidates_embedding'][e1]

        norm_delta_col0, norm_delta_col1, norm_delta_rel0, norm_delta_rel1 = 0, 0, 0, 0
        if not item0['disambiguation'] and not item1['disambiguation']:
            if item0['c_idx'] == item1['c_idx']:  # 同列
                norm_delta_col0 = self._norm_var(self._delta_col_var(col0_emb + [e1_emb], e0_emb))
                norm_delta_col1 = self._norm_var(self._delta_col_var(col1_emb + [e0_emb], e1_emb))
                if item1['r_idx'] in emb_in_col_sub:
                    row_set0.add(item1['r_idx'])
                    norm_delta_rel0 = self._norm_var(self._delta_rel_var(item0['c_idx'], item0['r_idx'], row_set0, emb_in_col_sub, e0_emb,
                                                                         temp_r_idx=item1['r_idx'], temp_emb=e1_emb))
                else:
                    norm_delta_rel0 = self._norm_var(self._delta_rel_var(item0['c_idx'], item0['r_idx'], row_set0, emb_in_col_sub, e0_emb))
                if item0['r_idx'] in emb_in_col_sub:
                    row_set1.add(item0['r_idx'])
                    norm_delta_rel1 = self._norm_var(self._delta_rel_var(item1['c_idx'], item1['r_idx'], row_set1, emb_in_col_sub, e1_emb,
                                                                         temp_r_idx=item0['r_idx'], temp_emb=e0_emb))
                else:
                    norm_delta_rel1 = self._norm_var(self._delta_rel_var(item1['c_idx'], item1['r_idx'], row_set1, emb_in_col_sub, e1_emb))
            else:  # 同行 or 特殊情况处理（处理方式并没有发生变化）
                norm_delta_col0 = self._norm_var(self._delta_col_var(col0_emb, e0_emb))
                norm_delta_col1 = self._norm_var(self._delta_col_var(col1_emb, e1_emb))
                norm_delta_rel0 = self._norm_var(self._delta_rel_var(item0['c_idx'], item0['r_idx'], row_set0, emb_in_col_sub, e0_emb))
                norm_delta_rel1 = self._norm_var(self._delta_rel_var(item1['c_idx'], item1['r_idx'], row_set1, emb_in_col_sub, e1_emb))
        elif not item0['disambiguation']:
            norm_delta_col0 = self._norm_var(self._delta_col_var(col0_emb, e0_emb))
            norm_delta_rel0 = self._norm_var(self._delta_rel_var(item0['c_idx'], item0['r_idx'], row_set0, emb_in_col_sub, e0_emb))
        elif not item1['disambiguation']:
            norm_delta_col1 = self._norm_var(self._delta_col_var(col1_emb, e1_emb))
            norm_delta_rel1 = self._norm_var(self._delta_rel_var(item1['c_idx'], item1['r_idx'], row_set1, emb_in_col_sub, e1_emb))
        else:
            raise ValueError('不可能到达的情况，两个mention至少有一个没有进行链接')

        var_diff = 0
        if not item0['disambiguation'] and not item1['disambiguation']:
            var_diff = (norm_delta_col0 + norm_delta_col1 + norm_delta_rel0 + norm_delta_rel1) / 1
        elif not item0['disambiguation']:
            var_diff = (norm_delta_col0 + norm_delta_rel0) / 1
        elif not item1['disambiguation']:
            var_diff = (norm_delta_col1 + norm_delta_rel1) / 1
        return var_diff

    def _delta_col_var(self, col_emb, e_emb):
        """
        :param col_emb:
        :param e_emb:
        :return:
        """
        if not col_emb:  # 如果列中为空，放入新实体方差增量为0
            return 0.0
        var_before = np.sum(np.var(col_emb, axis=0)) / len(e_emb)
        var_after = np.sum(np.var(col_emb + [e_emb], axis=0)) / len(e_emb)

        coherent_before = 10000 if self._close_zero(var_before) else 1 / var_before
        coherent_after = 10000 if self._close_zero(var_after) else 1 / var_after
        return coherent_after - coherent_before

    def _close_zero(self, num):
        return -0.00005 <= num <= 0.00005

    def _delta_rel_var(self, c_idx, r_idx, row_set, emb_in_col_sub, e_emb, temp_r_idx=None, temp_emb=None):
        if c_idx == self.subject_c_idx or not row_set or r_idx not in emb_in_col_sub:
            return 0.0
        emb_in_col = {}
        for _ in self.col_entities[c_idx].values():
            emb_in_col[_[0]] = _[1]
        if temp_r_idx is not None and temp_emb is not None:  # 同时链接两个mention可能，假设另一个已经是已链接的
            emb_in_col[temp_r_idx] = temp_emb
        rel_emb = []
        for r in row_set:
            rel_emb.append((np.array(emb_in_col_sub[r]) - np.array(emb_in_col[r])).tolist())
        r_emb = (np.array(emb_in_col_sub[r_idx]) - np.array(e_emb)).tolist()
        return self._delta_col_var(rel_emb, r_emb)

    def _norm_var(self, x):
        return sigmoid(x) - 0.5


def disambiguate_one_table(table_fp, dest_fp, m_e_pred, part_prior,
                           alpha, beta, verbose):
    dis = JointDisambiguation(table_fp, dest_fp, m_e_pred, part_prior, alpha, beta, verbose)


def run_proc(param):
    disambiguate_one_table(param[0], param[1], param[2], param[3], param[4], param[5], param[6])


class PairLinkingFast:
    def __init__(self, mention_abstract_emb_fp, prior_fp, has_attention, has_lstm=True):
        if has_attention:
            if has_lstm:
                out_dir = config.attention_mgcn_elstm_model
            else:
                out_dir = config.attention_gcn_model
            match_model_fp = os.path.join(out_dir, 'best_model')
            self.predictor = AttentionPredictor(match_model_fp, mention_abstract_emb_fp, has_lstm=has_lstm)
        else:
            match_model_fp = os.path.join(config.base_gcn_lstm_model, 'best_model')
            self.predictor = Predictor(match_model_fp, mention_abstract_emb_fp)
        _, self.prior = load_prior(prior_fp, topn=config.candidate_topn)

    def get_m_e_pred(self, table):
        # predictor很耗费时间，按batch，先计算好
        pred_t0 = time.time()
        batch = []
        for item in table:
            for c, c_a in item['candidates_abstract'].items():
                item_copy = copy.deepcopy(item)
                item_copy['entity'] = c
                item_copy['abstract'] = c_a
                batch.append(item_copy)

        batch_pred = []
        if len(batch) == 1:
            batch_pred.append(self.predictor.predict_one(batch[0]))
        elif len(batch) >= 2:
            span = 200
            for i in range(0, len(batch), span):
                temp = batch[i:min(len(batch), i + span)]
                if len(temp) >= 2:
                    preds = self.predictor.predict_batch(temp)
                    batch_pred.extend(preds)
                elif len(temp) == 1:
                    pred = self.predictor.predict_one(temp[0])
                    batch_pred.append(pred)
        assert len(batch) == len(batch_pred), 'empty table???'
        print('<get_m_e_pred>')
        print('mention entity pairs', len(batch))
        print('predict time', time.time() - pred_t0, 's')

        m_e_pred = {}
        for item, p in zip(batch, batch_pred):
            m_e_pred[(item['r_idx'], item['c_idx'], item['mention'], item['entity'])] = p
        return m_e_pred

    def get_part_prior(self, table):
        part_prior = defaultdict(dict)
        mention_cnt = len(table)
        for i in range(mention_cnt):
            for c in table[i]['candidates_embedding']:
                part_prior[table[i]['mention']][c] = self.prior[table[i]['mention']][c]
        return part_prior

    @timethis
    def link_table(self, table_fp, dest_fp, verbose=0, max_mention_cnt=100, alpha=0.5, beta=0.5):
        with open(table_fp, 'r', encoding='utf-8') as f:
            table = json.load(f)
            if len(table) > max_mention_cnt:
                print('Too large table, give up!!!')
                return False

            part_prior = self.get_part_prior(table)
            m_e_pred = self.get_m_e_pred(table)
            dis = JointDisambiguation(table_fp, dest_fp, m_e_pred, part_prior, alpha, beta, verbose)
            return True

    @timethis
    def link_tables(self, table_dir, dest_dir, max_mention_cnt=100, test_cnt=None, verbose=0, alpha=0.5, beta=0.5):
        if os.path.exists(dest_dir):
            print('clear', dest_dir)
            shutil.rmtree(dest_dir)
        create_dir(dest_dir)

        table_names = os.listdir(table_dir)
        if test_cnt is None:
            test_cnt = len(table_names)
        cnt = 0
        for i, name in enumerate(table_names):
            if cnt >= test_cnt:
                break
            status = self.link_table(os.path.join(table_dir, name),
                                     os.path.join(dest_dir, name),
                                     verbose=verbose,
                                     max_mention_cnt=max_mention_cnt,
                                     alpha=alpha,
                                     beta=beta)
            if status:
                cnt += 1
            print_line(s='%d/%d' % (i + 1, len(table_names)))
        print('****', 'alpha', alpha, 'beta', beta)
        check_result(dest_dir)

    @timethis
    def multiprocess_link_tables(self, table_dir, dest_dir,
                                 alpha=0.5, beta=0.5,
                                 verbose=1, max_mention_cnt=100000, test_cnt=10000):
        if os.path.exists(dest_dir):
            print('clear', dest_dir)
            shutil.rmtree(dest_dir)
        create_dir(dest_dir)

        time_begin = time.time()
        table_names = os.listdir(table_dir)
        all_param = []
        for i, table_name in enumerate(table_names):
            table_fp = os.path.join(table_dir, table_name)
            dest_fp = os.path.join(dest_dir, table_name)

            if i >= test_cnt:
                break

            with open(table_fp, 'r', encoding='utf-8') as f:
                table = json.load(f)
                mention_cnt = len(table)
                if mention_cnt > max_mention_cnt:
                    print('Too large table({}), give up {}'.format(mention_cnt, table_fp))
                    continue
                part_prior = self.get_part_prior(table)
                m_e_pred = self.get_m_e_pred(table)
            assert m_e_pred, 'm_e_pred没被正确'
            all_param.append((table_fp, dest_fp, m_e_pred, part_prior, alpha, beta, verbose))
            print('table param %d/%d' % (i+1, len(table_names)))
        
        print('*' * 40)
        print('prepare parameters', time.time() - time_begin, 's')
        print('execute tables', len(all_param))

        pool = Pool(processes=8)  # 只开四个进程，防止资源过度占用
        pool.map(run_proc, all_param)
        pool.close()
        pool.join()

        time.sleep(10)
        print('***', 'alpha', alpha, 'beta', beta)
        check_result(dest_dir)


if __name__ == '__main__':
    pass
