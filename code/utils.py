import numpy as np
import json
import config
import os
import os.path
import time
import math
from urllib.parse import unquote, urlparse
import string
import re
from functools import wraps

from prettytable import PrettyTable
import warnings


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


def timethis(func):
    """
    Decorator that reports the execution time.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, 'time elapse', end-start, 's')
        return result
    return wrapper


def sigmoid(x):
    if x < -709:  # exp(710)就会溢出
        return 0.0
    return 1 / (1 + math.exp(-x))


def create_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def print_line(s=''):
    print()
    print('-' * (79-len(s)), s)


def is_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def clean_mention(m):
    m = re.sub(r'\s+', ' ', m)
    m = m.strip()
    m = re.sub(r'\s([!"#$%&()*+,-./:;<=>?@[\]^_`{|}~](?:\s|$))', r'\1', m)  # 去除标点之前的空格，而不是之后的
    m = re.sub(r"\s([!'#$%&()*+,-./:;<=>?@[\]^_`{|}~](?:\s|$))", r'\1', m)
    m = m.strip(string.punctuation + string.digits + ' ')  # 去除两端数字符号空格
    return m


def check_result(director, table_names=None):
    print_line('RESULT ' + str(director))
    mention_cnt = 0
    correct_cnt = 0
    if table_names is None:
        table_result_names = os.listdir(director)
    else:
        table_result_names = table_names

    each_table_acc = []
    each_table_p = []  # precision
    each_table_r = []  # recall
    each_table_f1 = []  # f1
    mention_golden_e = set()
    mention_result_e = set()

    for name in table_result_names:
        fp = os.path.join(director, name)
        with open(fp, 'r', encoding='utf-8') as f:
            result = json.load(f)
            if len(result) == 0:
                continue
            one_table_correct_cnt = 0
            one_table_golden_e = set()
            one_table_result_e = set()
            for item in result:
                one_table_golden_e.add((item['mention'], item['entity']))
                one_table_result_e.add((item['mention'], item['disambiguation']))
                mention_cnt += 1
                if item['entity'] == item['disambiguation']:
                    correct_cnt += 1
                    one_table_correct_cnt += 1
            each_table_acc.append(one_table_correct_cnt / len(result))
            mention_golden_e |= one_table_golden_e
            mention_result_e |= one_table_result_e

            p = len(one_table_golden_e & one_table_result_e) / len(one_table_golden_e)
            r = len(one_table_golden_e & one_table_result_e) / len(one_table_result_e)
            if p > 0 and r > 0:
                f1 = 2 * p * r / (p + r)
            else:
                f1 = 0.
            each_table_p.append(p)
            each_table_r.append(r)
            each_table_f1.append(f1)

    print('micro-accuracy: 所有cell正确的数量 / 整个数据集cell的数量')
    print('micro-accuracy: %d/%d=%.4f' % (correct_cnt, mention_cnt, correct_cnt/mention_cnt))

    print('macro-accuracy: 每个表格accuracy的平均')
    print('macro-accuracy: %.4f' % (sum(each_table_acc) / len(each_table_acc)))

    print_line()

    print('!!!将表格中的相同mention看作是一样的话，则计算precision和recall是有效的，否则precision==recall，没有意义')
    micro_p = len(mention_golden_e & mention_result_e) / len(mention_golden_e)
    micro_r = len(mention_golden_e & mention_result_e) / len(mention_result_e)
    if micro_p > 0 and micro_r > 0:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    else:
        micro_f1 = 0.
    print('micro-precision', micro_p)
    print('micro-recall', micro_r)
    print('micro-f1', micro_f1)

    print('macro-X: 将每个表格的指标平均')
    print('macro-precision', sum(each_table_p) / len(each_table_p))
    print('macro-recall', sum(each_table_r) / len(each_table_r))
    print('macro-f1', sum(each_table_f1) / len(each_table_f1))


if __name__ == '__main__':
    pass
