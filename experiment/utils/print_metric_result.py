from collections import defaultdict, Counter
from os import system
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import binom_test, binom_test_reject_interval
from scipy.stats import spearmanr
from datetime import datetime
from pathlib import Path


def get_star(p):
    if p > 0.1 or np.isnan(p):
        return ''
    elif p > 0.05:
        return '+'
    elif p > 0.01:
        return '*'
    elif p > 0.001:
        return '**'
    else:
        return '***'


def prop_test(samples, alternative, threshold):
    # samples = Counter([0.9999 < float(s) < 1.0001 for s in samples])
    samples = Counter([float(s) == 1 for s in samples])
    p_value = binom_test(samples[True], sum(samples.values()), alternative=alternative, prop=threshold)
    x_low, x_high = binom_test_reject_interval(samples[True], sum(samples.values()), alternative=alternative)
    star = get_star(p_value)

    return p_value, x_low, x_high, star


def print_table(data: dict, index: list, columns: list, clear_all=True):
    if clear_all:
        system('clear')
    print(f'{datetime.now().ctime()}\n')
    for c in range(0, len(columns), 10):
        print(' ' * 24 + ''.join(f'{col:18s} ' for col in columns[c:c+10]))

    dash_len = 24 + 19 * 10
    print('-' * dash_len)

    header = ''
    for r, row in enumerate(index):
        if header and header != row[:2]:
            print('.' * dash_len)

        header = row[:2]
        print(f'{row[:20]:20s}', end='')
        print('...' if len(row) > 20 else '   ', end=' ')
        data_row = data.get(row, {})
        data_max = max({float(t[0]) for t in data_row.values() if len(t)}, default=0)

        for c, col in enumerate(columns):
            if c > 0 and c % 10 == 0:
                print('\n' + ' ' * 24, end='')

            value = data_row.get(col, None)
            if value is None:
                print(' ' * 19, end='')
            elif len(value):
                if float(value[0]) == data_max:
                    print(f'\033[1;96m', end='')

                if np.isnan(value[3]):
                    seconds = float(value[1])
                    if seconds > 10000:
                        seconds = '9999+'
                    else:
                        seconds = f'{float(value[1]):4.0f}s'
                    print(f'{float(value[0]):6.4f}\033[0;93m{value[2]:3s} \033[37m[{seconds}] \033[0m',
                          end=' ')
                else:
                    p = value[3]
                    if p > .9995:
                        pvalue = '>.999'
                    elif p < .0005:
                        pvalue = '<.000'
                    else:
                        pvalue = f'{p:5.4f}'[1:]
                    print(f'{float(value[0]):6.4f}\033[0;93m{value[2]:3s} \033[37m[{pvalue}] \033[0m',
                          end=' ')
            else:
                print(f'\033[91m           ERROR  \033[0m', end=' ')

        print('\033[0m\n', end='')


def summary(result_path: Path, metrics: list = []):
    overall = pd.read_csv((result_path / 'overall.csv').absolute(), header=0, index_col=0)
    per_item = pd.read_csv((result_path / 'per-item.csv').absolute(), header=0, index_col=0)

    final_table = defaultdict(dict)
    values = defaultdict(lambda: defaultdict(list))
    times = defaultdict(lambda: defaultdict(lambda: 0.0))

    metrics_reported = set()
    baseline_reported = set()
    for _, row in overall.iterrows():
        exp_name = row['name']
        metric_name = row['metric']
        if metrics and metric_name not in metrics:
            continue

        metrics_reported.add(metric_name)
        item_data = per_item[(per_item['name'] == exp_name) & (per_item['metric'] == metric_name)]['score'].tolist()
        values[exp_name[:2].upper()][metric_name] += item_data
        times[exp_name[:2].upper()][metric_name] += row['seconds']
        if not exp_name.startswith('v'):
            if exp_name.startswith('ne'):
                p, _, _, s = prop_test(item_data, alternative='smaller', threshold=0.001)
            else:
                p, _, _, s = prop_test(item_data, alternative='larger', threshold=0.999)
        else:
            p = float('NaN')
            s = ''
            if exp_name.startswith('v3'):
                baseline_reported.add(exp_name)

        final_table[row['name']][row['metric']] = (row['score'], row['seconds'], s, p)

        # Add overall field
        for key in values:
            if metric_name not in values[key] or not values[key][metric_name]:
                continue
            final_table[f'{key}__MAX(s)'][metric_name] = (max(values[key][metric_name]), 0, '', float('nan'))
            final_table[f'{key}__MIN(s)'][metric_name] = (min(values[key][metric_name]), 0, '', float('nan'))
            final_table[f'{key}__P(s=1)'][metric_name] = (sum(s == 1 for s in values[key][metric_name]) /
                                                          len(values[key][metric_name]), 0, '', float('nan'))
            final_table[f'{key}__CPU Sec/ITEM'][metric_name] = (times[key][metric_name] / len(values[key][metric_name]),
                                                                0, 'sec', float('nan'))

    print_table(final_table, index=sorted(final_table.keys()),
                columns=metrics if metrics else sorted({k for d in final_table.values() for k in d.keys()}))

    # Add rank correlation only for v3
    metrics_reported = sorted(metrics_reported)
    rank_table = defaultdict(dict)
    for r, c in combinations(metrics_reported, r=2):
        metric_r = per_item[(per_item['name'].str.startswith('v3')) & (per_item['metric'] == r)]['score'].tolist()
        metric_c = per_item[(per_item['name'].str.startswith('v3')) & (per_item['metric'] == c)]['score'].tolist()
        if len(metric_r) != len(metric_c):
            continue

        rho, p = spearmanr(metric_r, metric_c)
        rank_table[r][c] = (rho, 0, get_star(p), p)

    print_table(rank_table, index=sorted(rank_table.keys()), columns=sorted(rank_table.keys()), clear_all=False)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--result', '-result', '-r', type=Path, default='/mnt/sda/Code_local/RoSE/experiment/result',
                        help='Path where overall.csv and per-item.csv stored.')
    parser.add_argument('--metrics', '-metrics', '-m', type=str, nargs='*',
                        help='List of metric names to be displayed. Leave empty to print all result.')
    args = parser.parse_args()
    summary(args.result, args.metrics)
