import re
import subprocess
import uuid
from collections import namedtuple
from multiprocessing import Queue
from pathlib import Path
from sys import stderr
from time import sleep
from typing import List, Optional

import numpy as np

ProcessResult = namedtuple('ProcessResult', ('time', 'stdout'))
ExperimentTriple = namedtuple('ExperimentTriple', ('name', 'ref', 'hyp'))
ExperimentResult = namedtuple('ExperimentResult', ('time', 'overall', 'per_item'))


def read_time_data(path):
    time_path = Path(path)
    value = None
    while value is None:
        sleep(.1)
        if not time_path.exists():
            continue

        with time_path.open('rt') as fp:
            for line in fp.readlines():
                if line.startswith('TS'):
                    value = eval(line[2:].strip())

    time_path.unlink(missing_ok=True)
    return value or float('NaN')


def execute_and_get_timings(cmd_args: List[str], **kwargs) -> Optional[ProcessResult]:
    time_id = str(Path(f'./.tmp/.time.{uuid.uuid4()}').absolute())
    result = subprocess.run(['/usr/bin/time', '-o', time_id, '-f', 'TS%U+%S'] + cmd_args,
                            capture_output=True, **kwargs)

    if result.returncode or result.stderr:
        print(f'EXEC {" ".join(cmd_args)} = {result.returncode}', file=stderr)
        print('ERR:', result.stderr.decode('UTF-8'), file=stderr)

    if result.returncode == 0:
        time = read_time_data(time_id)
        return ProcessResult(time, result.stdout.decode('UTF-8'))
    else:
        return None


def run_rose_metric(experiments: List[ExperimentTriple], queue: Queue):
    for iteration in ['5', '4', '3']:
    # for iteration in ['5', '2']:
        # for threshold in ['.99', '.90', '.80', '.75']:  # 1/100, 1/10, 1/5, 1/4
        for threshold in ['.99']:
            for exp in experiments:
                item_file = f'./.tmp/.rose.{uuid.uuid4()}'
                exp_result = execute_and_get_timings(['python3', '/mnt/sda/Code_local/RoSE/metric/rose.py', '-r', exp.ref, '-p', exp.hyp, '-o',
                                                      item_file, '-n', iteration, '-t', threshold, '-v', 'WARNING'],
                                                     cwd='./')

                result = None
                if exp_result is not None:
                    tmp_path = Path(item_file)
                    with tmp_path.open('rt') as fp:
                        per_item_result = [l.strip() for l in fp.readlines() if l.strip()]

                    tmp_path.unlink()
                    result = ExperimentResult(exp_result.time, exp_result.stdout.strip(), per_item_result)

                queue.put((exp.name, f'RoSE{iteration}-{threshold[1:]}', result))


def run_sembleu_metric(experiments: List[ExperimentTriple], queue: Queue, call='sembleu'):
    # SemBleu does not provide per-item score and overall score at the same time. We need to execute it once more.
    for exp in experiments:
        score = None
        time_measure = None
        item_scores = []

        for exec_file in ['./eval.sh', './src/per_sentence_eval.py']:
            exp_result = execute_and_get_timings([exec_file, exp.hyp, exp.ref], cwd=f'./{call}')

            if exp_result is None:
                break

            if exec_file == './eval.sh':
                for line in exp_result.stdout.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    if re.fullmatch('^\\d+(\\.\\d+)?$', line):
                        score = float(line)
                        break
                else:
                    score = float('NaN')
                time_measure = exp_result.time
            else:
                item_scores = [x[6:].strip() for x in exp_result.stdout.split('\n') if x.startswith('score:')]

        if score is not None and time_measure is not None and item_scores:
            queue.put((exp.name, call.upper().replace('SHARP', '#'), ExperimentResult(time_measure, f'{score:.4f}', item_scores)))
        else:
            queue.put((exp.name, call.upper().replace('SHARP', '#'), None))


def run_smatch_metric(experiments: List[ExperimentTriple], queue: Queue, call='smatch'):
    # Smatch does not provide per-item score and overall score at the same time. We need to execute it once more.
    for exp in experiments:
        time_measure = None
        fscores = []

        for per_score_option in [['--ms'], []]:
            exp_result = execute_and_get_timings(['python3', '-m', call, '-f', exp.ref, exp.hyp,
                                                  '--significant', '4', *per_score_option])
            if exp_result is None:
                break

            for line in exp_result.stdout.split('\n'):
                if line.startswith('F-score:'):
                    fscores.append(line[8:].strip())

            if not len(per_score_option):
                time_measure = exp_result.time

        if time_measure is not None and len(fscores) > 1:
            queue.put((exp.name, call.upper(), ExperimentResult(time_measure, fscores[-1], fscores[:-1])))
        else:
            queue.put((exp.name, call.upper(), None))


def run_wwlk_k3e2n_metric(experiments: List[ExperimentTriple], queue: Queue):
    # Run WWLK-k3e2n (score range: [-1, 1] due to the cosine similarity)
    for exp in experiments:
        exp_result = execute_and_get_timings(['python3', 'main_wlk_wasser.py', '-stability_level', '15',
                                              '-k', '3', '--edge_to_node_transform',
                                              '-random_init_relation', 'constant', '-output_type', 'score',
                                              '-round_decimals', '4', '-a', exp.ref, '-b', exp.hyp],
                                             cwd='./weisfeiler-leman-amr-metrics/src')
        if exp_result is None:
            queue.put((exp.name, 'WWLK-k3e2n', None))
            continue

        # Adjust scale [-1, 1] to [0, 1] to compare with the other metrics
        scores = [(float(s.strip()) + 1) / 2.0
                  for s in exp_result.stdout.split('\n') if s.strip()]
        overall = np.mean(scores)

        queue.put((exp.name, 'WWLK-k3e2n',
                   ExperimentResult(exp_result.time, '%.4f' % float(overall), ['%.4f' % s for s in scores])))


def run_wwlk_theta_metric(experiments: List[ExperimentTriple], queue: Queue):
    # Run WWLK-theta (score range: [-1, 1] due to the cosine similarity)
    bamboo_sts = Path('./weisfeiler-leman-amr-metrics/bamboo-amr-benchmark/sts')
    bamboo_sts_reify = bamboo_sts / 'reify'

    for exp in experiments:
        exp_result = execute_and_get_timings(['python3', 'main_wlk_wasser_optimized.py',
                                              '-a_train', str((bamboo_sts_reify / 'src.train.amr').absolute()),
                                              '-a_dev', str((bamboo_sts_reify / 'src.dev.amr').absolute()),
                                              '-a_test', exp.ref,
                                              '-b_train', str((bamboo_sts_reify / 'tgt.train.amr').absolute()),
                                              '-b_dev', str((bamboo_sts_reify / 'tgt.dev.amr').absolute()),
                                              '-b_test', exp.hyp,
                                              '-y_train', str((bamboo_sts / 'train.y').absolute()),
                                              '-y_dev', str((bamboo_sts / 'dev.y').absolute()),
                                              '-log_level', '50'],
                                             cwd='./weisfeiler-leman-amr-metrics/src')
        if exp_result is None:
            queue.put((exp.name, 'WWLK-theta', None))
            continue

        scores = []
        for s in exp_result.stdout.split('\n'):
            s = s.strip()
            if (not s) or s.startswith('new high'):
                continue

            # Adjust scale [-1, 1] to [0, 1] to compare with the other metrics
            scores.append((float(s) + 1) / 2.0)

        overall = np.mean([s for s in scores])
        queue.put((exp.name, 'WWLK-theta',
                   ExperimentResult(exp_result.time, '%.4f' % float(overall), ['%.4f' % s for s in scores])))


def run_s2match_metric(experiments: List[ExperimentTriple], queue: Queue):
    for exp in experiments:
        time_measure = None
        final_score = None
        fscores = []

        for per_score_option in [['--ms'], []]:
            exp_result = execute_and_get_timings(['python3', './py3-Smatch-and-S2match/smatch/s2match.py',
                                                  *per_score_option, '-f', exp.hyp, exp.ref,
                                                  '-vectors', './vectors/glove.6B.100d.txt', '-similarityfunction',
                                                  'cosine', '-cutoff', '0.5', '-diffsense', '0.5'],
                                                 cwd='./amr-metric-suite')

            if exp_result is None:
                break

            if len(per_score_option):
                for line in exp_result.stdout.split('\n'):
                    if line.startswith('Smatch score F1'):
                        fscores.append(line[15:].strip())
            else:
                time_measure = exp_result.time
                line = exp_result.stdout.strip()
                if line.startswith('Document F-score:'):
                    line = line[17:].strip().split(',')[1]  # Same score repeated twice (different precision level)
                    final_score = line.strip()

        if time_measure is not None:
            queue.put((exp.name, 'S2MATCH',
                       ExperimentResult(exp_result.time, '%.4f' % float(final_score), fscores)))
        else:
            queue.put((exp.name, 'S2MATCH', None))


def run_sema_metric(experiments: List[ExperimentTriple], queue: Queue):
    for exp in experiments:
        total_score = None
        total_time = None
        scores = []

        for executable in ['./sema.py', './sema_per_eval.py']:
            exp_result = execute_and_get_timings(['python3', executable, '-t', exp.hyp, '-g', exp.ref],
                                                 cwd='./sema')

            if exp_result is None:
                break

            if executable == './sema.py':
                line = exp_result.stdout.strip()
                if line.startswith('SEMA: '):
                    total_score = line.strip().split('F1')[1].strip()
                    total_time = exp_result.time
            else:
                for line in exp_result.stdout.split('\n'):
                    line = line.strip()
                    if line.startswith('SEMA: '):
                        scores.append(line.split('F1')[1].strip())

        if total_time is not None:
            queue.put((exp.name, 'SEMA',
                       ExperimentResult(total_time, '%.4f' % float(total_score), scores)))
        else:
            queue.put((exp.name, 'SEMA', None))


def run_smatchpp_metric(experiments: List[ExperimentTriple], queue: Queue, call='smatchpp'):
    for exp in experiments:
        exp_result = execute_and_get_timings(['python3', '-m', 'smatchpp', '-a', exp.hyp, '-b', exp.ref,
                                              '-solver', 'ilp', '-edges', 'dereify', '-score_dimension', 'main',
                                              '-score_type', 'pairwise', '-log_level', '50',
                                              '--bootstrap', '--remove_duplicates'],
                                             cwd=f'./{call}')

        if exp_result is None:
            break

        fscores = []
        for line in exp_result.stdout.split('\n'):
            line = line.strip()
            if 'F1:' not in line or 'F1: [' in line:
                continue

            line = float(line.split('F1:')[1].strip().split()[0]) / 100.0
            fscores.append(line)

        overall = np.mean(fscores)
        queue.put((exp.name, call.upper().replace('PP', '++mac').replace('SHARP', '#mac'),
                   ExperimentResult(exp_result.time, '%.4f' % float(overall), ['%.4f' % float(s) for s in fscores])))


def run_esmatch_metric(experiments: List[ExperimentTriple], queue: Queue):
    run_smatch_metric(experiments, queue, call='esmatch')


def run_smatchsharp_metric(experiments: List[ExperimentTriple], queue: Queue):
    run_smatchpp_metric(experiments, queue, call='esmatchpp')


FUNCTIONS_NO_WWLK_THETA = [
    run_rose_metric,
    run_sema_metric,
    run_sembleu_metric,
    run_smatch_metric,
    run_s2match_metric,
    run_wwlk_k3e2n_metric,
    run_esmatch_metric,
    run_smatchsharp_metric,
    run_smatchpp_metric,
]


FUNCTIONS = FUNCTIONS_NO_WWLK_THETA + [run_wwlk_theta_metric]
REPR_METRICS = ['RoSE3-99', 'RoSE4-99', 'RoSE5-99',
                'SMATCH', 'S2MATCH', 'SEMBLEU', 'SMATCH++mac', 'WWLK-k3e2n',
                'ESMATCH', 'ESMATCH++mac']
