import queue
from collections import defaultdict
from multiprocessing import Process, Queue
from pathlib import Path
from os import cpu_count

import numpy as np
import pandas as pd
from penman import load as penman_load
from penman.models.amr import model as amr_model

from utils.print_metric_result import print_table, summary
from argparse import ArgumentParser

from utils.runner import ExperimentTriple, FUNCTIONS, REPR_METRICS

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--annotations', '-ann', '-a', type=Path, default=Path('./resources/amr-annotations'))
    parser.add_argument('--baseline-results', '-base', '-b', type=Path, default=Path('./resources/amr-baselines'))
    parser.add_argument('--robustness', '-robust', '-r', type=Path, default=Path('./resources/amr-robustness'))
    parser.add_argument('--output', '-out', '-o', type=Path, default=Path('./result'))
    parser.add_argument('--num-cpus', '-cpu', '-c', type=int, default=int(cpu_count() * 0.8))
    args = parser.parse_args()

    experiments = []
    for baseline in sorted(args.baseline_results.glob('**/*.txt')):
        # for paraphrase
        # amr_version = baseline.name[0:-4]

        # for normal baselines
        amr_version = baseline.parent.parent.name

        experiments.append(ExperimentTriple(name=f'v{amr_version[3:]}_{baseline.stem}',
                                            ref=str((args.annotations / (amr_version + '-test.txt')).absolute()),
                                            hyp=str(baseline.absolute())))
    # skip robustness test for additional paraphrasing results
    for robustness in sorted(args.robustness.glob('*.txt')):
        experiments.append(ExperimentTriple(name=robustness.stem,
                                            ref=str((args.robustness / 'original.txt').absolute()),
                                            hyp=str(robustness.absolute())))

    experiments = sorted(experiments, key=lambda t: t.name, reverse=True)

    graph_ids = {exp.name: [graph.metadata['id'] for graph in penman_load(exp.ref, model=amr_model)]
                 for exp in experiments}

    if not args.output.exists():
        args.output.mkdir(parents=True)

    if not Path('./.tmp').exists():
        Path('./.tmp').mkdir(parents=True)
    else:
        for f in Path('./.tmp').iterdir():
            f.unlink()

    processes = []
    result_queue = Queue()
    for metric in FUNCTIONS:
        parallel_count = max(1, args.num_cpus // len(FUNCTIONS))
        for i in range(parallel_count):
            chunk = experiments[i::parallel_count]
            processes.append(Process(target=metric, args=(chunk, result_queue)))
            processes[-1].start()

    overall_results = []
    per_item_results = []
    final_table = defaultdict(dict)
    while not result_queue.empty() or any(proc.is_alive() for proc in processes):
        try:
            result_triple = result_queue.get(timeout=60)
        except queue.Empty:
            continue

        exp_name, key, result = result_triple
        if result is None:
            final_table[exp_name][key] = ()
        else:
            overall_results.append(dict(name=exp_name, metric=key, score=float(result.overall), seconds=result.time))

            if result.per_item is not None:
                assert len(result.per_item) == len(graph_ids[exp_name]), \
                    f'Per-item result has different length compared to the expected: ' \
                    f'{len(result.per_item)} vs {len(graph_ids[exp_name])}'
                scores = [float(s) for s in result.per_item]
                for i, item_score in zip(graph_ids[exp_name], scores):
                    per_item_results.append(dict(name=exp_name, metric=key, item=i, score=item_score))

            final_table[exp_name][key] = (result.overall, result.time, '', float('nan'))

        # Print as table form (without statistical test)
        print_table(final_table, index=sorted(exp.name for exp in experiments),
                    columns=sorted({k for d in final_table.values() for k in d.keys()}))

        # Write intermediate collected results as CSV
        pd.DataFrame(overall_results).to_csv(str(args.output / 'overall.csv'))
        pd.DataFrame(per_item_results).to_csv(str(args.output / 'per-item.csv'))

    for proc in processes:
        proc.close()

    # Print summary with statistical test (only for the selected metrics)
    summary(args.output, REPR_METRICS)
