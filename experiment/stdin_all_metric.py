import queue
from collections import defaultdict
from multiprocessing import Queue
from pathlib import Path

from tqdm import tqdm
from utils.print_metric_result import print_table
from utils.runner import ExperimentTriple, FUNCTIONS_NO_WWLK_THETA

if __name__ == '__main__':
    TMP_REF = Path('.tmp', 'ref')
    TMP_HYP = Path('.tmp', 'hyp')

    if not TMP_REF.parent.exists():
        TMP_REF.parent.mkdir(parents=True)
    else:
        for f in TMP_REF.parent.iterdir():
            f.unlink()

    while True:
        reference = input('An encoded string for reference AMR: ').strip()
        if not reference:
            print('You input empty string! Process will be terminated.')
            break

        hypothesis = input('An encoded string for AMR to be compared: ').strip()
        if not hypothesis:
            print('You input empty string! Process will be terminated.')
            break

        TMP_REF.write_text(reference)
        TMP_HYP.write_text(hypothesis)

        experiments = [ExperimentTriple(name='stdin', ref=str(TMP_REF.absolute()), hyp=str(TMP_HYP.absolute()))]

        final_table = defaultdict(dict)
        result_queue = Queue()
        for metric in tqdm(FUNCTIONS_NO_WWLK_THETA, desc='Metrics'):
            metric(experiments, result_queue)

        while not result_queue.empty():
            try:
                result_triple = result_queue.get(timeout=60)
            except queue.Empty:
                continue

            exp_name, key, result = result_triple
            if result is None:
                final_table[exp_name][key] = ()
            else:
                final_table[exp_name][key] = (result.overall, result.time, '', float('NaN'))

        # Print as table form
        print_table(final_table, index=sorted(exp.name for exp in experiments),
                    columns=sorted({k for d in final_table.values() for k in d.keys()}), clear_all=False)
