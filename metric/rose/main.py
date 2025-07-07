import sys

from .scorer_hash import RoSE


def file_main():
    from argparse import ArgumentParser
    from pathlib import Path

    import logging

    parser = ArgumentParser()
    parser.add_argument('--reference', '-ref', '-r', type=Path, nargs='+',
                        help='List of reference AMR annotation files')
    parser.add_argument('--predicted', '--generated', '-pred', '-gen', '-p', '-g', type=Path, nargs='+',
                        help='List of generated AMR annotation files')
    parser.add_argument('--num-iterations', '--iter', '-iter', '-n', type=int, default=5,
                        help='Maximum number of iterations for WL algorithm (Default: 5, Recommend: 5 and 2)')
    parser.add_argument('--similarity-threshold-tau', '--threshold', '-tau', '-t', type=float, default=0.99,
                        help='Minimum threshold of marking two nodes as compatible '
                             '(Default: 0.99, Recommend: 0.75 and 0.99)')
    parser.add_argument('--rounding', '--round', '-round', type=int, default=5,
                        help='Number of precision digits under the decimal point')
    parser.add_argument('--output-txt', '--output', '-out', '-o', type=Path,
                        help='Path to store per-item scores. '
                             'If not exists, this module just print the average of them.')
    parser.add_argument('--verbose', '-v', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                        help='Log Level when computing RoSE. If not specified, RoSE will run in silent mode.')
    parser.add_argument('--mode', '-m', choices=['AMR', 'DP', 'SRL'], default='AMR',
                        help='mode for RoSE.') # 23. 3. 29. 추가실험 위해 설정
    args = parser.parse_args()

    if args.verbose is not None:
        import penman.transform
        import penman.layout
        penman.transform.logger.setLevel(logging.WARN)
        penman.layout.logger.setLevel(logging.WARN)

        logging.root.setLevel(args.verbose)
        logging.root.addHandler(logging.StreamHandler(sys.stderr))

        scorer = RoSE(verbosity=args.verbose, num_iterations=args.num_iterations,
                      precision_digit=args.rounding, similarity_threshold_tau=args.similarity_threshold_tau, mode=args.mode)
    else:
        scorer = RoSE(num_iterations=args.num_iterations,
                      precision_digit=args.rounding, similarity_threshold_tau=args.similarity_threshold_tau)

    score = scorer.compute_from_files(args.reference, args.predicted, return_all_scores=True)

    precision_format = '%%.%df' % args.rounding
    print(precision_format % score[scorer.name()])

    if args.output_txt:
        with args.output_txt.open('w+t') as fp:
            fp.write('\n'.join([precision_format % s
                                for s in score[f'list_{scorer.name()}']]))


def stdin_main():
    import logging
    logging.root.setLevel(logging.DEBUG)
    logging.root.addHandler(logging.StreamHandler(sys.stderr))

    scorer = RoSE(verbosity=logging.DEBUG)
    while True:
        reference = input('An encoded string for reference AMR: ').strip()
        if not reference:
            print('You input empty string! Process will be terminated.')
            break

        hypothesis = input('An encoded string for AMR to be compared: ').strip()
        if not hypothesis:
            print('You input empty string! Process will be terminated.')
            break

        score_dict = scorer.compute_from_string(reference, hypothesis)
        print('SCORE:', score_dict)
