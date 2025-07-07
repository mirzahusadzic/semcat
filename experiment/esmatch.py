from smatch import *
from standardizer import *


def score_amr_pairs(f1, f2, justinstance=False, justattribute=False, justrelation=False):
    """
    Score one pair of AMR lines at a time from each file handle
    :param f1: file handle (or any iterable of strings) to read AMR 1 lines from
    :param f2: file handle (or any iterable of strings) to read AMR 2 lines from
    :param justinstance: just pay attention to matching instances
    :param justattribute: just pay attention to matching attributes
    :param justrelation: just pay attention to matching relations
    :return: generator of cur_amr1, cur_amr2 pairs: one-line AMR strings
    """
    # matching triple number, triple number in test file, triple number in gold file
    total_match_num = total_test_num = total_gold_num = 0
    # Read amr pairs from two files
    for sent_num, (cur_amr1, cur_amr2) in enumerate(generate_amr_lines(f1, f2), start=1):
        # ---- E-SMATCH MODIFICATION START ----
        try:
            cur_amr1 = to_standard_amr(cur_amr1)
            cur_amr2 = to_standard_amr(cur_amr2)
        except:
            # If AMR has errors inside, give zero.
            match_triple_dict.clear()
            if not single_score:
                if veryVerbose:
                    print("F-score:", "0.0", file=DEBUG_LOG)
                yield 0.00, 0.00, 0.00
            continue
        # ---- E-SMATCH MODIFICATION END   ----

        best_match_num, test_triple_num, gold_triple_num = get_amr_match(cur_amr1, cur_amr2,
                                                                         sent_num=sent_num,  # sentence number
                                                                         justinstance=justinstance,
                                                                         justattribute=justattribute,
                                                                         justrelation=justrelation)
        total_match_num += best_match_num
        total_test_num += test_triple_num
        total_gold_num += gold_triple_num
        # clear the matching triple dictionary for the next AMR pair
        match_triple_dict.clear()
        if not single_score:  # if each AMR pair should have a score, compute and output it here
            yield compute_f(best_match_num, test_triple_num, gold_triple_num)
    if verbose:
        print("Total match number, total triple number in AMR 1, and total triple number in AMR 2:", file=DEBUG_LOG)
        print(total_match_num, total_test_num, total_gold_num, file=DEBUG_LOG)
        print("---------------------------------------------------------------------------------", file=DEBUG_LOG)
    if single_score:  # output document-level smatch score (a single f-score for all AMR pairs in two files)
        yield compute_f(total_match_num, total_test_num, total_gold_num)


def main(arguments):
    """
    Main function of smatch score calculation
    """
    global verbose
    global veryVerbose
    global iteration_num
    global single_score
    global pr_flag
    global match_triple_dict
    # set the iteration number
    # total iteration number = restart number + 1
    iteration_num = arguments.r + 1
    if arguments.ms:
        single_score = False
    if arguments.v:
        verbose = True
    if arguments.vv:
        veryVerbose = True
    if arguments.pr:
        pr_flag = True
    # significant digits to print out

    floatdisplay = "%%.%df" % arguments.significant
    for (precision, recall, best_f_score) in score_amr_pairs(args.f[0], args.f[1],
                                                             justinstance=arguments.justinstance,
                                                             justattribute=arguments.justattribute,
                                                             justrelation=arguments.justrelation):
        # print("Sentence", sent_num)
        if pr_flag:
            print("Precision: " + floatdisplay % precision)
            print("Recall: " + floatdisplay % recall)
        print("F-score: " + floatdisplay % best_f_score)
    args.f[0].close()
    args.f[1].close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smatch calculator")
    parser.add_argument(
        '-f',
        nargs=2,
        required=True,
        type=argparse.FileType('r'),
        help=('Two files containing AMR pairs. '
              'AMRs in each file are separated by a single blank line'))
    parser.add_argument(
        '-r',
        type=int,
        default=4,
        help='Restart number (Default:4)')
    parser.add_argument(
        '--significant',
        type=int,
        default=2,
        help='significant digits to output (default: 2)')
    parser.add_argument(
        '-v',
        action='store_true',
        help='Verbose output (Default:false)')
    parser.add_argument(
        '--vv',
        action='store_true',
        help='Very Verbose output (Default:false)')
    parser.add_argument(
        '--ms',
        action='store_true',
        default=False,
        help=('Output multiple scores (one AMR pair a score) '
              'instead of a single document-level smatch score '
              '(Default: false)'))
    parser.add_argument(
        '--pr',
        action='store_true',
        default=False,
        help=('Output precision and recall as well as the f-score. '
              'Default: false'))
    parser.add_argument(
        '--justinstance',
        action='store_true',
        default=False,
        help="just pay attention to matching instances")
    parser.add_argument(
        '--justattribute',
        action='store_true',
        default=False,
        help="just pay attention to matching attributes")
    parser.add_argument(
        '--justrelation',
        action='store_true',
        default=False,
        help="just pay attention to matching relations")

    args = parser.parse_args()
    main(args)
