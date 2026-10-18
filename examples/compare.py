# Copyright 2026 Mirza Husadzic. SPDX-License-Identifier: GPL-3.0-only
"""Compare a relation with its equivalent reified AMR representation."""
import os

os.environ.setdefault("SOLVER_NAME", "HIGHS")

import pandas as pd
import penman
from penman.models.amr import model

from calc_SEMCAT import build_semcat
from rose import RoSE
from smatchpp import Smatchpp, preprocess, solvers
from smatchpp.eval_statistics import f1_score


def main():
    reference = "(k / knife :location (d / drawer))"
    hypothesis = "(b / be-located-at-91 :ARG1 (k / knife) :ARG2 (d / drawer))"
    local = Smatchpp(
        alignmentsolver=solvers.ILP(max_seconds=5),
        graph_standardizer=preprocess.AMRGraphStandardizer(
            lower=False, remove_quote=False, relabel_vars=False
        ),
    )
    matches, (lower, upper), _ = local.process_pair(reference, hypothesis)
    if abs(lower - upper) > 1e-6 or abs(lower - matches["main"][0]) > 1e-6:
        raise RuntimeError("The local alignment did not reach a proven optimum")
    local_score = float(f1_score(matches["main"]))
    global_score = float(RoSE(num_iterations=3).compute_graph_similarity(
        penman.decode(reference, model=model), penman.decode(hypothesis, model=model)
    ))
    frame = pd.DataFrame([
        dict(index=0, name="example", item=0, metric="ESMATCH++mac", score=local_score),
        dict(index=0, name="example", item=0, metric="RoSE3-99", score=global_score),
    ])
    result = build_semcat(frame, alpha=0.25, require_both=True)
    score = float(result.loc[result.metric == "SEMCAT", "score"].iloc[0])
    print(dict(local=local_score, global_=global_score, semcat=score))


if __name__ == "__main__":
    main()
