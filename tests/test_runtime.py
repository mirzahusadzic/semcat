# Copyright 2026 Mirza Husadzic. SPDX-License-Identifier: GPL-3.0-only
"""Semantic regressions and an exhaustive oracle, run against the installed wheel."""
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("SOLVER_NAME", "HIGHS")

import pandas as pd
import penman
from penman.models.amr import model

import amr_normalization
import calc_SEMCAT
import rose
import smatchpp
import standardizer_rose
from rose import RoSE
from smatchpp import Smatchpp, preprocess, solvers
from smatchpp.align import GraphAligner
from smatchpp.data_helpers import PenmanReader
from smatchpp.eval_statistics import f1_score
from smatchpp.score import IDTripleMatcher
from smatchpp.standardizer import to_standard_amr


class TraceAligner(GraphAligner):
    def align(self, *args):
        result = super().align(*args)
        self.mapping = self._get_var_map(result[0], result[1])
        return result


class LocalScorer:
    """Reuse immutable frame tables; keep each solver and pair's state separate."""
    def __init__(self):
        self.resources = {}

    def compare(self, reference, hypothesis):
        matcher = IDTripleMatcher()
        solver = solvers.ILP(max_seconds=5)
        aligner = TraceAligner(matcher, solver)
        local = Smatchpp(
            triplematcher=matcher, graph_aligner=aligner, alignmentsolver=solver,
            graph_standardizer=preprocess.AMRGraphStandardizer(
                lower=False, remove_quote=False, relabel_vars=False
            ),
            **self.resources,
        )
        matches, bounds, _ = local.process_pair(reference, hypothesis)
        self.resources.update(
            graph_scorer=local.graph_scorer, subgraph_extractor=local.subgraph_extractor
        )
        return local, matches["main"], bounds, aligner.mapping


class InstalledPackageTests(unittest.TestCase):
    def test_imports_resources_and_license_notices_are_installed(self):
        distribution = importlib.metadata.distribution("semcat-runtime")
        root = Path(distribution.locate_file("")).resolve()
        self.assertIn("site-packages", root.parts)
        self.assertTrue(root.is_relative_to(Path(sys.prefix).resolve()))
        for module in (amr_normalization, calc_SEMCAT, rose, smatchpp, standardizer_rose):
            self.assertTrue(Path(module.__file__).resolve().is_relative_to(root), module.__file__)
        self.assertEqual(distribution.metadata["License-Expression"], "GPL-3.0-only AND MIT")
        files = {str(path): path for path in distribution.files}
        self.assertEqual({name for name in files if name.startswith("rose/") and name.endswith(".py")},
                         {"rose/__init__.py", "rose/main.py", "rose/scorer_hash.py"})
        for suffix in (
            "licenses/LICENSING.md", "licenses/metric/LICENSE.txt",
            "licenses/experiment/esmatchpp/LICENSE", "licenses/LICENSES/fork-MIT.txt",
            "smatchpp/resource/reify_table.txt", "smatchpp/resource/amr_aspects.json",
            "smatchpp/resource/concept_groups.json",
            "smatchpp/resource/propbank-amr-frames-arg-descr.txt",
        ):
            matches = [path for name, path in files.items() if name.endswith(suffix)]
            self.assertEqual(len(matches), 1, suffix)
            self.assertGreater(distribution.locate_file(matches[0]).stat().st_size, 0)


class NormalizationTests(unittest.TestCase):
    def test_lexical_roles_and_true_inverses_at_each_stage(self):
        cases = [
            (":consist-of", ("a", ":consist-of", "b")),
            (":prep-out-of", ("a", ":prep-out-of", "b")),
            (":prep-on-behalf-of", ("a", ":prep-on-behalf-of", "b")),
            (":ARG0-of", ("b", ":ARG0", "a")),
            (":domain", ("b", ":mod", "a")),
            (":mod-of", ("b", ":mod", "a")),
        ]
        for role, expected in cases:
            with self.subTest(role=role):
                self.assertEqual(amr_normalization.canonical_triple(("a", role, "b")), expected)
                self.assertEqual(amr_normalization.canonical_triple(expected), expected)
                triples = [("a", role, "b")]
                preprocess.AMRGraphStandardizer()._deinvert_e(triples)
                self.assertEqual(triples, [expected])
                text, origins = to_standard_amr(f"(a / alpha {role} (b / beta))", with_origins=True)
                normalized = penman.decode(text, model=model)
                edges = {(origins[s], r, origins[t]) for s, r, t in normalized.edges()}
                self.assertEqual(edges, {expected})

    def test_duplicates_after_role_normalization(self):
        text = "(f / fail-01 :mod (u / urgent :domain f))"
        graph = penman.decode(to_standard_amr(text), model=model)
        self.assertEqual(len(graph.edges()), 1)

    def test_reader_preserves_literals_and_synthetic_root(self):
        for value in ["New York", "New  York", "NewYork", "a (b)", 'a "quote"', "path\\file", "Malmö"]:
            with self.subTest(value=value):
                literal = json.dumps(value, ensure_ascii=False)
                triples = PenmanReader().string2graph(f"(n / name :op1 {literal})")
                self.assertIn(("n", ":op1", literal), triples)
                self.assertIn(("ROOT_OF_GRAPH", ":root", "n"), triples)

    def test_reader_uses_amr_role_model(self):
        for role in [":consist-of", ":prep-out-of", ":prep-on-behalf-of"]:
            self.assertIn(("a", role, "b"), PenmanReader().string2graph(f"(a / alpha {role} (b / beta))"))
        self.assertIn(("b", ":ARG0", "a"), PenmanReader().string2graph("(a / alpha :ARG0-of (b / beta))"))

    def test_original_ids_survive_renaming_and_dereification(self):
        text, origins = to_standard_amr(
            "(event / be-located-at-91 :ARG1 (knife / knife) :ARG2 (drawer / drawer))",
            with_origins=True,
        )
        self.assertEqual(set(origins.values()), {"knife", "drawer"})
        self.assertEqual(set(origins), penman.decode(text, model=model).variables())
        self.assertIsInstance(to_standard_amr("(p / person)"), str)


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local = LocalScorer()

    def test_semantic_regressions(self):
        fail = "(f / fail-01 :mod (u / urgent))"
        cases = [
            ("identity", fail, fail, True),
            ("renaming", fail, "(x / fail-01 :mod (y / urgent))", True),
            ("rerooting", fail, "(u / urgent :mod-of (f / fail-01))", True),
            ("domain", fail, "(u / urgent :domain (f / fail-01))", True),
            ("negation", fail, "(f / fail-01 :mod (u / urgent) :polarity -)", False),
            ("qualification", fail, "(f / fail-01 :mod (u / frequent))", False),
            ("roles", "(s / sell-01 :ARG0 (c / company) :ARG1 (v / service))",
             "(s / sell-01 :ARG1 (c / company) :ARG0 (v / service))", False),
            ("sense", "(p / pay-01)", "(p / pay-04)", False),
            ("operator", "(a / and :op1 (f / frequent) :op2 (u / urgent))",
             "(a / or :op1 (f / frequent) :op2 (u / urgent))", False),
            ("reentrancy", "(s / see-01 :ARG0 (p / person) :ARG1 p)",
             "(s / see-01 :ARG0 (p / person) :ARG1 (q / person))", False),
            ("duplicate", "(s / see-01 :ARG0 (p / person))",
             "(s / see-01 :ARG0 (p / person) :ARG0 p)", True),
            ("reification", "(k / knife :location (d / drawer))",
             "(b / be-located-at-91 :ARG1 (k / knife) :ARG2 (d / drawer))", True),
            ("qualified reification", "(k / knife :location (d / drawer))",
             "(b / be-located-at-91 :ARG1 (k / knife) :ARG2 (d / drawer) :polarity -)", False),
            ("Unicode identity", '(n / name :op1 "Mirža")', '(n / name :op1 "Mirža")', True),
            ("Unicode distinction", '(n / name :op1 "Malmo")', '(n / name :op1 "Malmö")', False),
            ("Unicode concept", '(n / Malmo)', '(n / Malmö)', False),
            ("literal case", '(n / name :op1 "A")', '(n / name :op1 "a")', False),
            ("literal spaces", '(n / name :op1 "New York")', '(n / name :op1 "NewYork")', False),
            ("repeated spaces", '(n / name :op1 "New  York")', '(n / name :op1 "New York")', False),
            ("literal parentheses", '(n / name :op1 "a(b)")', '(n / name :op1 "a (b)")', False),
        ]
        for name, reference, hypothesis, equal in cases:
            with self.subTest(name=name):
                _, matches, (lower, upper), _ = self.local.compare(reference, hypothesis)
                self.assertAlmostEqual(lower, upper)
                self.assertAlmostEqual(lower, matches[0])
                local = float(f1_score(matches))
                global_ = float(RoSE(num_iterations=3).compute_graph_similarity(
                    penman.decode(reference, model=model), penman.decode(hypothesis, model=model)
                ))
                self.assertEqual(local == 1, equal, (name, "local", local))
                self.assertEqual(global_ == 1, equal, (name, "global", global_))

    def test_solver_mapping_back_to_original_ids(self):
        local, _, _, mapping = self.local.compare(
            "(f / fail-01 :mod (u / urgent))", "(x / fail-01 :mod (y / urgent))"
        )
        left, right = local.last_normalization
        original = {(left["origins"][a[3:]], right["origins"][b[3:]])
                    for a, b in mapping if a and b}
        self.assertEqual(original, {("f", "x"), ("u", "y")})

    def test_wl_color_collision_still_distinguishes_incidence(self):
        cycle = "(r / and :op1 (a0 / person :ARG0 (b0 / team) :ARG0 (b1 / team)) :op1 (a1 / person :ARG0 b1 :ARG0 (b2 / team)) :op1 (a2 / person :ARG0 b2 :ARG0 (b3 / team)) :op1 (a3 / person :ARG0 b3 :ARG0 b0))"
        split = "(r / and :op1 (a0 / person :ARG0 (b0 / team) :ARG0 (b1 / team)) :op1 (a1 / person :ARG0 b1 :ARG0 b0) :op1 (a2 / person :ARG0 (b2 / team) :ARG0 (b3 / team)) :op1 (a3 / person :ARG0 b3 :ARG0 b2))"
        _, matches, (lower, upper), _ = self.local.compare(cycle, split)
        self.assertAlmostEqual(lower, upper)
        self.assertLess(float(f1_score(matches)), 1)
        self.assertLess(RoSE().compute_graph_similarity(
            penman.decode(cycle, model=model), penman.decode(split, model=model)
        ), 1)

    def test_aggregation_retains_upstream_weights(self):
        frame = pd.DataFrame([
            dict(index=0, name="test", item=0, metric="ESMATCH++mac", score=0.8),
            dict(index=0, name="test", item=0, metric="RoSE3-99", score=0.4),
        ])
        for alpha in (0, 0.25, 1):
            result = calc_SEMCAT.build_semcat(frame, alpha=alpha, require_both=True)
            score = float(result.loc[result.metric == "SEMCAT", "score"].iloc[0])
            self.assertAlmostEqual(score, (0.8 + (1 - alpha) * 0.4) / (2 - alpha))

    def test_ilp_matches_exhaustive_partial_injection_oracle(self):
        family = ["(x / person)", "(x / team)"]
        for a, b, role in itertools.product(("person", "team"), ("person", "team"), ("ARG0", "ARG1")):
            family.append(f"(a / {a} :{role} (b / {b}))")
        family.extend([
            "(a / person :ARG0 (b / team) :ARG1 (c / team))",
            "(a / person :ARG0 (b / team :ARG1 (c / team)))",
        ])
        pairs = 0
        for reference, hypothesis in itertools.product(family, repeat=2):
            with self.subTest(reference=reference, hypothesis=hypothesis):
                local, matches, (lower, upper), mapping = self.local.compare(reference, hypothesis)
                graphs = [penman.decode(n["penman"], model=model) for n in local.last_normalization]
                left, right = [set(g.triples) | {(g.top, ":root", next(
                    t.target for t in g.instances() if t.source == g.top
                ))} for g in graphs]
                source, target = [sorted(g.variables()) for g in graphs]

                def count(m):
                    # Missing mappings get None, which is never a node or literal.
                    return sum((m.get(s), r, m.get(t) if t in source and r not in
                                (":instance", ":root") else t) in right for s, r, t in left)

                optimum = 0
                for size in range(min(len(source), len(target)) + 1):
                    for selected in itertools.combinations(source, size):
                        for assigned in itertools.permutations(target, size):
                            optimum = max(optimum, count(dict(zip(selected, assigned))))
                actual = {a[3:]: b[3:] for a, b in mapping if a and b}
                self.assertEqual(count(actual), optimum)
                self.assertAlmostEqual(lower, optimum)
                self.assertAlmostEqual(upper, optimum)
                self.assertAlmostEqual(float(f1_score(matches)), 2 * optimum / (len(left) + len(right)))
                pairs += 1
        self.assertEqual(pairs, 144)


if __name__ == "__main__":
    unittest.main()
