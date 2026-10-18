# SEMCAT runtime

A maintained, installable fork of [snucclab/ssharp](https://github.com/snucclab/ssharp).
It packages SEMCAT's local Smatch++ and global RoSE scorers, with focused fixes
for AMR normalization, original variable IDs and lossless text handling.

Based on upstream commit
[`5c5d4bd`](https://github.com/snucclab/ssharp/commit/5c5d4bd5a15cf1e1a0db05b16610ee39a90f7c2d).
The upstream optimizer and SEMCAT weighting are preserved; normalization fixes
can change scores. This is a maintained variant, not a claim to reproduce the
original paper's results. See [changes and scoring details](CHANGELOG.md).

## Use

Requires Python 3.11+. From a checkout of this fork:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install .
python examples/compare.py
```

[The example](examples/compare.py) runs both scorers and the upstream aggregator.
It explicitly uses HiGHS for ILP and preserves literal case and quotes. Existing
imports remain available: `smatchpp`, `rose` and `calc_SEMCAT`.

Install from the **repository root**. The nested upstream package definitions and
experimental instructions are retained for reference; they do not install the
complete patched runtime. Do not install the standalone `smatchpp` or
`semantic-rose` distributions alongside this package in the same environment:
they share import names.

For repeatable results, pin this fork to a full commit and lock its dependencies.
Keep input validation, timeouts and workload limits in the consuming application;
the global scorer can enumerate many alignments on highly symmetric graphs.

## Develop

```sh
python -m pip install build
python -m build
python -m unittest discover -s tests -v
```

CI builds a wheel from the source distribution, installs it in a clean environment,
and tests it outside the checkout on Linux and macOS. Tests include semantic
regressions and an independent exhaustive alignment oracle over 144 graph pairs.

## License

The runtime combines **GPLv3 Smatch++** and **MIT RoSE**. Original notices are
preserved and included in distributions. See [licensing and scope](LICENSING.md).
