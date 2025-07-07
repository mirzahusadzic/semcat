# SEMCAT: Semantic Evaluation Metric Conforming to AMR Theory
This is a repository for the paper [SEMCAT Metric](TBA).

## Howto: execute this metric

There's several ways to execute RoSE metric.

### (1) Using huggingface's evaluate library
TBA

### (2) Using PyPI package
TBA

### (3) Downloading this repository

Execute `python main.py -r [reference files] -p [hypothesis files] -n [N] -t [Tau]` for Rose N-Tau.

For example, let us assume that we want to compute the similarity between `ref.amr` and `hyp.amr`.
As we recommend you to report both RoSE2-75 and RoSE5-99 in your experiment, you need to run the following two scripts:
```bash
python rose.py -r ref.amr -p hyp.amr -n 2 -t 0.75
python rose.py -r ref.amr -p hyp.amr -n 5 -t 0.99
```

## What is `main.py`?

The main script file for comparing two list of AMR graphs, stored in PENMAN notation.

```text
usage: main.py [-h] [--reference REFERENCE [REFERENCE ...]]
               [--predicted PREDICTED [PREDICTED ...]]
               [--num-iterations NUM_ITERATIONS] [--rounding ROUNDING]
               [--similarity-threshold-tau SIMILARITY_THRESHOLD_TAU]
               [--output-txt OUTPUT_TXT]
               [--verbose {CRITICAL,ERROR,WARNING,INFO,DEBUG}]

optional arguments:
  -h, --help            show this help message and exit
  --reference REFERENCE [REFERENCE ...], -ref REFERENCE [REFERENCE ...], -r REFERENCE [REFERENCE ...]
                        List of reference AMR annotation files
  --predicted PREDICTED [PREDICTED ...], --generated PREDICTED [PREDICTED ...], -pred PREDICTED [PREDICTED ...], -gen PREDICTED [PREDICTED ...], -p PREDICTED [PREDICTED ...], -g PREDICTED [PREDICTED ...]
                        List of generated AMR annotation files
  --num-iterations NUM_ITERATIONS, --iter NUM_ITERATIONS, -iter NUM_ITERATIONS, -n NUM_ITERATIONS
                        Maximum number of iterations for WL algorithm (Default: 5, Recommend: 5 and 2)
  --similarity-threshold-tau SIMILARITY_THRESHOLD_TAU, --threshold SIMILARITY_THRESHOLD_TAU, -tau SIMILARITY_THRESHOLD_TAU, -t SIMILARITY_THRESHOLD_TAU
                        Minimum threshold of marking two nodes as compatible (Default: 0.99, Recommend: 0.75 and 0.99)
  --rounding ROUNDING, --round ROUNDING, -round ROUNDING
                        Number of precision digits under the decimal point
  --output-txt OUTPUT_TXT, --output OUTPUT_TXT, -out OUTPUT_TXT, -o OUTPUT_TXT
                        Path to store per-item scores. If not exists, this module just print the average of them.
  --verbose {CRITICAL,ERROR,WARNING,INFO,DEBUG}, -v {CRITICAL,ERROR,WARNING,INFO,DEBUG}
                        Log Level when computing RoSE. If not specified, RoSE will run in silent mode.
```


## What is `stdin.py`?

The main script file for comparing two list of AMR graphs from the standard input.
This will use `RoSE5-99` (N=5, Tau=0.99). (SEMCAT's WL kernel-its past name)
To use this, just type `python stdin.py` and follow the instruction.


## File structure

- `/main.py` Python script for executing WL hashing for stored files
- `/stdin.py` Python script for running WL hashing on the fly, using standard input.
- `/requirements.txt` Text file that specifies required libraries for Python.
- `/rose` Python package for WL hashing part
    - `/rose/scorer.py` Definition of WL hashing
    - `/rose/__init__.py` Definition of calling WL hashing (for main.py and stdin.py)

## Citation
TBA
