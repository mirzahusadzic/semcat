# RoSE: RObust Metric for Semantic Evaluation
This is a repository for experiments that we did in the paper [RoSE Metric](TBA).

## Howto: reproduce two experiments

Do the followings (all things happen inside the `./experiment` directory):

1. Install RoSE using `pip install [TBA]`.

2. Install other requirements using `pip install -r requirements.txt`

3. Download AMR2.0 and AMR3.0 dataset from LDC.

4. Concatenate all test files by running `python concat_dataset.py -i [AMR test split files] -o [output path]`.
    For example, if your AMR 3.0 dataset path is `./AMR3` and you want to store the result in `resources/amr-annotations/amr3.0-test.txt`, 
    run `python create_robustness_data.py -r ./AMR3/data/amrs/split/test/* -o resources/amr-annotations/amr3.0-test.txt`.

    After this step, you should have two files `resources/amr-annotations/amr2.0-test.txt` and `resources/amr-annotations/amr3.0-test.txt`.

5. Create robustness data first. Run `PYTHONHASHSEED=1 python create_robustness_data.py -r [AMR 3.0 Train Split files] -o [output path]`
    For example, if your AMR 3.0 dataset path is `./AMR3` and you want to store the result in `resources/amr-robustness` (default path), 
    run `PYTHONHASHSEED=1 python create_robustness_data.py -r ./AMR3/data/amrs/split/training/* -o resources/amr-robustness`.

    After this step, you should have 14 files in `resources/amr-robustness`.

6. Download other metrics by running `./prepare_other_metrics.sh`

7. Prepare state-of-the-art parser results on AMR2.0 and AMR3.0, and store it under `resources/amr-baselines`
    Our code expect that `amr-baselines` directory has the following structure.
    - `...amr-baselines/amr2.0` Results on AMR2.0 testset
      - `...amr-baselines/amr2.0/[MODEL]/[RESULT].txt` The total result of the specified model on the AMR 2.0 testset.
      - (for example, `...amr-baselines/amr2.0/AMRBART/AMRBART_2.0.txt`)
    - `...amr-baselines/amr3.0` Results on AMR3.0 testset
      - `...amr-baselines/amr3.0/[MODEL]/[RESULT].txt` The total result of the specified model on the AMR 3.0 testset.
      - (for example, `...amr-baselines/amr3.0/AMRBART/AMRBART_3.0.txt`)

8. Run the experiment by running `python run_all_metric.py -o [OUTPUT] -c [NUM CPUS for parallel exec]`
    If you're using other directory for storing the result of 2, 3, and 5, then use `-a`, `-r`, and `-b` option to specify the directories.

    To get a clear view without standard error output, use `python run_all_metric.py -o [OUTPUT] -c [NUM CPUS for parallel exec] 2>errors.log`


## Citation
TBA