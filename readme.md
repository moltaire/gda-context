# Gaze-dependent evidence accumulation predicts multi-alternative risky choice behaviour

This repository contains all raw data, preprocessing, analysis and visualization code used in the paper:

- tbd

All analyses<sup>[*](#bms)</sup> are written in Python.
Dependencies and package versions used are listed in the `environment.yml` file, which you can use to reproduce the computing environment (e.g., using the Anaconda Python distribution).

All Python analyses can be run in sequence by calling the `run_all_analyses.sh` script. This script calls the individual analysis scripts in the `src` folder.

Note that the project involves fitting of many (~200) behavioural model variants, which can take days to weeks, depending on the machine. **By default, model fitting results are *not* overwritten, but read from the repository.** If you plan to reproduce the model-fitting, you can adapt the `overwrite` and `ncores` command line arguments for the `3-1_behavioural-modeling_fitting.py` and `4-1_switchboard_fitting.py` scripts.

Statistical analyses use sampling-based Bayesian estimation methods and can yield slightly different results each run, even if random seeds are set, due to powers that I don't understand.

### Contact

Questions or comments should be addressed at felixmolter@gmail.com.

---

<a name="bms">*</a> The BMS script `3-3_run_bms.m` must be run in MATLAB manually. Alternatively, the code includes a basic Python implementation of a Bayesian Model Selection procedure in `src/analysis/bms.py` that can be used to compute basic exceedance probabilities in Python.
