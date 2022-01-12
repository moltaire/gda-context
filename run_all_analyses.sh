#!usr/bin/bash
echo 
echo "Performing all analyses for project"
echo "'Gaze-dependent evidence accumulation predicts multi-alternative risky choice behaviour'"
echo "Contact: felixmolter@gmail.com"
echo 

cd src

# Preprocessing of raw data
echo
echo "####################"
echo "# 0. Preprocessing #"
echo "####################"
echo
python 0-0_preprocessing.py --verbose
python 0-1_process_dwells-fixations.py
python 0-2_process_transitions.py

# Behavioral
echo
echo "###########################"
echo "# 1. Behavioural analyses #"
echo "###########################"
echo
python 1-1_behaviour_descriptives.py
python 1-2_behaviour_context-effects.py

# Gaze
echo
echo "####################"
echo "# 2. Gaze Analyses #"
echo "####################"
echo
python 2-1_gaze_descriptives.py
python 2-2_gaze_confirmatory.py

# Behavioural Modeling
echo
echo "###########################"
echo "# 3. Behavioural Modeling #"
echo "###########################"
echo

echo "\t/!\ Note, that model estimation estimation can take a long, long time. In particular, the switchboard analysis contains estimation of 192 model variants per participant. It took multiple weeks to finish on a machine with 24 cores. By default, model estimation results are *not* overwritten, but existing results are loaded. If you want to re-run the model-fitting, please delete existing results in '../results/3-behavioural_modeling/[estimates, predictions]' and '../results/4-switchboard/[estimates, predictions]'."
echo "Note that you can set the number of CPU cores (threads) to use the `ncores` command line argument."
echo
echo "Running model estimation..."
python 3-1_behavioural-modeling_fitting.py --label de1 --optmethod differential_evolution --nruns 1 --nsim 50 --ncores 24 --seed 1 --verbose 2
python 3-2_behavioural-modeling_analyses.py

# Switchboard
echo
echo "##################"
echo "# 4. Switchboard #"
echo "##################"
echo

echo "Running switchboard estimation..."
python 4-1_switchboard_fitting.py --label de1 --optmethod differential_evolution --nruns 1 --ncores 24 --seed 1 --verbose 2
python 4-2_switchboard_analyses.py

# Figures
echo
echo "###########"
echo "# Figures #"
echo "###########"
echo
python 2-F_context-effects_figure.py
python 3-F_behavioural-modeling_figures.py
python 4-F_switchboard_figures.py
python 5-F_hybrid-model_figure.py

# Supplement
echo
echo "#########################"
echo "# Supplemental analyses #"
echo "#########################"
echo

echo "Running dwell time regressions..."
python Supplement_run-dwell-regression.py --verbose 2 --mixed  --seed 1 --label fulltrial
python Supplement_run-dwell-regression.py --verbose 2 --mixed  --timebinned --seed 2 --label timebinned

echo "Running analysis of transition patterns"
python Supplement_transitions_stats.py

echo "Running analyses testing simple choice rule"
python Supplement_choicerule-analysis.py

echo "Supplemental figures..."
python Supplement_gaze-predictors_figure.py
python Supplement_dwell-regression-weights_figure.py
python Supplement_GLA-estimates_figure.py
python Supplement_model-predicted-probabilities.py


cd ..