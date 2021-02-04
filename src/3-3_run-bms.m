% Gaze-dependent accumulation in context-dependent risky choice
% This script runs Bayesian Model Selection (Stephan et al., 2009, Rigoux et al., 2014)
% Must be run from src directory so that relative filenames work.
% Author: Felix Molter, felixmolter@gmail.com

% Read BIC file
bic_table = readtable(fullfile("..", "results", "3-behavioural-modeling", "model-comparison_bics.csv"));

% Read model names
model_names = bic_table.Properties.VariableNames;

% Read BIC values (drop subject index)
bics = table2array(bic_table);
bics = bics(:, 2:(numel(model_names)));

% Convert BICs to Stephan et al format
% From:
% BIC = -2 * ln L + K * ln N
% where ln L is the log likelihood of the data, K is the number of
% parameters, and N is the number of data points
% To:
% BIC = ln L - (K / 2) * ln N
bics = -2 * bics;

% Run BMS
[alpha, exp_r, xp, pxp, bor] = spm_BMS(bics);

% Save result as a table
result = table(model_names(2:numel(model_names))', round(alpha', 2), round(exp_r', 2), round(xp', 2), round(pxp', 2), 'VariableNames', {'model', 'alpha', 'exp_r', 'xp', 'pxp'});
writetable(result, fullfile("..", "results", "3-behavioural-modeling", "model-comparison_bms_results.csv"), 'Delimiter', ',');