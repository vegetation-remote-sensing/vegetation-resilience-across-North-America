%% Example usage script for resilience trend analysis.
% This script demonstrates how to use the standardized functions
% for analyzing vegetation resilience


%% Configuration
config_trend = struct();

% === Input directories ===
% Resilience maps (TAC = AR1 coefficient)
config_trend.input_dir = './input/AR1/';

% === Output directory ===
config_trend.output_dir = './output/';

% Spatial reference (mask value = 0)
config_trend.reference_file = './input/land_mask.tif';

% === Data identifiers ===
config_trend.vegetation_index = 'kNDVI';
config_trend.resilience_indicator = 'AR1';
config_trend.decomposition_method = 'STL';
config_trend.temporal_window = 48;

% === Analysis parameters ===
config_trend.significance_level = 0.05;

%% Run trend analysis
tic;
Sen_MK_TAC(config_trend);
elapsed = toc;

fprintf('\n=== Analysis Complete ===\n');
fprintf('Trend results: %s\n', config_trend.output_dir);
fprintf('\nTotal processing time: %.1f minutes\n', elapsed/60);






