folder = '/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_bench';
resultFile = fullfile(folder, '/result LUNG Info1 params1 beam20.mat');
load(resultFile);
dose = result.dose;
targetFile = fullfile(folder, 'dose.bin');
fileID = fopen(targetFile, 'w');
fwrite(fileID, dose, 'double');