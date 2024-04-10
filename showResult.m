M_file = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_bench/LUNG_M.mat";
load(M_file, 'M');
resultFile = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_bench/result LUNG Info1 params1 beam20.mat";
load(resultFile);

xPolish = result.xPolish;
dose = M * xPolish;
doseFile = '/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_bench/dose.bin';
fileID = fopen(doseFile, 'w');
fwrite(fileID, dose, 'double');
fclose(fileID);