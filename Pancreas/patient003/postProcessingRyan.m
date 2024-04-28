patientFolder = '/data/qifan/projects/FastDoseWorkplace/Pancreas/Patient003';
optFolder = fullfile(patientFolder, 'QihuiRyan');
polishResultFile = fullfile(optFolder, 'PolishResultBW2000.mat');
load(polishResultFile, 'PolishResult');
doseArray = PolishResult.dose;
doseArary = single(doseArray);
doseFile = fullfile(optFolder, 'binaryDose2000.bin');
fileID = fopen(doseFile, 'w');
fwrite(fileID, doseArary, 'single');
fclose(fileID);