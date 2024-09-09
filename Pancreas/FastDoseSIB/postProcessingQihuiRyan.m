targetFolder = '/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB';
numPatients = 5;

for i = 1:numPatients
    patientName = ['Patient00', num2str(i)];
    expFolder = fullfile(targetFolder, patientName, 'QihuiRyan');
    polishResultFile = fullfile(expFolder, 'PolishResult.mat');
    load(polishResultFile, 'polishResult');
    doseArray = polishResult.dose;

    % transpose it back to the original order
    doseArray_result = permute(doseArray, [2, 1, 3]);
    doseArray_result = single(doseArray_result);
    doseFile = fullfile(expFolder, 'doseQihuiRyan.bin');
    fileID = fopen(doseFile, 'w');
    fwrite(fileID, doseArray_result, 'single');
    fclose(fileID);
end