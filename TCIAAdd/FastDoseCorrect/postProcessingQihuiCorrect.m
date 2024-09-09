targetFolder = '/data/qifan/projects/FastDoseWorkplace/TCIAAdd/plansAngleCorrect';
patientList = {'002', '003', '009', '013', '070', '125', '132', '190'};
numPatients = length(patientList);

for i = 1:numPatients
    patientName = patientList{i};
    expFolder = fullfile(targetFolder, patientName, 'QihuiRyan');
    polishResultFile = fullfile(expFolder, 'PolishResult_S1_P1.mat');
    load(polishResultFile, 'polishResult');
    doseArray = polishResult.dose;

    % transpose it back to the original order
    doseArray_result = permute(doseArray, [2, 1, 3]);
    doseArray_result = single(doseArray_result);
    doseFile = fullfile(expFolder, 'doseRef.bin');
    fileID = fopen(doseFile, 'w');
    fwrite(fileID, doseArray_result, 'single');
    fclose(fileID);
end