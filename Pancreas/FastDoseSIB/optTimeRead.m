sourceFolder = '/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB';
targetFile = '/data/qifan/projects/AAPM2024/manucode/BOOTimePancreasSIBBaseline.txt';
numPatients = 5;
text = '';

for i = 1:numPatients
    patientName = ['Patient00', num2str(i)];
    resultFile = fullfile(sourceFolder, patientName, 'QihuiRyan', 'BOOresult_else5.mat');
    if i == 2
        resultFile = fullfile(sourceFolder, patientName, 'QihuiRyan', 'BOOresult_else5_PTV50.mat');
    end
    load(resultFile, 'BOOresult');
    % timeData(i) = BOOresult.timeBeamSelect;
    if strcmp(text, '')
        text = [num2str(BOOresult.timeBeamSelect)];
    else
        text = [text, '\n', num2str(BOOresult.timeBeamSelect)];
    end
end
fileID = fopen(targetFile, 'w');
fprintf(fileID, text);
fprintf([text, '\n'])