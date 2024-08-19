resultFolder = '/data/qifan/projects/FastDoseWorkplace/Pancreas/plansAngleCorrect';
numPatients = 5;
for i = 1:numPatients
    patientName = ['Patient00', num2str(i)];
    resultFile = fullfile(resultFolder, patientName, 'QihuiRyan', 'BOOresult.mat');
    load(resultFile, 'BOOresult')
    time = BOOresult.timeBeamSelect
end