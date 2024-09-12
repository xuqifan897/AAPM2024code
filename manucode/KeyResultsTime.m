% resultFolder = '/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB';
% numPatients = 5;
% content = cell(5, 1);
% for i = 1:numPatients
%     patientName = ['Patient00', num2str(i)];
%     resultFile = fullfile(resultFolder, patientName, 'QihuiRyan', 'BOOresult.mat');
%     load(resultFile, 'BOOresult')
%     time = BOOresult.timeBeamSelect;
%     content{i} = num2str(time);
% end
% content = strjoin(content, '\n');
% contentFile = './BOOTimePancreasSIBBaseline.txt';
% fileID = fopen(contentFile, 'w');
% fprintf(fileID, '%s', content);
% fclose(fileID);


resultFolder = '/data/qifan/projects/FastDoseWorkplace/TCIAAdd/plansAngleCorrect';
patients = {'002', '003', '009', '013', '070', '125', '132', '190'};
content = cell(length(patients), 1);
for i = 1:length(patients)
    patientName = patients{i};
    resultFile = fullfile(resultFolder, patientName, 'QihuiRyan', 'BOOresult_S1_P1.mat');
    load(resultFile, 'BOOresult');
    time = BOOresult.timeBeamSelect;
    content{i} = num2str(time);
end
content = strjoin(content, '\n');
contentFile = './BOOTimeTCIABaseline.txt';
fileID = fopen(contentFile, 'w');
fprintf(fileID, '%s', content);
fclose(fileID);