rootFolder = '/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB';
numPatients = 5;
body = 'BODY';

if false
    for i = 1:numPatients
        patientName = ['Patient00', num2str(i)];
        patientFolder = fullfile(rootFolder, patientName, 'QihuiRyan');
        StructureInfoInput = fullfile(patientFolder, 'StructureInfo0.mat');
        load(StructureInfoInput, 'StructureInfo');
        
        for j = 1:length(StructureInfo)
            if (strcmp(StructureInfo(j).Name, body))
                StructureInfo(j) = [];
                break;
            end
        end

        StructureInfoOutput = fullfile(patientFolder, 'StructureInfo1.mat');
        save(StructureInfoOutput, 'StructureInfo');
    end
end

if true
    for i = 1:numPatients
        patientName = ['Patient00', num2str(i)];
        patientFolder = fullfile(rootFolder, patientName, 'QihuiRyan');
        StructureInfoInput = fullfile(patientFolder, 'StructureInfo1.mat');
        load(StructureInfoInput, 'StructureInfo');
        fprintf([patientName, '\n']);
        for j = 1:length(StructureInfo)
            % fprintf([num2str(StructureInfo(j).maxWeights), '    ']);
            fprintf([num2str(StructureInfo(j).minDoseTargetWeights), '    ']);
        end
        fprintf('\n\n');
    end
end