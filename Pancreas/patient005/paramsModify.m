patientFolder = '/data/qifan/projects/FastDoseWorkplace/Pancreas/Patient005/QihuiRyan';
paramsFile = fullfile(patientFolder, 'params0.mat');
StructureFile = fullfile(patientFolder, 'StructureInfo0.mat');

load(paramsFile, 'params');
params.beamWeight = 2000;

load(StructureFile, 'StructureInfo');
for i = 3:size(StructureInfo, 2)
    StructureInfo(i).OARWeights = 5;
end
% Modify the weights
names = {'stomach', 'liver', 'spinal_cord', 'RingStructure'};
weights = [2, 2, 1, 1];
for i = 1:size(StructureInfo, 2)
    current_name = StructureInfo(i).Name;
    for j = 1:size(names, 2)
        if strcmp(current_name, names{j})
            StructureInfo(i).OARWeights = weights(j);
            break
        end
    end
end

paramsTargetFile = fullfile(patientFolder, 'params1.mat');
StructureTargetFile = fullfile(patientFolder, 'StructureInfo1.mat');
save(paramsTargetFile, 'params');
save(StructureTargetFile, 'StructureInfo');