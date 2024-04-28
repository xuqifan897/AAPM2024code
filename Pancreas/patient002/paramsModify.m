patientFolder = '/data/qifan/projects/FastDoseWorkplace/Pancreas/Patient002/QihuiRyan';
paramsFile = fullfile(patientFolder, 'params0.mat');
StructureFile = fullfile(patientFolder, 'StructureInfo0.mat');

load(paramsFile, 'params');
params.beamWeight = 2000;

load(StructureFile, 'StructureInfo');
for i = 3:size(StructureInfo, 2)
    StructureInfo(i).OARWeights = 5;
end
% Modify the weight of RingStructure
StructureInfo(end).OARWeights = 0.5;

paramsTargetFile = fullfile(patientFolder, 'params1.mat');
StructureTargetFile = fullfile(patientFolder, 'StructureInfo1.mat');
save(paramsTargetFile, 'params');
save(StructureTargetFile, 'StructureInfo');