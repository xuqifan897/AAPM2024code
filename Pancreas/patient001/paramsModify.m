patientFolder = '/data/qifan/projects/FastDoseWorkplace/Pancreas/Patient001/QihuiRyan';
paramsFile = fullfile(patientFolder, 'params0.mat');
StructureFile = fullfile(patientFolder, 'StructureInfo0.mat');

load(paramsFile, 'params');
params.beamWeight = 1000;

load(StructureFile, 'StructureInfo');
% remove RingStructure
StructureInfo = StructureInfo(1:end-1);
for i = 3:size(StructureInfo, 2)
    StructureInfo(i).OARWeights = 5;
end

paramsTargetFile = fullfile(patientFolder, 'params1.mat');
StructureTargetFile = fullfile(patientFolder, 'StructureInfo1.mat');
save(paramsTargetFile, 'params');
save(StructureTargetFile, 'StructureInfo');