restoredefaultpath;
packagePath = '/data/qifan/projects/BeamOpt/BOO';
BOO_QL_path = fullfile(packagePath, 'BOO_QL');
CERR2016_path = fullfile(packagePath, 'CERR2016');
CERRaddins_path = fullfile(packagePath, 'CERRaddins');
utilities_path = fullfile(packagePath, 'utilities');
beamlogs_path = fullfile(packagePath, 'beamlogs');
addpath(genpath(BOO_QL_path), '-end');
addpath(genpath(CERR2016_path), '-end');
addpath(genpath(CERRaddins_path), '-end');
addpath(genpath(utilities_path), '-end');
addpath(genpath(beamlogs_path), '-end');

clear; clc;
sourceFolder = '/data/qifan/projects/FastDoseWorkplace/Pancreas';
targetFolder = fullfile(sourceFolder, 'plansAngleCorrect');
numPatients = 5;
thresh = 1e-6;
PrescriptionDose = 20;

for i = 2: numPatients
% for i = 1:1
    patientName = ['Patient00', num2str(i)];
    expFolder = fullfile(targetFolder, patientName, 'QihuiRyan');
    h5file = fullfile(expFolder, 'Dose_Coefficients.h5');
    maskfile = fullfile(expFolder, 'Dose_Coefficients.mask');
    [M, dose_data, masks] = BuildDoseMatrix(h5file, maskfile, thresh);
    fullMatFile = fullfile(expFolder, [patientName, '_M.mat']);
    save(fullMatFile, 'M', 'dose_data', 'masks', '-v7.3');

    % save parameter files
    [StructureInfo, params] = InitIMRTparams_NoRing(M, dose_data, masks, PrescriptionDose);
    paramsNum = 0;
    paramsFile = fullfile(expFolder, ['params', num2str(paramsNum), '.mat']);
    save(paramsFile, 'params');
    infoNum = 0;
    infoFile = fullfile(expFolder, ['StructureInfo', num2str(infoNum), '.mat']);
    save(infoFile, 'StructureInfo');

    % draw beams
    PTV = StructureInfo(1).Mask;
    BODY = StructureInfo(2).Mask;
    draw_beammask_QL(params.beamfpangles, BODY, PTV);
    beamFigureFile = fullfile(expFolder, 'beamsView.png');
    saveas(gcf, beamFigureFile);
end