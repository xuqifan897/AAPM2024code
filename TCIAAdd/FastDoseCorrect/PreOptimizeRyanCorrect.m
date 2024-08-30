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

clearvars -except i; clc;
sourceFolder = '/data/qifan/projects/FastDoseWorkplace/TCIAAdd';
targetFolder = fullfile(sourceFolder, 'plansAngleCorrect');
patientList = {'002', '003', '009', '013', '070', '125', '132', '190'};
thresh = 1e-6;
PrescriptionDose = 20;

% The parameter i is supposed to be provided by the user
if ~exist('i', 'var')
    fprintf('The user is supposed to provide the variable i\n');
    return;
end
stride = 4;
numPatients = length(patientList);
for idx = i : stride: numPatients
    patient = patientList{idx};
    j = 0;
    expFolder = fullfile(targetFolder, patient, 'QihuiRyan', ['preprocess', num2str(j)]);
    h5file = fullfile(expFolder, 'Dose_Coefficients.h5');
    maskfile = fullfile(expFolder, 'Dose_Coefficients.mask');
    [M0, dose_data0, masks0] = BuildDoseMatrix(h5file, maskfile, thresh);

    j = 1;
    expFolder = fullfile(targetFolder, patient, 'QihuiRyan', ['preprocess', num2str(j)]);
    h5file = fullfile(expFolder, 'Dose_Coefficients.h5');
    maskfile = fullfile(expFolder, 'Dose_Coefficients.mask');
    [M1, dose_data1, masks1] = BuildDoseMatrix(h5file, maskfile, thresh);

    j = 2;
    expFolder = fullfile(targetFolder, patient, 'QihuiRyan', ['preprocess', num2str(j)]);
    h5file = fullfile(expFolder, 'Dose_Coefficients.h5');
    maskfile = fullfile(expFolder, 'Dose_Coefficients.mask');
    [M2, dose_data2, masks2] = BuildDoseMatrix(h5file, maskfile, thresh);

    j = 3;
    expFolder = fullfile(targetFolder, patient, 'QihuiRyan', ['preprocess', num2str(j)]);
    h5file = fullfile(expFolder, 'Dose_Coefficients.h5');
    maskfile = fullfile(expFolder, 'Dose_Coefficients.mask');
    [M3, dose_data3, masks3] = BuildDoseMatrix(h5file, maskfile, thresh);

    QihuiRyanFolder = fullfile(targetFolder, patient, 'QihuiRyan');
    M = [M0, M1, M2, M3];
    [StructureInfo, params] = InitIMRTparams_NoRing(M3, dose_data3, masks3, PrescriptionDose);
    paramsNum = 0;
    paramsFile = fullfile(QihuiRyanFolder, ['params', num2str(paramsNum), '.mat']);
    save(paramsFile, 'params');
    infoNum = 0;
    infoFile = fullfile(QihuiRyanFolder, ['StructureInfo', num2str(infoNum), '.mat']);
    save(infoFile, 'StructureInfo');

    % draw beams
    PTV = StructureInfo(1).Mask;
    BODY = StructureInfo(2).Mask;
    draw_beammask_QL(params.beamfpangles, BODY, PTV);
    beamFigureFile = fullfile(QihuiRyanFolder, 'beamsView.png');
    saveas(gcf, beamFigureFile);
end