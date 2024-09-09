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

% % The parameter i is supposed to be provided by the user
% if ~exist('i', 'var')
%     fprintf('The user is supposed to provide the variable i\n');
%     return;
% end
for i = 1:length(patientList)
    patient = patientList{i};

    j = 0;
    expFolder = fullfile(targetFolder, patient, 'QihuiRyan', ['preprocess', num2str(j)]);
    h5file = fullfile(expFolder, 'Dose_Coefficients.h5');
    maskfile = fullfile(expFolder, 'Dose_Coefficients.mask');
    [M0, dose_data0, masks0] = BuildDoseMatrix(h5file, maskfile, thresh);
    [StructureInfo0, params0] = InitIMRTparams_NoRing(M0, dose_data0, masks0, PrescriptionDose);

    j = 1;
    expFolder = fullfile(targetFolder, patient, 'QihuiRyan', ['preprocess', num2str(j)]);
    h5file = fullfile(expFolder, 'Dose_Coefficients.h5');
    maskfile = fullfile(expFolder, 'Dose_Coefficients.mask');
    [M1, dose_data1, masks1] = BuildDoseMatrix(h5file, maskfile, thresh);
    [StructureInfo1, params1] = InitIMRTparams_NoRing(M1, dose_data1, masks1, PrescriptionDose);

    j = 2;
    expFolder = fullfile(targetFolder, patient, 'QihuiRyan', ['preprocess', num2str(j)]);
    h5file = fullfile(expFolder, 'Dose_Coefficients.h5');
    maskfile = fullfile(expFolder, 'Dose_Coefficients.mask');
    [M2, dose_data2, masks2] = BuildDoseMatrix(h5file, maskfile, thresh);
    [StructureInfo2, params2] = InitIMRTparams_NoRing(M2, dose_data2, masks2, PrescriptionDose);

    j = 3;
    expFolder = fullfile(targetFolder, patient, 'QihuiRyan', ['preprocess', num2str(j)]);
    h5file = fullfile(expFolder, 'Dose_Coefficients.h5');
    maskfile = fullfile(expFolder, 'Dose_Coefficients.mask');
    [M3, dose_data3, masks3] = BuildDoseMatrix(h5file, maskfile, thresh);
    [StructureInfo3, params3] = InitIMRTparams_NoRing(M3, dose_data3, masks3, PrescriptionDose);

    M = [M0, M1, M2, M3];

    % coalease the paramsAttributes
    params = struct('beamWeight', 300, 'gamma', 1, 'eta', 0.1, 'numBeamsWeWant', 20, ...
        'stepSize', 1e-5, 'maxIter', 8000, 'showTrigger', 10, ...
        'ChangeWeightsTrigger', 1000, 'beamWeightsInit', [], ...
        'beamSizes', [], ...
        'BeamletLog0', [], 'beamVarianIEC', [], 'beamfpangles', []);

    % generate beamWeightsInit
    PTV = masks0{1}.mask;
    beamletnumTable = [params0.beamSizes; params1.beamSizes; params2.beamSizes; params3.beamSizes];
    beamWeightsInit = findBeamWeights(M, beamletnumTable, PTV);
    beamWeightsInit = beamWeightsInit / max(beamWeightsInit(:));
    beamWeightsInit(beamWeightsInit<0.1) = 0.1;

    params.beamWeightsInit = beamWeightsInit;
    params.beamSizes = beamletnumTable;
    params.BeamletLog0 = cat(3, params0.BeamletLog0, params1.BeamletLog0, params2.BeamletLog0, params3.BeamletLog0);
    params.beamVarianIEC = [params0.beamVarianIEC; params1.beamVarianIEC; params2.beamVarianIEC; params3.beamVarianIEC];
    params.beamfpangles = [params0.beamfpangles; params1.beamfpangles; params2.beamfpangles; params3.beamfpangles];
    StructureInfo = StructureInfo0;

    QihuiRyanFolder = fullfile(targetFolder, patient, 'QihuiRyan');
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