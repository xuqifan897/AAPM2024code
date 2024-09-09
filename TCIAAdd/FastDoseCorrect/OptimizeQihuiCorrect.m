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
thresh = 1e-6;
targetFolder = '/data/qifan/projects/FastDoseWorkplace/TCIAAdd/plansAngleCorrect';
% patientList = {'002', '003', '009', '013', '070', '125', '132', '190'};
patientList = {'132', '190', '125'};
numPatients = length(patientList);

for i = 1:numPatients
    patientName = patientList{i};
    expFolder = fullfile(targetFolder, patientName, 'QihuiRyan');

    % % firstly, get the four matrices
    j = 0;
    matFolder = fullfile(expFolder, ['preprocess', num2str(j)]);
    h5file = fullfile(matFolder, 'Dose_Coefficients.h5');
    maskfile = fullfile(matFolder, 'Dose_Coefficients.mask');
    [M0, dose_data0, masks0] = BuildDoseMatrix(h5file, maskfile, thresh);
    fprintf('Finished loading matrix 0\n');

    j = 1;
    matFolder = fullfile(expFolder, ['preprocess', num2str(j)]);
    h5file = fullfile(matFolder, 'Dose_Coefficients.h5');
    maskfile = fullfile(matFolder, 'Dose_Coefficients.mask');
    [M1, dose_data1, masks1] = BuildDoseMatrix(h5file, maskfile, thresh);
    fprintf('Finished loading matrix 1\n');

    j = 2;
    matFolder = fullfile(expFolder, ['preprocess', num2str(j)]);
    h5file = fullfile(matFolder, 'Dose_Coefficients.h5');
    maskfile = fullfile(matFolder, 'Dose_Coefficients.mask');
    [M2, dose_data2, masks2] = BuildDoseMatrix(h5file, maskfile, thresh);
    fprintf('Finished loading matrix 2\n');

    j = 3;
    matFolder = fullfile(expFolder, ['preprocess', num2str(j)]);
    h5file = fullfile(matFolder, 'Dose_Coefficients.h5');
    maskfile = fullfile(matFolder, 'Dose_Coefficients.mask');
    [M3, dose_data3, masks3] = BuildDoseMatrix(h5file, maskfile, thresh);
    fprintf('Finished loading matrix 3\n');

    M = [M0, M1, M2, M3];
    % clear unused memory
    clearvars M0 M1 M2 M3;
    fprintf('Finished coaleasing the four matrices\n')

    StructNum = 1;
    StructureInfo = fullfile(expFolder, ['StructureInfo', num2str(StructNum), '.mat']);
    load(StructureInfo, 'StructureInfo');
    paramsNum = 1;
    params = fullfile(expFolder, ['params', num2str(paramsNum), '.mat']);
    load(params, 'params');

    DS = 1;
    [A, Weights] = CreateA(M, StructureInfo, DS);
    ATrans = A';

    [Dx, Dy] = CreateDxDyFMO(params.BeamletLog0);
    D = [Dx; Dy];

    seed = 2;
    rng(seed);
    tic
    [xFista, costsFista, activeBeams, activeNorms, topN] = ...
        BOO_IMRT_L2OneHalf_cpu_QL(A, ATrans, D, Weights, params);
    timeBeamSelect = toc;
    BOOresult = struct('patientName', patientName, ...
        'params', params, 'StructureInfo', StructureInfo, 'xFista', xFista, ...
        'activeBeams', activeBeams, 'activeNorms', activeNorms, ...
        'costsFista', costsFista, 'timeBeamSelect', timeBeamSelect);
    save(fullfile(expFolder, ['BOOresult_S', num2str(StructNum), '_P', num2str(paramsNum), '.mat']));

    % Show selected beams
    finalBeams = activeBeams;
    finalBeamsVarianIEC = params.beamVarianIEC(finalBeams, :);
    gantryVarianIEC = finalBeamsVarianIEC(:, 1);
    couchVarianIEC = finalBeamsVarianIEC(:, 2);

    PTV = StructureInfo(1).Mask;
    BODY = StructureInfo(2).Mask;
    draw_beammask_QL(params.beamfpangles(finalBeams,:),BODY,PTV);
    figure = fullfile(expFolder, ['BOOresult_S', num2str(StructNum), '_P', num2str(paramsNum) '.png']);
    saveas(gcf, figure);
    clf;

    % Polish step
    paramsPolish = params;
    paramsPolish.maxIter = 500;
    tic
    [xPolish,costsDF_polish, costs_polish] = polish_BOO_IMRT_cpu(finalBeams,A,D,Weights,paramsPolish);
    timePolish = toc;

    dose = M * xPolish;
    dose = reshape(dose, size(PTV));
    polishResult = struct('patientName', patientName, 'dose', dose, 'finalBeams', finalBeams, ...
        'xPolish', xPolish, 'timePolish', timePolish, 'costsDF_polish', costsDF_polish, ...
        'gantryVarianIEC', gantryVarianIEC, 'couchVarianIEC', couchVarianIEC);
    save(fullfile(expFolder, ['PolishResult_S', num2str(StructNum), '_P', num2str(paramsNum), '.mat']));

    finalBeams = finalBeams';
    selected_angles = struct('beamId',finalBeams,'gantryVarianIEC',gantryVarianIEC,'couchVarianIEC',couchVarianIEC);
    T = struct2table(selected_angles);
    filename = fullfile(expFolder,['selected_angles_S', num2str(StructNum), '_P', num2str(paramsNum), '.csv']);
    writetable(T,filename)
end