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
targetFolder = '/data/qifan/projects/FastDoseWorkplace/Pancreas/plansAngleCorrect';
numPatients = 5;

for i = 1:numPatients
    patientName = ['Patient00', num2str(i)];
    expFolder = fullfile(targetFolder, patientName, 'QihuiRyan');

    % load StructureInfo
    StructureInfoFile = fullfile(expFolder, 'StructureInfo1.mat');
    load(StructureInfoFile, 'StructureInfo');
    % load params
    paramsFile = fullfile(expFolder, 'params1.mat');
    load(paramsFile, 'params');
    % load full dose matrix
    matFile = fullfile(expFolder, [patientName, '_M.mat']);
    load(matFile, 'M');

    DS = 1;
    [A, Weights] = CreateA(M, StructureInfo, DS);
    ATrans = A';

    [Dx, Dy] = CreateDxDyFMO(params.BeamletLog0);
    D = [Dx; Dy];

    % beam selection
    seed = 2;
    rng(seed);
    tic
    [xFista, costsFista, activeBeams, activeNorms, topN] = BOO_IMRT_L2OneHalf_cpu_QL(A,ATrans,D,Weights,params);
    timeBeamSelect = toc;
    BOOresult = struct('patientName',patientName,...
        'params',params,'StructureInfo',StructureInfo,'xFista',xFista,...
        'activeBeams',activeBeams,'activeNorms',activeNorms,...
        'costsFista',costsFista,'timeBeamSelect',timeBeamSelect);
    save(fullfile(expFolder, ['BOOresult.mat']), 'BOOresult');

    % Show selected beams
    finalBeams = activeBeams;
    finalBeamsVarianIEC = params.beamVarianIEC(finalBeams,:);
    gantryVarianIEC = finalBeamsVarianIEC(:,1);
    couchVarianIEC = finalBeamsVarianIEC(:,2);

    PTV = StructureInfo(1).Mask;
    BODY = StructureInfo(2).Mask;
    draw_beammask_QL(params.beamfpangles(finalBeams,:),BODY,PTV);
    figure = fullfile(expFolder, 'BOOresult.png');
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
    save(fullfile(expFolder, 'PolishResult.mat'), 'polishResult');

    finalBeams = finalBeams';
    selected_angles = struct('beamId',finalBeams,'gantryVarianIEC',gantryVarianIEC,'couchVarianIEC',couchVarianIEC);
    T = struct2table(selected_angles);
    filename = fullfile(expFolder,['selected_angles','.csv']);
    writetable(T,filename)
end