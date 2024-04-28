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
optFolder = '/data/qifan/projects/FastDoseWorkplace/Pancreas/Patient003/QihuiRyan';
patientName = 'Patient003';

load(fullfile(optFolder, [patientName, '_M.mat']), 'M');

InfoNum = 1;
load(fullfile(optFolder, ['StructureInfo', num2str(InfoNum), '.mat']), 'StructureInfo');

ParamsNum = 1;
load(fullfile(optFolder, ['params', num2str(ParamsNum), '.mat']), 'params');

% Downsample and prepare matrix
DS = 1;  % DS=1 for no downsampling, DS>1 for downsampling with a factor of DS
[A, Weights] = CreateA(M, StructureInfo, DS);
ATrans = A';

[Dx, Dy] = CreateDxDyFMO(params.BeamletLog0);
D = [Dx; Dy];

% beam selection
seed = 2;
rng(seed);
tic
[xFista,costsFista,activeBeams,activeNorms,topN] = BOO_IMRT_L2OneHalf_cpu_QL(A,ATrans,D,Weights,params);
timeBeamSelect = toc;
BOOresult = struct('patientName',patientName,...
    'params',params,'StructureInfo',StructureInfo,'xFista',xFista,...
    'activeBeams',activeBeams,'activeNorms',activeNorms,...
    'costsFista',costsFista,'timeBeamSelect',timeBeamSelect);
planName = [patientName 'Info' num2str(InfoNum)...
    'Params' num2str(ParamsNum) 'Beam' num2str(nnz(activeBeams))];
save(fullfile(optFolder, ['BOO_', planName, '.mat']), 'BOOresult');

% Show selected beams
finalBeams = activeBeams;
finalBeamsVarianIEC = params.beamVarianIEC(finalBeams,:);
gantryVarianIEC = finalBeamsVarianIEC(:,1);
couchVarianIEC = finalBeamsVarianIEC(:,2);

PTV = StructureInfo(1).Mask;
BODY = StructureInfo(2).Mask;
draw_beammask_QL(params.beamfpangles(finalBeams,:),BODY,PTV);
figureFile = fullfile(optFolder, 'beams_selected_Qihui.png');
saveas(gcf, figureFile);
clf;

% Polish step
paramsPolish = params;
paramsPolish.maxIter = 2000;
tic
[xPolish,costsDF_polish, costs_polish] = polish_BOO_IMRT_cpu(finalBeams,A,D,Weights,paramsPolish);
timePolish = toc;

dose = M * xPolish;
dose = reshape(dose, size(PTV));
dose(BODY==0&PTV==0) = 0;

PolishResult = struct('patientName', patientName, 'dose', dose, 'finalBeams', finalBeams, ...
    'xPolish', xPolish, 'timePolish', timePolish, 'costsDF_polish', costsDF_polish, ...
    'params', params, 'StructureInfo', StructureInfo, ...
    'gantryVarianIEC', gantryVarianIEC, 'couchVarianIEC', couchVarianIEC, ...
    'paramsPolish', paramsPolish, 'planName', planName);
save(fullfile(optFolder, ['Polish_', planName, '.mat']), 'PolishResult');

selected_angles = struct('gantryVarianIEC',gantryVarianIEC,'couchVarianIEC',couchVarianIEC);
T = struct2table(selected_angles);
filename = fullfile(optFolder,['selected_angles_',planName,'.csv']);
writetable(T,filename)