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

clearvars -except idx;
clc;
sourceFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB";
numPatients = 5;
thresh = 1e-6;

idx = 2;
    patientName = ['Patient00', num2str(idx)];
    expFolder = fullfile(sourceFolder, patientName, 'QihuiRyan');

    % load StructureInfo
    StructureInfoFile = fullfile(expFolder, 'StructureInfo_else5_PTV50.mat');
    load(StructureInfoFile, 'StructureInfo');

    % load params
    paramsFile = fullfile(expFolder, 'params_else5.mat');
    load(paramsFile, 'params');

    % load target dose array
    targetDoseFile = fullfile(sourceFolder, patientName, 'doseNorm.bin');
    fileID = fopen(targetDoseFile, 'r');
    targetDose = fread(fileID, 'float32');
    fclose(fileID);
    shape = size(StructureInfo(1).Mask);
    shape_flip = [shape(2), shape(1), shape(3)];
    targetDose = reshape(targetDose, shape_flip);
    targetDose = permute(targetDose, [2, 1, 3]);

    % load matrix M
    h5file = fullfile(expFolder, 'Dose_Coefficients.h5');
    maskfile = fullfile(expFolder, 'Dose_Coefficients.mask');
    [M, dose_data, masks] = BuildDoseMatrix(h5file, maskfile, thresh);

    % construct matrix A and weights
    DS = 1;
    [A, Weights] = CreateA_SIB(M, StructureInfo,targetDose, DS);
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
    save(fullfile(expFolder, 'BOOresult_else5_PTV50.mat'), 'BOOresult');

    % Show selected beams
    finalBeams = activeBeams;
    finalBeamsVarianIEC = params.beamVarianIEC(finalBeams,:);
    gantryVarianIEC = finalBeamsVarianIEC(:,1);
    couchVarianIEC = finalBeamsVarianIEC(:,2);

    % Polish step
    paramsPolish = params;
    paramsPolish.maxIter = 500;
    tic
    [xPolish,costsDF_polish, costs_polish] = polish_BOO_IMRT_cpu(finalBeams,A,D,Weights,paramsPolish);
    timePolish = toc;

    dose = M * xPolish;
    dose = reshape(dose, shape);
    polishResult = struct('patientName', patientName, 'dose', dose, 'finalBeams', finalBeams, ...
        'xPolish', xPolish, 'timePolish', timePolish, 'costsDF_polish', costsDF_polish, ...
        'gantryVarianIEC', gantryVarianIEC, 'couchVarianIEC', couchVarianIEC);
    save(fullfile(expFolder, 'PolishResult_else5_PTV50.mat'), 'polishResult');

    finalBeams = finalBeams';
    selected_angles = struct('beamId',finalBeams,'gantryVarianIEC',gantryVarianIEC,'couchVarianIEC',couchVarianIEC);
    T = struct2table(selected_angles);
    filename = fullfile(expFolder,['selected_angles_else5_PTV50','.csv']);
    writetable(T,filename)


function [A, Weights] = CreateA_SIB(M, StructureInfo, targetDose, DS)
    tic
    targetDose = targetDose(:);
    modFactor = DS;
    CalcA = 1;

    PTVind = 1;
    assert(strcmp('PTV', StructureInfo(PTVind).Name));
    PTV = StructureInfo(PTVind).Mask;
    PTVdilate = imdilate(PTV, ones(3, 3, 3));

    StructureName = {StructureInfo.Name};
    IsPTV = contains(StructureName, 'PTV', 'IgnoreCase', true);
    IsOAR = ~IsPTV;

    numVOI = length(StructureInfo);
    A_mats = cell(numVOI,1);
    maxDose_vecs = cell(numVOI,1);
    maxWeights_vecs = cell(numVOI,1);
    minDoseTarget_vecs = cell(numVOI,1);
    minDoseTargetWeights_vecs = cell(numVOI,1);
    OARWeights_vecs = cell(numVOI,1);
    numVoxs = zeros(numVOI,1);

    IsVOI = true(1, numVOI);
    for idx = 1:numVOI
        IsVOI(idx) = IsZeroOrNaN([StructureInfo(idx).maxWeights...
            StructureInfo(idx).minDoseTargetWeights StructureInfo(idx).OARWeights]);
    end

    PTVsind = find(IsPTV&IsVOI);
    OARsind = find(IsOAR&IsVOI);
    numPTV = length(PTVsind);
    PTV0mask = false(size(PTV));
    PTV0mask_dilate = imdilate(PTV0mask,ones(6,6,6));

    for idx = 1:numPTV
        StructureInfo(PTVsind(idx)).Mask(PTV0mask_dilate ==1) = 0;
        if idx<numPTV
            PTV0mask = (PTV0mask | StructureInfo(PTVsind(idx)).Mask);
            PTV0mask_dilate = imdilate(PTV0mask,ones(3,3,3));
        end
    end

    for idx = 1:numVOI
        if(isempty(find(idx==find(IsVOI), 1)))
            continue
        end

        if(find(idx==OARsind))
            StructureInfo(idx).Mask(PTVdilate==1) = 0;
        end
        fnd = find(StructureInfo(idx).Mask);
        [I, J, K] = ind2sub(size(PTV),fnd);
        fnd2 = (mod(I,modFactor) == 0 & mod(J,modFactor) == 0 & mod(K,modFactor) == 0);
        fnd = fnd(fnd2);
        
        if(CalcA==1)
            A_mats{idx} = M(fnd,:);
        end

        numVox = length(fnd);
        numVoxs(idx) = numVox;
        maxWeights_vecs{idx} = StructureInfo(idx).maxWeights + zeros(numVox,1);
        minDoseTargetWeights_vecs{idx} = StructureInfo(idx).minDoseTargetWeights + zeros(numVox,1);
        OARWeights_vecs{idx} = StructureInfo(idx).OARWeights + zeros(numVox,1);

        if(find(idx==OARsind))
            maxDose_vecs{idx} = StructureInfo(idx).maxDose + zeros(numVox,1);
            minDoseTarget_vecs{idx} = StructureInfo(idx).minDoseTarget + zeros(numVox,1);
        elseif(find(idx==PTVsind))
            % If PTV, copy the target dose
            maxDose_vecs{idx} = targetDose(fnd);
            minDoseTarget_vecs{idx} = maxDose_vecs{idx};
        end
    end

    Weights.maxDose = cat(1,maxDose_vecs{[PTVsind,OARsind]});
    Weights.maxWeightsLong = cat(1,maxWeights_vecs{[PTVsind,OARsind]});
    Weights.minDoseTarget = cat(1,minDoseTarget_vecs{PTVsind});
    Weights.minDoseTargetWeights = cat(1,minDoseTargetWeights_vecs{PTVsind});
    Weights.OARWeightsLong = cat(1,OARWeights_vecs{OARsind});

    A_ptv = cat(1,A_mats{PTVsind});
    A_noPtv = cat(1,A_mats{OARsind});
    A = [A_ptv;A_noPtv];
    toc
end

function flag = IsZeroOrNaN(x)
    flag = ~all(isnan(x)|x==0);
end