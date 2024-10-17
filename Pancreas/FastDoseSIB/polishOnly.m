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

clear;clc;
sourceFolder = '/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB';
numPatients = 5;
fluenceDim = [20, 20];

for i = 1:1
    patientName = ['Patient00', num2str(i)];
    patientFolder = fullfile(sourceFolder, patientName);

    % load StructureInfo
    StructureInfoFile = fullfile(patientFolder, 'QihuiRyan', 'StructureInfo1.mat');
    load(StructureInfoFile, 'StructureInfo');

    % load params
    paramsFile = fullfile(patientFolder, 'QihuiRyan', 'params1.mat');
    paramsOrg = fullfile(patientFolder, 'FastDose', 'params.txt');
    params = loadParams(paramsFile, paramsOrg);

    phantomDim = fullfile(patientFolder, 'FastDose', 'prep_output', 'dimension.txt');
    phantomDim = readDimension(phantomDim);

    % load target dose array
    targetDoseFile = fullfile(patientFolder, 'doseNorm.bin');
    fileID = fopen(targetDoseFile, 'r');
    targetDose = fread(fileID, 'float32');
    fclose(fileID);
    targetDose = reshape(targetDose, phantomDim);
    targetDose = permute(targetDose, [2, 1, 3]);

    selectedBeams = fullfile(patientFolder, 'QihuiRyan', 'selected_angles_UHPP.csv');
    selectedBeams = getSelectedBeams(selectedBeams);

    matFolder1 = fullfile(patientFolder, 'FastDose', 'doseMat1', 'doseMatFolder');
    matFolder2 = fullfile(patientFolder, 'FastDose', 'doseMat2', 'doseMatFolder');

    BeamletLog1 = BeamletLogGen(matFolder1, fluenceDim);
    BeamletLog2 = BeamletLogGen(matFolder2, fluenceDim);
    BeamletLog = cat(3, BeamletLog1, BeamletLog2);
    [Dx, Dy] = CreateDxDyFMO(BeamletLog);
    D = [Dx; Dy];
    params.BeamletLog0 = BeamletLog;
    params.beamSizes = squeeze(sum(params.BeamletLog0, [1, 2]));
    beamSizes_org = params.beamSizes;

    params.beamWeightsInit = params.beamWeightsInit(selectedBeams);
    params.beamSizes = params.beamSizes(selectedBeams);
    params.BeamletLog0 = params.BeamletLog0(:, :, selectedBeams);
    params.beamVarianIEC = params.beamVarianIEC(selectedBeams, :);
    params.beamfpangles = params.beamfpangles(selectedBeams, :);
    
    mFilter = getMFilter(beamSizes_org, selectedBeams);

    % load matrix M
    M1 = loadM(matFolder1, phantomDim);
    M2 = loadM(matFolder2, phantomDim);
    M_org = [M1, M2];
    M = M_org(:, mFilter);

    % construct matrix A and weights
    DS = 1;
    [A, Weights] = CreateA_SIB(M, StructureInfo, targetDose, DS);
    ATrans = A';
    [Dx, Dy] = CreateDxDyFMO(BeamletLog);
    D = [Dx; Dy];

    % Polish step
    paramsPolish = params;
    paramsPolish.maxIter = 500;
    finalBeams = [1:numel(params.beamSizes)];
    tic
    [xPolish,costsDF_polish, costs_polish] = polish_BOO_IMRT_cpu(finalBeams,A,D,Weights,paramsPolish);
    timePolish = toc;

    dose = M * xPolish;
    dose = reshape(dose, shape);
    gantryVarianIEC = params.beamVarianIEC(:, 1);
    couchVarianIEC = params.beamVarianIEC(:, 2);
    polishResult = struct('patientName', patientName, 'dose', dose, 'finalBeams', finalBeams, ...
        'xPolish', xPolish, 'timePolish', timePolish, 'costsDF_polish', costsDF_polish, ...
        'gantryVarianIEC', gantryVarianIEC, 'couchVarianIEC', couchVarianIEC);
    save(fullfile(patientFolder, 'QihuiRyan', 'PolishResult_UHPP.mat'), 'polishResult');
end

function mFilter = getMFilter(beamSizes, selectedBeams)
    mFilter = [];
    index = 1;
    for i = 1:numel(beamSizes)
        localBeamSize = beamSizes(i);
        if ismember(i, selectedBeams)
            for j = 1:localBeamSize
                mFilter(end+1) = index;
                index = index + 1;
            end
        else
            index = index + localBeamSize;
        end
    end
    mFilter = mFilter';
end

function BeamletLog = BeamletLogGen(matFolder, fluenceDim)
    fluenceMapFile = fullfile(matFolder, 'fluenceMap.bin');
    fileID = fopen(fluenceMapFile, 'r');
    fluenceMap = fread(fileID, inf, 'uint8');
    nBeams = numel(fluenceMap) / (fluenceDim(1) * fluenceDim(2));
    shape_new = [fluenceDim(1), fluenceDim(2), nBeams];
    BeamletLog = reshape(fluenceMap, shape_new);
end

function [params] = loadParams(paramsFile, paramsOrg)
    load(paramsFile, 'params');
    paramsOrg = fileread(paramsOrg);
    paramsOrg = splitlines(paramsOrg);
    beamWeight = strsplit(paramsOrg{1}, ',');
    beamWeight = str2num(beamWeight{2});
    gamma = strsplit(paramsOrg{2}, ',');
    gamma = str2num(gamma{2});
    eta = strsplit(paramsOrg{3}, ',');
    eta = str2num(eta{2});
    params.beamWeight = beamWeight;
    params.gamma = gamma;
    params.eta = eta;
end

function phantomdim = readDimension(dimensionFile)
    dimensionText = fileread(dimensionFile);
    dimension = strtok(dimensionText, newline);
    dimension = strsplit(dimension);
    phantomdim = [str2num(dimension{1}), str2num(dimension{2}), str2num(dimension{3})];
end

function result = getSelectedBeams(selectedBeams)
    selectedBeams = fileread(selectedBeams);
    selectedBeams = splitlines(selectedBeams);
    entries = numel(selectedBeams);
    result = {};
    for i = 2:entries-1
        line = selectedBeams{i};
        line = strsplit(line, ',');
        line = str2num(line{1});
        result{end+1} = line;
    end
    result = cell2mat(result)';
end

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

function M = loadM(doseFolder, phantomDim)
    offsetsBufferFile = fullfile(doseFolder, "offsetsBuffer.bin");
    columnsBufferFile = fullfile(doseFolder, "columnsBuffer.bin");
    valuesBufferFile = fullfile(doseFolder, "valuesBuffer.bin");
    NonZeroElementsFile = fullfile(doseFolder, "NonZeroElements.bin");
    numRowsPerMat = fullfile(doseFolder, "numRowsPerMat.bin");

    fid = fopen(offsetsBufferFile, 'rb');
    if fid == -1
        error('Could not open the binary file.');
    end
    offsetsBuffer = fread(fid, inf, "uint64");
    fclose(fid);
    fprintf('Finished loading the offsets buffer.\n');

    fid = fopen(columnsBufferFile, 'rb');
    if fid == -1
        error('Could not open the file.');
    end
    columnsBuffer = fread(fid, inf, "uint64");
    fclose(fid);
    fprintf('Finished loading the columns buffer.\n')

    fid = fopen(valuesBufferFile, 'rb');
    if fid == -1
        error('Could not open the file.');
    end
    valuesBuffer = fread(fid, inf, "float32");
    fclose(fid);
    fprintf('Finished loading the values buffer.\n');

    fid = fopen(numRowsPerMat, 'rb');
    if fid == -1
        error('Could not open the file.');
    end
    numRowsPerMat = fread(fid, inf, "uint64");
    fprintf('Finished loading the number of rows per matrix.\n')

    indexii = 1;  % row index
    indexjj = 1;  % element index
    mats = zeros(numel(columnsBuffer),1);
    for ii = 1:numel(numRowsPerMat)
        seg = offsetsBuffer(indexii:indexii+numRowsPerMat(ii));  % the offset vector of the current matrix
        mat1 = zeros(seg(end),1);  % of length Nnz of the current matrix
        mat1(seg(1:end-1)+1) = 1;  % set the first element of each line to be 1
        mats(indexjj:indexjj+seg(end)-1) = mat1;
        indexii = indexii+numRowsPerMat(ii)+1;
        indexjj = indexjj+seg(end);
    end
    rows = cumsum(mats);  % the row indices of each line
    M = sparse(columnsBuffer+1,rows,valuesBuffer,prod(phantomDim), sum(numRowsPerMat));
    fprintf('Finished sparse matrix construction\n');
end

function flag = IsZeroOrNaN(x)
    flag = ~all(isnan(x)|x==0);
end