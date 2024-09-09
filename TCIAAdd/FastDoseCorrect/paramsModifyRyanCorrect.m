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

sourceFolder = '/data/qifan/projects/FastDoseWorkplace/TCIAAdd';
targetFolder = fullfile(sourceFolder, 'plansAngleCorrect');
patientList = {'002', '003', '009', '013', '070', '125', '132', '190'};
numPatietns = length(patientList);

% for i = 1:numPatietns
i = 6;
    patientName = patientList{i};
    patientFolder = fullfile(targetFolder, patientName);
    QihuiRyanFolder = fullfile(patientFolder, 'QihuiRyan');
    paramsTemplate = fullfile(QihuiRyanFolder, 'params0.mat');
    load(paramsTemplate, 'params');
    
    paramsText = fullfile(patientFolder, 'FastDose', 'params1.txt');
    fileID = fopen(paramsText, 'r');
    paramsLines = {};
    while ~ feof(fileID)
        paramsLines{end+1} = fgetl(fileID);
    end
    fclose(fileID);
    beamWeight = paramsLines{1};
    beamWeight = split(beamWeight, ',');
    beamWeight = str2num(beamWeight{2});
    beamWeight_QihuiRyan = beamWeight / 110;

    % assign the new beamweight to params
    params.beamWeight = beamWeight_QihuiRyan;
    params.stepSize = 1e-5;
    paramsFileOut = fullfile(QihuiRyanFolder, 'params1.mat');
    save(paramsFileOut, 'params');

    structureInfoText = fullfile(targetFolder, patientName, 'FastDose', 'StructureInfo1.csv');
    fileID = fopen(structureInfoText, 'r');
    structureInfoLines = {};
    while ~feof(fileID)
        structureInfoLines{end+1} = fgetl(fileID);
    end
    fclose(fileID);
    % skip the first line
    structureInfoLines = structureInfoLines(2:end);

    % read masks
    maskFile = fullfile(sourceFolder, patientName, 'FastDose', 'prep_output', 'roi_list.h5');
    masks = loadMasks(maskFile);

    StructureInfo(length(structureInfoLines)) = struct('Name', '', 'maxWeights', [], ...
        'maxDose', [], 'minDoseTargetWeights', [], 'minDoseTarget', [], 'OARWeights', [], ...
        'IdealDose', [], 'Mask', [], 'VoxelNum', []);
    for j = 1:length(structureInfoLines)
        line = structureInfoLines{j};
        line = split(line, ',');
        Name = line{1};
        maxWeights = str2num(line{2});
        maxDose = str2num(line{3});
        minDoseTargetWeights = str2num(line{4});
        minDoseTarget = str2num(line{5});
        OARWeights = str2num(line{6});
        IdealDose = str2num(line{7});

        StructureInfo(j).Name = Name;
        StructureInfo(j).maxWeights = maxWeights;
        StructureInfo(j).maxDose = maxDose;
        StructureInfo(j).minDoseTargetWeights = minDoseTargetWeights;
        StructureInfo(j).minDoseTarget = minDoseTarget;
        StructureInfo(j).OARWeights = OARWeights;
        StructureInfo(j).IdealDose = IdealDose;

        % copy the mask
        found = false;
        for k = 1:length(masks)
            if strcmp(Name, masks{k}.name)
                found = true;
                StructureInfo(j).Mask = masks{k}.mask;
                StructureInfo(j).VoxelNum = nnz(masks{k}.mask);
            end
        end
        assert(found);
    end
    StructureInfoFileOut = fullfile(QihuiRyanFolder, 'StructureInfo1.mat');
    save(StructureInfoFileOut, 'StructureInfo');
    patientName
% end


function masks = loadMasks(maskfile)
    [ masks ] = open_masks( maskfile, 'xyz', 2);
    for i = 1:length(masks)
        masks{i}.mask = permute(masks{i}.mask,[2,1,3]);
    end
end