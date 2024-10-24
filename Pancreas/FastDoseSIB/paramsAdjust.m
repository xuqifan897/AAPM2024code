sourceFolder = '/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB';
targetFolder = fullfile(sourceFolder, 'beamExamine');
if ~ isfolder(targetFolder)
    mkdir(targetFolder);
end
PTVName = 'ROI';
BODYName = 'SKIN';
RingStructure = 'ELSE';
excludeList = {PTVName, BODYName, RingStructure};
numPatients = 5;
for i = 1:numPatients
    patientName = ['Patient00', num2str(i)];
    patientFolder = fullfile(sourceFolder, patientName);
    QihuiRyanFolder = fullfile(patientFolder, 'QihuiRyan');
    StructureInfo1 = fullfile(QihuiRyanFolder, 'StructureInfo1.mat');
    load(StructureInfo1, 'StructureInfo');
    StructureInfo_template = StructureInfo;
    clear StructureInfo;
    
    dimension = fullfile(patientFolder, 'FastDose', 'prep_output_else', 'dimension.txt');
    fileID = fopen(dimension, 'r');
    lines = textscan(fileID, '%s', 'Delimiter', '\n');
    fclose(fileID);
    dimension = str2num(lines{1}{1});

    maskFolder = fullfile(patientFolder, 'InputMask');
    structListing = dir(maskFolder);
    structures = {};
    for j = 1:length(structListing)
        name = split(structListing(j).name, '.');
        name = name{1};
        if ~ strcmp(name, '')
            structures{end+1} = name;
        end
    end
    OARs = {};
    for j = 1:length(structures)
        name = structures{j};
        flag = true;
        for k = 1:length(excludeList)
            if strcmp(name, excludeList{k})
                flag = false;
                break;
            end
        end
        if flag
            OARs{end + 1} = name;
        end
    end

    nStructures = numel(OARs) + 3;
    StructureInfo = struct('Name', '', 'maxWeights', 0, 'minDoseTargetWeights', 0, ...
        'minDoseTarget', 0, 'OARWeights', 0, 'IdealDose', 0, 'Mask', [], ...
        'VoxelNum', 0);
    StructureInfo = repmat(StructureInfo, nStructures, 1);

    % Initialize PTV
    StructureInfo(1).Name = 'PTV';
    StructureInfo(1).maxWeights = 100;
    StructureInfo(1).maxDose = 20;
    StructureInfo(1).minDoseTargetWeights = 100;
    StructureInfo(1).minDoseTarget = 20;
    StructureInfo(1).OARWeights = NaN;
    StructureInfo(1).IdealDose = 20;
    StructureInfo(1).Mask = getMask(maskFolder, PTVName, dimension);
    StructureInfo(1).VoxelNum = sum(StructureInfo(1).Mask(:));

    % Initialize BODY
    StructureInfo(2).Name = 'BODY';
    StructureInfo(2).maxWeights = 0;
    StructureInfo(2).maxDose = 18;
    StructureInfo(2).minDoseTargetWeights = NaN;
    StructureInfo(2).minDoseTarget = NaN;
    StructureInfo(2).OARWeights = 0;
    StructureInfo(2).IdealDose = 0;
    StructureInfo(2).Mask = getMask(maskFolder, BODYName, dimension);
    StructureInfo(2).VoxelNum = sum(StructureInfo(2).Mask(:));

    % Initialize OAR
    for j = 1:numel(OARs)
        Name = OARs{j};
        StructureInfo(j+2).Name = Name;
        StructureInfo(j+2).maxWeights = 0;
        StructureInfo(j+2).maxDose = 18;
        StructureInfo(j+2).minDoseTargetWeights = NaN;
        StructureInfo(j+2).minDoseTarget = NaN;
        StructureInfo(j+2).OARWeights = 5;
        StructureInfo(j+2).IdealDose = 0;
        StructureInfo(j+2).Mask = getMask(maskFolder, Name, dimension);
        StructureInfo(j+2).VoxelNum = sum(StructureInfo(j+2).Mask(:));
    end

    % Initialize ELSE
    StructureInfo(end).Name = RingStructure;
    StructureInfo(end).maxWeights = 0;
    StructureInfo(end).maxDose = 18;
    StructureInfo(end).minDoseTargetWeights = NaN;
    StructureInfo(end).minDoseTarget = NaN;
    StructureInfo(end).OARWeights = 5;
    StructureInfo(end).IdealDose = 0;
    StructureInfo(end).Mask = getMask(maskFolder, RingStructure, dimension);
    StructureInfo(end).VoxelNum = sum(StructureInfo(end).Mask(:));

    StructureInfoFile = fullfile(QihuiRyanFolder, 'StructureInfo_else5.mat');
    save(StructureInfoFile, 'StructureInfo');
    fprintf([StructureInfoFile, '\n']);
    % showStruct(StructureInfo)


    paramsInput = fullfile(QihuiRyanFolder, 'params1.mat');
    load(paramsInput, 'params');
    params.beamWeight = params.beamWeight * 1.25;
    paramsOutput = fullfile(QihuiRyanFolder, 'params_else5.mat');
    save(paramsOutput, 'params');
    fprintf([paramsOutput, '\n\n']);
end

function showStruct(Struct)
    result = {}
    for i = 1:numel(Struct)
        result{end+1} = Struct(i).maxDose;
    end
    result
end

function Mask = getMask(folder, name, dimension)
    size = prod(dimension(:));
    file = fullfile(folder, [name, '.bin']);
    fileID = fopen(file);
    Mask = fread(fileID, size, 'uint8');
    Mask = reshape(Mask, dimension);
    Mask = logical(Mask);
    Mask = permute(Mask, [2, 1, 3]);
end