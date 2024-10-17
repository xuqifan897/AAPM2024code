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
sourceFolder = '/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB';
numPatients = 5;
thresh = 1e-6;
PrescriptionDose = 20;  % dummy


if false
    for i = 5:numPatients
        patientName = ['Patient00', num2str(i)];
        expFolder = fullfile(sourceFolder, patientName, "QihuiRyan");
        h5file = fullfile(expFolder, 'Dose_Coefficients.h5');
        maskfile = fullfile(expFolder, 'Dose_Coefficients.mask');
        [M, dose_data, masks] = BuildDoseMatrix(h5file, maskfile, thresh);

        paramsText = fullfile(sourceFolder, patientName, 'FastDose', 'params.txt');
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

        % rename the PTV
        ptvIn = 'ROI';
        ptvOut = 'PTV';
        assert(strcmp(masks{1}.name, ptvIn));
        masks{1}.name = ptvOut;
        [StructureInfo, params] = InitIMRTparams_NoRing(M, dose_data, masks, PrescriptionDose);
        params.beamWeight = beamWeight_QihuiRyan;
        paramsFileOut = fullfile(expFolder, 'params1.mat');
        save(paramsFileOut, 'params');

        structureInfoText = fullfile(sourceFolder, patientName, "FastDose", "StructureInfo.csv");
        fileID = fopen(structureInfoText, "r");
        structureInfoLines = {};
        while ~feof(fileID)
            structureInfoLines{end+1} = fgetl(fileID);
        end
        fclose(fileID);
        % skip the first line
        structureInfoLines = structureInfoLines(2:end);

        for j = 1:length(structureInfoLines)
            line = structureInfoLines{j};
            line = split(line, ',');
            Name = line{1};
            if (strcmp(Name, ptvIn))
                Name = ptvOut;
            end
            maxWeights = str2num(line{2});
            maxDose = str2num(line{3});
            minDoseTargetWeights = str2num(line{4});
            minDoseTarget = str2num(line{5});
            OARWeights = str2num(line{6});
            IdealDose = str2num(line{7});

            found = false;
            for k = 1:length(StructureInfo)
                if (strcmp(Name, StructureInfo(k).Name))
                    found = true;
                    break;
                end
            end
            assert(found, Name);
            StructureInfo(k).maxWeights = maxWeights;
            StructureInfo(k).maxDose = maxDose;
            StructureInfo(k).minDoseTargetWeights = minDoseTargetWeights;
            StructureInfo(k).minDoseTarget = minDoseTarget;
            StructureInfo(k).OARWeights = OARWeights;
            StructureInfo(k).IdealDose = IdealDose;
        end

        StructureInfoFileOut = fullfile(expFolder, "StructureInfo1.mat");
        save(StructureInfoFileOut, 'StructureInfo');
        patientName
    end
end

% change the parameters based on existing structureInfo
for i = 1:5
    patientName = ['Patient00', num2str(i)];
    patientFolder = fullfile(sourceFolder, patientName, 'QihuiRyan');
    sourceFile = fullfile(patientFolder, 'StructureInfo0.mat');
    load(sourceFile);
    OARWeightArray = zeros(1, length(StructureInfo));
    for j = 1:length(StructureInfo)
        if abs(StructureInfo(j).OARWeights - 1) < eps
            StructureInfo(j).OARWeights = 5;
        end
        OARWeightArray(j) = StructureInfo(j).OARWeights;
    end
    % OARWeightArray
    outputFile = fullfile(patientFolder, 'StructureInfo1.mat');
    save(outputFile, 'StructureInfo');
end