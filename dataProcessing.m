%% Process liver dataset
if (false)
    maskFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/LIVERSept24";
    outputFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/Liver_mask";
    if ~ isfolder(outputFolder)
        mkdir(outputFolder);
    end
    directoryContent = dir(maskFolder);
    phantomDim = [217, 217, 168];
    n_voxels = prod(phantomDim);
    roi_list = {};
    for i = 1 : length(directoryContent)
        name = directoryContent(i).name;
        path = fullfile(maskFolder, name);
        if contains(name, 'VOILIST.mat')
            load(path);
            maskArray = zeros(n_voxels, 1, "uint8");
            maskArray(v) = 1;
            roi_name = split(name, '_');
            roi_name = roi_name{1};
            if (ismember(roi_name, roi_list))
                'roi_name aready processed'
            end
            roi_list{end+1} = roi_name;
            outputFile = fullfile(outputFolder, [roi_name, '.bin']);
            file_id = fopen(outputFile, 'w');
            fwrite(file_id, maskArray, "uint8");
            fclose(file_id);
        end
    end
end


%% Process prostate dataset
if (false)
    sourceDir = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE";
    file = fullfile(sourceDir, "prostate3mmvoxels.mat");
    load(file);
    outputFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_binary";
    if (~ isfolder(outputFolder))
        mkdir(outputFolder)
    end

    %% write density file
    if (false)
        densityFile = fullfile(outputFolder, 'density.bin');
        file_id = fopen(densityFile, 'w');
        fwrite(file_id, planC{3}.scanArray, "uint16");
        fclose(file_id);
    end

    phantomDim = [184, 184, 90];
    n_voxels = prod(phantomDim);
    roi_list = {};
    directoryContent = dir(sourceDir);
    for i = 1 : length(directoryContent)
        name = directoryContent(i).name;
        path = fullfile(sourceDir, name);
        pattern = '_VOILIST.mat';
        if (contains(name, pattern))
            load(path);
            maskArray = zeros(n_voxels, 1, "uint8");
            maskArray(v) = 1;
            roi_name = name(1: length(name) - length(pattern));
            if (ismember(roi_name, roi_list))
                'roi_name aready processed'
                roi_name
            end
            roi_list{end+1} = roi_name;
            outputFile = fullfile(outputFolder, [roi_name, '.bin']);
            file_id = fopen(outputFile, 'w');
            fwrite(file_id, maskArray, "uint8");
            fclose(file_id);
        end
    end
end


%% translate beamlist
% sourceFile = '/data/qifan/projects/FastDoseWorkplace/PlanTune/HN_withoutDij/GantryCouchAngles.mat';
sourceFile = '/data/qifan/projects/FastDoseWorkplace/PlanTune/LIVERSept24/GantryCouchAngles.mat';
load(sourceFile);