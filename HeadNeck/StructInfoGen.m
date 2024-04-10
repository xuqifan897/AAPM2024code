structFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/HN_withoutDij";
files = dir(structFolder);
names = {};
for i = 1:length(files)
    file = files(i);
    name = file.name;
    if strcmp(name, '..') || strcmp(name, '.')
        continue;
    end
    if contains(name, "VOILIST.mat")
        names{end+1} = name;
    end
end

targetFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/HN_binary";
if ~ isfolder(targetFolder)
    mkdir(targetFolder)
end

shape = [160, 160, 67];
length_to_remove = length('_VOILIST.mat');
for i = 1:length(names)
    file = names{i};
    name = file(1:length(file) - length_to_remove);
    path = fullfile(structFolder, file);
    load(path);
    fullMask = zeros(shape, "uint8");
    fullMask(v) = 1;
    outputFile = fullfile(targetFolder, [name, '.bin']);
    f = fopen(outputFile, 'w');
    fwrite(f, fullMask, "uint8");
    fclose(f);
    outputFile
end