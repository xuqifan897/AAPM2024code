beamlogFolder = '/data/qifan/projects/EndtoEnd4/BOO/beamlogs';
anglesFile = fullfile(beamlogFolder, '4pi_angles.mat');
head_beamlog_file = fullfile(beamlogFolder, 'head_beamlog.mat');
load(anglesFile);
load(head_beamlog_file);

MLCangle = 0; % MLC angle is set to zero;
Gantry = theta_VarianIEC(beamlog_iso==1,1);
Couch = theta_VarianIEC(beamlog_iso==1,2);
MLCangles = MLCangle*ones(length(Gantry),1);
Angles = [Gantry Couch MLCangles];

beamlistFile = '/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/FastDose/beamlist.txt';
fileID = fopen(beamlistFile, 'w');
for ii = 1:size(Angles,1)
    fprintf(fileID,'%6.4f %6.4f %6.4f \n',Angles(ii,:));
end
fclose(fileID);