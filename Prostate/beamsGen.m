FourPiAnglePath = "/data/qifan/projects/EndtoEnd4/BOO/beamlogs/4pi_angles.mat";
LiverBeamlogPath = "/data/qifan/projects/EndtoEnd4/BOO/beamlogs/prostate_beamlog.mat";
load(FourPiAnglePath);
load(LiverBeamlogPath);

Gantry = theta_VarianIEC(beamlog_iso==1, 1);
Couch = theta_VarianIEC(beamlog_iso==1, 2);
MLCangles = zeros(length(Gantry), 1);
Angles = [Gantry, Couch, MLCangles];

beamListFile = "/data/qifan/projects/FastDoseWorkplace/CORTTune/Prostate/beamlist.txt";
fileID = fopen(beamListFile, "w");
for ii = 1:size(Angles, 1)
    fprintf(fileID, '%6.4f %6.4f %6.4f \n', Angles(ii, :, :));
end
fclose(fileID);