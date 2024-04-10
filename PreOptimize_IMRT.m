restoredefaultpath;
addpath(genpath('BOO_QL'), '-end');
addpath(genpath('CERR2016'), '-end');
addpath(genpath('CERRaddins'), '-end');
addpath(genpath('utilities'), '-end');
addpath(genpath('beamlogs'), '-end');

clear; clc;
patientName = 'LUNG';
PrescriptionDose = 20;
beamlogfile = 'lung_beamlog.mat';

%% prepare for optimization
optFolder = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_bench";

h5file = fullfile(optFolder, 'Dose_Coefficients.h5');
maskfile = fullfile(optFolder, 'Dose_Coefficients.mask');
thresh = 1e-06;
[M, dose_data, masks] = BuildDoseMatrix(h5file, maskfile, thresh);
save(fullfile(optFolder, [patientName, '_M_original.mat']), 'M', 'dose_data', 'masks', '-v7.3');

% save parameter files
[StructureInfo, params] = InitIMRTparams(M, dose_data, masks, PrescriptionDose);
ParamsNum = 0;
save(fullfile(optFolder, ['params', num2str(ParamsNum), '_original.mat']), 'params');
InfoNum = 0;
save(fullfile(optFolder, ['StructureInfo', num2str(InfoNum), '.mat']), 'StructureInfo');

%% Remove beams going through cut off CT images
PTV = StructureInfo(1).Mask;
BODY = StructureInfo(2).Mask;
% figure; imshow3D([PTV, BODY], []);

Isos = GetPTV_COM(PTV);
[zendpos,zendmaskPos,zstartpos, zstartmaskPos] = GetBODYend_dimenless(BODY);

beamfpangles = params.beamfpangles;
[srcpos, dirXray] = fpangles2sourcerayinmask(beamfpangles,Isos,3000);

dr2 = 50;
numbeams = size(beamfpangles,1);
badbeams = zeros(numbeams,1);
for ii = 1:size(beamfpangles,1)
    isrcpos = srcpos(ii,:);
    idirXray = dirXray(ii,:);
    if(sign((zendpos-Isos(3))*(isrcpos(3) - Isos(3)))==1)
        if((zendpos-Isos(3))/idirXray(3)<0)
            error('Wrong direction!')
        end
        endpos = Isos + (zendpos-Isos(3))/idirXray(3)*idirXray;
        if(~isempty(zendmaskPos))
            if(find(sum((endpos'-zendmaskPos).^2,1)<dr2))
                badbeams(ii) = 1;
            end
        end
    elseif(sign((zstartpos-Isos(3))*(isrcpos(3) - Isos(3)))==1)
        if((zstartpos-Isos(3))/idirXray(3)<0)
            error('Wrong direction!')
        end
        endpos = Isos + (zstartpos-Isos(3))/idirXray(3)*idirXray;
        if(~isempty(zstartmaskPos))
            if(any(sum((endpos'-zstartmaskPos).^2,1)<dr2))
                badbeams(ii) = 1;
            end
        end
    end
end

draw_beammask_QL(params.beamfpangles(badbeams==1,:),BODY,PTV);
draw_beammask_QL(params.beamfpangles(badbeams==0,:),BODY,PTV);

goodbeamind = find(badbeams==0);
newparams = params;
newparams.beamWeightsInit = params.beamWeightsInit(goodbeamind);
newparams.beamSizes = params.beamSizes(goodbeamind);
newparams.BeamletLog0 = params.BeamletLog0(:,:,goodbeamind);
newparams.beamVarianIEC = params.beamVarianIEC(goodbeamind,:);
newparams.beamfpangles = params.beamfpangles(goodbeamind,:);

BeamletLog0 = params.BeamletLog0;
BeamletLog0Ind = BeamletLog0;
BeamletLog0Ind(BeamletLog0==1) = 1:nnz(BeamletLog0);
newbeamletind = BeamletLog0Ind(:,:,goodbeamind);
newbeamletind = newbeamletind(newbeamletind>0);

if(sum(newparams.beamSizes)~=numel(newbeamletind))
    error('Dimension error in removing beams!')
end

M = M(:,newbeamletind);
save(fullfile(optFolder,[patientName '_M.mat']),'M','-v7.3');

params = newparams;
save(fullfile(optFolder,['params' num2str(ParamsNum) '.mat']),'params');