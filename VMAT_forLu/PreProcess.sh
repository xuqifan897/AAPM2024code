#!/bin/bash

exec="/data/qifan/projects/FastDose/build/bin/preprocess"
rootFolder="/data/qifan/projects/FastDoseWorkplace/VMAT_forLu"
patientName="HN_002"
groupName="our_model"

patientFolder="${rootFolder}/${patientName}"
planFolder="${patientFolder}/planFolder"
inputFolder="${planFolder}/prep_output_${groupName}"
if [ ! -d ${inputFolder} ]; then
    mkdir ${inputFolder}
fi

if [[ ${patientName} == "HGJ_001" ]]; then
    shape="199 199 119"
elif [[ ${patientName} == "HN_002" ]]; then
    shape="200 200 156"
fi

${exec} \
    --mode 1 \
    --structuresFile "${planFolder}/structures_${groupName}.json" \
    --ptv_name "PTVMerge" \
    --bbox_name "BODY" \
    --voxelSize 0.25 \
    --inputFolder ${inputFolder} \
    --shape ${shape} \
    --phantomPath "${planFolder}/density_raw.bin" \
    --RescaleSlope 1.0 \
    --RescaleIntercept -1000.0 \
    --maskFolder "${planFolder}/InputMask_${groupName}"