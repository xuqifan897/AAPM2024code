#!/bin/bash

exec="/data/qifan/projects/FastDose/build/bin/preprocess"
rootFolder="/data/qifan/projects/FastDoseWorkplace/Breast"
patientName="33445300"

patientFolder="${rootFolder}/${patientName}"
expFolder="${patientFolder}/expFolder"
inputFolder="${expFolder}/prep_output"
if [ ! -d ${inputFolder} ]; then
    mkdir ${inputFolder}
fi

${exec} \
    --mode 1 \
    --structuresFile "${expFolder}/structures.json" \
    --ptv_name "PTV_PBI_L" \
    --bbox_name "External" \
    --voxelSize 0.25 \
    --inputFolder ${inputFolder} \
    --shape 240 240 174 \
    --phantomPath "${expFolder}/density_raw.bin" \
    --RescaleSlope 1.0 \
    --RescaleIntercept -1000.0 \
    --maskFolder "${expFolder}/MaskInput"