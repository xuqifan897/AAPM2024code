#!/bin/bash

patientFolder="/data/qifan/projects/FastDoseWorkplace/TCIASelect/027"
expFolder="${patientFolder}/FastDose"
inputFolder="${expFolder}/prep_output"
if [ ! -d ${expFolder} ]; then
    mkdir ${expFolder}
fi
if [ ! -d ${inputFolder} ]; then
    mkdir ${inputFolder}
fi

exec="/data/qifan/projects/FastDose/build/bin/preprocess"

${exec} \
    --mode 1 \
    --structuresFile "${patientFolder}/structures.json" \
    --ptv_name "PTVMerge" \
    --bbox_name "SKIN" \
    --voxelSize 0.25 \
    --inputFolder ${inputFolder} \
    --shape 200 200 210 \
    --phantomPath "${patientFolder}/density_raw.bin" \
    --RescaleSlope 1.0 \
    --RescaleIntercept -1024.0 \
    --maskFolder "${patientFolder}/InputMask"