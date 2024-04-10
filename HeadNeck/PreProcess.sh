#!/bin/bash

globalFolder="/data/qifan/projects/FastDoseWorkplace/CORTTune/HeadNeck"
inputFolder="${globalFolder}/prep_output"
if [ ! -d ${inputFolder} ]; then
    mkdir ${inputFolder}
fi

exec="/data/qifan/projects/FastDose/build/bin/preprocess"
sourceFolder="/data/qifan/projects/FastDoseWorkplace/circDicom/HeadNeck"

${exec} \
    --mode 1 \
    --structuresFile "${sourceFolder}/structures.json" \
    --ptv_name "PTV_crop" \
    --bbox_name "External" \
    --voxelSize 0.25 \
    --inputFolder "${globalFolder}/prep_output" \
    --shape 193 193 108 \
    --phantomPath "${sourceFolder}/density_raw.bin" \
    --RescaleSlope 1.0 \
    --RescaleIntercept -1024.0 \
    --maskFolder "${sourceFolder}/InputMask"