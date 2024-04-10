#!/bin/bash

globalFolder="/data/qifan/projects/FastDoseWorkplace/CORTTune/Liver"
inputFolder="${globalFolder}/prep_output"
if [ ! -d ${inputFolder} ]; then
    mkdir ${inputFolder}
fi

exec="/data/qifan/projects/FastDose/build/bin/preprocess"
sourceFolder="/data/qifan/projects/FastDoseWorkplace/circDicom/Liver"

${exec} \
    --mode 1 \
    --structuresFile "${sourceFolder}/structures.json" \
    --ptv_name "PTV" \
    --bbox_name "Skin" \
    --voxelSize 0.25 \
    --inputFolder "${globalFolder}/prep_output" \
    --shape 260 260 168 \
    --phantomPath "${sourceFolder}/density_raw.bin" \
    --RescaleSlope 1.0 \
    --RescaleIntercept -1024.0 \
    --maskFolder "${sourceFolder}/InputMask"