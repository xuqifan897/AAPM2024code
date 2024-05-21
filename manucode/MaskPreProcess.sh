#!/bin/bash

exec="/data/qifan/projects/FastDose/build/bin/preprocess"

rootFolder="/data/qifan/projects/FastDoseWorkplace/DoseBench/composite"
maskFolder="${rootFolder}/InputMask"
inputFolder="${rootFolder}/prep_result"

${exec} \
    --mode 1 \
    --structuresFile "${rootFolder}/structures.json" \
    --ptv_name "PTV" \
    --bbox_name "BODY" \
    --voxelSize 0.1 \
    --inputFolder "${inputFolder}" \
    --shape 99 256 99 \
    --phantomPath "${rootFolder}/density_raw.bin" \
    --RescaleSlope 1.0 \
    --RescaleIntercept -1000.0 \
    --maskFolder "${maskFolder}"