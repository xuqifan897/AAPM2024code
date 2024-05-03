#!/bin/bash

rootFolder="/data/qifan/projects/FastDoseWorkplace/DoseBench/water"
exec="/data/qifan/projects/FastDose/build/bin/preprocess"

structuresFile="${rootFolder}/structures.json"
inputFolder="${rootFolder}/prep_output"
densityFile="${rootFolder}/waterDensity.bin"
maskFolder="${rootFolder}/InputMask"
if [ ! -d ${inputFolder} ]; then
    mkdir ${inputFolder}
fi

${exec} \
    --mode 1 \
    --structuresFile ${structuresFile} \
    --ptv_name "PTV" \
    --bbox_name "SKIN" \
    --voxelSize 0.25 \
    --inputFolder ${inputFolder} \
    --shape 100 100 100 \
    --phantomPath ${densityFile} \
    --RescaleSlope 1.0 \
    --RescaleIntercept -1000.0 \
    --maskFolder ${maskFolder}