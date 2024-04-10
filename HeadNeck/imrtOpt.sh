#!/bin/bash

exec="/data/qifan/projects/FastDose/build/bin/IMRT"
dataFolder="/data/qifan/projects/FastDose/scripts"
globalFolder="/data/qifan/projects/FastDoseWorkplace/CORTTune/HeadNeck"
inputFolder="${globalFolder}/prep_output"
outputFolder="${globalFolder}/doseMat"
planFolder="${globalFolder}/plan2"

if [ ! -d ${inputFolder} ]; then
    echo "The folder ${inputFolder} doesn't exist."
fi

if [ ! -d ${outputFolder} ]; then
    mkdir ${outputFolder}
fi

if [ ! -d ${planFolder} ]; then
    mkdir ${planFolder}
fi

dimFile="${inputFolder}/dimension.txt"
readarray -t lines < ${dimFile}
phantomDim=${lines[0]}
voxelSize=${lines[1]}
VOIs=${lines[2]}

OMP_NUM_THREADS=64 ${exec} \
    --phantomDim ${phantomDim} \
    --voxelSize ${voxelSize} \
    --SAD 100 \
    --density "${inputFolder}/density.raw" \
    --structures ${VOIs} \
    --masks "${inputFolder}/roi_list.h5" \
    --primaryROI "PTV_crop" \
    --bboxROI "External" \
    --structureInfo "${globalFolder}/StructureInfo.csv" \
    --params "${globalFolder}/params.txt" \
    --beamlist "${globalFolder}/beamlist_full.txt" \
    --mode 1 \
    --deviceIdx 2 \
    --spectrum "${dataFolder}/spec_6mv.spec" \
    --kernel "${dataFolder}/kernel_exp_6mv.txt" \
    --subFluenceDim 16 \
    --outputFolder ${outputFolder} \
    --planFolder ${planFolder} \
    --nBeamsReserve 1162 \
    --EstNonZeroElementsPerMat 2000000 \