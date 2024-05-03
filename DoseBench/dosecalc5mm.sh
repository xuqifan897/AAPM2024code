#!/bin/bash

exec="/data/qifan/projects/FastDose/build/bin/IMRT"
dataFolder="/data/qifan/projects/FastDose/scripts"
expFolder="/data/qifan/projects/FastDoseWorkplace/DoseBench/water/width5mm"

inputFolder="/data/qifan/projects/FastDoseWorkplace/DoseBench/water/prep_output"
if [ ! -d ${inputFolder} ]; then
    echo "The folder ${inputFolder} doesn't exist."
fi

outputFolder="${expFolder}/doseMat"
if [ ! -d ${outputFolder} ]; then
    mkdir ${outputFolder}
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
    --primaryROI "PTV" \
    --bboxROI "SKIN" \
    --structureInfo "${expFolder}/StructureInfo.csv" \
    --params "${expFolder}/params.txt" \
    --beamlist "/data/qifan/projects/FastDoseWorkplace/DoseBench/water/beamlist.txt" \
    --mode 0 \
    --deviceIdx 1 \
    --spectrum "${dataFolder}/spec_6mv.spec" \
    --kernel "${dataFolder}/kernel_exp_6mv.txt" \
    --fluenceDim 9 \
    --subFluenceDim 16 \
    --outputFolder ${outputFolder} \
    --planFolder ${planFolder} \
    --nBeamsReserve 1 \
    --EstNonZeroElementsPerMat 10000000