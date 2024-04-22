#!/bin/bash

exec="/data/qifan/projects/FastDose/build/bin/IMRT"
dataFolder="/data/qifan/projects/FastDose/scripts"
expFolder="/data/qifan/projects/FastDoseWorkplace/Pancreas/Patient003/FastDose"

inputFolder="${expFolder}/prep_output"
outputFolder="${expFolder}/doseMatMerge"
planFolder="${expFolder}/plan1"

if [ ! -d ${inputFolder} ]; then
    echo "The folder ${inputFolder} doesn't exist."
fi
if [ ! -d ${outputFolder} ]; then
    echo "The folder ${outputFolder} doesn't exist."
fi
if [ ! -d ${planFolder} ]; then
    mkdir ${planFolder}
fi

dimFile="${inputFolder}/dimension.txt"
readarray -t lines < ${dimFile}
phantomDim=${lines[0]}
voxelSize=${lines[1]}
VOIs=${lines[2]}

logFile="${expFolder}/optimize.log"

OMP_NUM_THREADS=64 ${exec} \
    --phantomDim ${phantomDim} \
    --voxelSize ${voxelSize} \
    --SAD 100 \
    --density "${inputFolder}/density.raw" \
    --structures ${VOIs} \
    --masks "${inputFolder}/roi_list.h5" \
    --primaryROI "PTVMerge" \
    --bboxROI "SKIN" \
    --structureInfo "${expFolder}/StructureInfo.csv" \
    --params "${expFolder}/params.txt" \
    --beamlist "${expFolder}/beamlist$.txt" \
    --mode 1 \
    --deviceIdx 1 \
    --spectrum "${dataFolder}/spec_6mv.spec" \
    --kernel "${dataFolder}/kernel_exp_6mv.txt" \
    --fluenceDim 20 \
    --subFluenceDim 16 \
    --outputFolder ${outputFolder} \
    --planFolder ${planFolder} \
    --nBeamsReserve 200 \
    --EstNonZeroElementsPerMat 6000000 \
| tee ${logFile}