#!/bin/bash

exec="/data/qifan/projects/FastDose/build/bin/IMRT"
dataFolder="/data/qifan/projects/FastDose/scripts"
rootFolder="/data/qifan/projects/FastDoseWorkplace/Breast"
patientName="12908474Right"

patientFolder="${rootFolder}/${patientName}"
expFolder="${patientFolder}/expFolder"
inputFolder="${expFolder}/prep_output"
optResultFolder="${expFolder}/plan1"
doseMatFolder="${expFolder}/doseMat"
if [ ! -d ${inputFolder} ]; then
    echo "The folder ${inputFolder} doesn't exist"
fi
if [ ! -d ${optResultFolder} ]; then
    mkdir ${optResultFolder}
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
    --primaryROI "PTV_PBI_R" \
    --bboxROI "External" \
    --structureInfo "${expFolder}/StructureInfo.csv" \
    --params "${expFolder}/params.txt" \
    --beamlist "${expFolder}/beamlist.txt" \
    --mode 1 \
    --deviceIdx 3 \
    --spectrum "${dataFolder}/spec_6mv.spec" \
    --kernel "${dataFolder}/kernel_exp_6mv.txt" \
    --fluenceDim 20 \
    --subFluenceDim 16 \
    --outputFolder ${doseMatFolder} \
    --planFolder ${optResultFolder} \
    --nBeamsReserve 452 \
    --EstNonZeroElementsPerMat 6000000 \
| tee ${logFile}