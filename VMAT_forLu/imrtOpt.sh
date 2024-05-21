#!/bin/bash

exec="/data/qifan/projects/FastDose/build/bin/IMRT"
dataFolder="/data/qifan/projects/FastDose/scripts"
rootFolder="/data/qifan/projects/FastDoseWorkplace/VMAT_forLu"
patientName="HN_002"
groupName="our_model"

patientFolder="${rootFolder}/${patientName}"
planFolder="${patientFolder}/planFolder"
inputFolder="${planFolder}/prep_output_${groupName}"
optResultFolder="${planFolder}/plan1_${groupName}"
if [ ! -d ${inputFolder} ]; then
    echo "The folder ${inputFolder} doesn't exist"
fi
if [ ! -d ${optResultFolder} ]; then
    mkdir ${optResultFolder}
fi

outputFolder="${planFolder}/${groupName}_doseMatMerge"
if [ ! -d ${outputFolder} ]; then
    mkdir ${outputFolder}
fi

dimFile="${inputFolder}/dimension.txt"
readarray -t lines < ${dimFile}
phantomDim=${lines[0]}
voxelSize=${lines[1]}
VOIs=${lines[2]}

logFile="${planFolder}/optimize_${groupName}.log"
OMP_NUM_THREADS=64 ${exec} \
    --phantomDim ${phantomDim} \
    --voxelSize ${voxelSize} \
    --SAD 100 \
    --density "${inputFolder}/density.raw" \
    --structures ${VOIs} \
    --masks "${inputFolder}/roi_list.h5" \
    --primaryROI "PTVMerge" \
    --bboxROI "BODY" \
    --structureInfo "${planFolder}/StructureInfo_${groupName}.csv" \
    --params "${planFolder}/params_${groupName}.txt" \
    --beamlist "${planFolder}/beamlist.txt" \
    --mode 1 \
    --deviceIdx 0 \
    --spectrum "${dataFolder}/spec_6mv.spec" \
    --kernel "${dataFolder}/kernel_exp_6mv.txt" \
    --fluenceDim 20 \
    --subFluenceDim 16 \
    --outputFolder ${outputFolder} \
    --planFolder ${optResultFolder} \
    --nBeamsReserve 200 \
    --EstNonZeroElementsPerMat 12000000 \
| tee ${logFile}