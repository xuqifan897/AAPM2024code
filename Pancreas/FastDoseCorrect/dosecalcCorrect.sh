#!/bin/bash

exec="/data/qifan/FastDose/build/bin/IMRT"
dataFolder="/data/qifan/FastDose/scripts"
sourceFolder="/data/qifan/FastDoseWorkplace/Pancreas"
targetFolder="${sourceFolder}/plansAngleCorrect"

patientName="Patient001"
inputFolder="${sourceFolder}/${patientName}/FastDose/prep_output"
dimFile="${inputFolder}/dimension.txt"
readarray -t lines < ${dimFile}
phantomDim=${lines[0]}
voxelSize=${lines[1]}
VOIs=${lines[2]}

patientFolder="${targetFolder}/${patientName}"
expFolder="${patientFolder}/FastDose"
if [ ! -d ${expFolder} ]; then
    mkdir ${expFolder}
fi

for ((segment=1; segment<=4; segment++)); do
    outputFolder="${expFolder}/doseMat${segment}"
    if [ ! -d ${outputFolder} ]; then
        mkdir ${outputFolder}
    fi
    logFile="${expFolder}/doseMat${segment}.log"
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
        --beamlist "${patientFolder}/beamlist${segment}.txt" \
        --mode 0 \
        --deviceIdx 2 \
        --spectrum "${dataFolder}/spec_6mv.spec" \
        --kernel "${dataFolder}/kernel_exp_6mv.txt" \
        --fluenceDim 20 \
        --subFluenceDim 16 \
        --outputFolder ${outputFolder} \
        --planFolder - \
        --nBeamsReserve 200 \
        --EstNonZeroElementsPerMat 10000000 \
    | tee ${logFile}
done