#!/bin/bash

exec="/data/qifan/projects/FastDose/build/bin/IMRT"
dataFolder="/data/qifan/projects/FastDose/scripts"
rootFolder="/data/qifan/projects/FastDoseWorkplace/Pancreas"

# for ((i=1; i<=5; i++)); do
for i in 2 3 4 5; do
    expFolder="${rootFolder}/Patient00${i}/FastDoseCorrect"
    planFolder="${expFolder}/plan1"
    if [ ! -d ${planFolder} ]; then
        mkdir ${planFolder}
    fi
    inputFolder="${expFolder}/prep_output"

    dimFile="${inputFolder}/dimension.txt"
    readarray -t lines < ${dimFile}
    phantomDim=${lines[0]}
    voxelSize=${lines[1]}
    VOIs=${lines[2]}

    logFile="${expFolder}/optimize.log"
    outputFolder="${expFolder}/doseMat1 ${expFolder}/doseMat2 ${expFolder}/doseMat3 ${expFolder}/doseMat4"

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
        --beamlist "${expFolder}/beamlist$.txt" \
        --mode 1 \
        --deviceIdx 2 \
        --spectrum "${dataFolder}/spec_6mv.spec" \
        --kernel "${dataFolder}/kernel_exp_6mv.txt" \
        --fluenceDim 20 \
        --subFluenceDim 16 \
        --outputFolder ${outputFolder} \
        --planFolder ${planFolder} \
        --nBeamsReserve 200 \
        --EstNonZeroElementsPerMat 6000000 \
    | tee ${logFile}
done