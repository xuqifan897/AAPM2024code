#!/bin/bash

exec="/data/qifan/projects/FastDose/build/bin/IMRT"
dataFolder="/data/qifan/projects/FastDose/scripts"
expFolder="/data/qifan/projects/FastDoseWorkplace/Pancreas/Patient005/FastDose"

inputFolder="${expFolder}/prep_output"
planFolder="${expFolder}/plan1"

if [ ! -d ${inputFolder} ]; then
    echo "The folder ${inputFolder} doesn't exist."
fi

if [ ! -d ${planFolder} ]; then
    mkdir ${planFolder}
fi



for segment in 1 2; do
    outputFolder="${expFolder}/doseMat${segment}"
    if [ ! -d ${outputFolder} ]; then
        mkdir ${outputFolder}
    fi

    dimFile="${inputFolder}/dimension.txt"
    readarray -t lines < ${dimFile}
    phantomDim=${lines[0]}
    voxelSize=${lines[1]}
    VOIs=${lines[2]}

    logFile="${expFolder}/dosecalc${segment}.log"

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
        --beamlist "${expFolder}/beamlist${segment}.txt" \
        --mode 0 \
        --deviceIdx 1 \
        --spectrum "${dataFolder}/spec_6mv.spec" \
        --kernel "${dataFolder}/kernel_exp_6mv.txt" \
        --fluenceDim 20 \
        --subFluenceDim 16 \
        --outputFolder ${outputFolder} \
        --planFolder ${planFolder} \
        --nBeamsReserve 250 \
        --EstNonZeroElementsPerMat 10000000 \
    | tee ${logFile}
done