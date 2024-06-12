#!/bin/bash

patient="190"

exec="/data/qifan/projects/FastDose/build/bin/IMRT"
dataFolder="/data/qifan/projects/FastDose/scripts"
RootFolder="/data/qifan/projects/FastDoseWorkplace/TCIASupp"

FastDoseFolder="${RootFolder}/${patient}/FastDose"
inputFolder="${FastDoseFolder}/prep_output"
planFolder="${FastDoseFolder}/plan1"
dimFile="${inputFolder}/dimension.txt"
readarray -t lines < ${dimFile}
phantomDim=${lines[0]}
voxelSize=${lines[1]}
VOIs=${lines[2]}

if [ ! -d ${inputFolder} ]; then
    echo "The folder ${inputFolder} doesn't exist."
fi

if [ ! -d ${planFolder} ]; then
    mkdir ${planFolder}
fi

for primaryROI in "PTVSeg0" "PTVSeg1" "PTVSeg2" "PTVSeg3"; do
    logFile="${FastDoseFolder}/dosecalc${primaryROI}.log"
    outputFolder="${FastDoseFolder}/doseMat${primaryROI}"
    if [ ! -d ${outputFolder} ]; then
        mkdir ${outputFolder}
    fi
    OMP_NUM_THREADS=64 ${exec} \
        --phantomDim ${phantomDim} \
        --voxelSize ${voxelSize} \
        --SAD 100 \
        --density "${inputFolder}/density.raw" \
        --structures ${VOIs} \
        --masks "${inputFolder}/roi_list.h5" \
        --primaryROI ${primaryROI} \
        --bboxROI "SKIN" \
        --structureInfo "-" \
        --params "-" \
        --beamlist "${RootFolder}/BeamList${primaryROI}.txt" \
        --mode 0 \
        --deviceIdx 2 \
        --spectrum "${dataFolder}/spec_6mv.spec" \
        --kernel "${dataFolder}/kernel_exp_6mv.txt" \
        --fluenceDim 20 \
        --subFluenceDim 16 \
        --outputFolder ${outputFolder} \
        --planFolder "-" \
        --nBeamsReserve 200 \
        --EstNonZeroElementsPerMat 12000000 \
    | tee ${logFile}
done