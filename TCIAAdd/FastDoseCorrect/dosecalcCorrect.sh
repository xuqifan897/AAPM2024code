#!/bin/bash

exec="/data/qifan/projects/FastDose/build/bin/IMRT"
dataFolder="/data/qifan/projects/FastDose/scripts"
sourceFolder="/data/qifan/projects/FastDoseWorkplace/TCIAAdd"
targetFolder="${sourceFolder}/plansAngleCorrect"

# for patientName in 002 003 009 013 070 125 132 190; do
# for patientName in 003 009 013 070 125 132 190; do
for patientName in 132; do
    sourcePatientFolder="${sourceFolder}/${patientName}"
    targetPatientFolder="${targetFolder}/${patientName}"
    inputFolder="${sourcePatientFolder}/FastDose/prep_output"
    dimFile="${inputFolder}/dimension.txt"
    readarray -t lines < ${dimFile}
    phantomDim=${lines[0]}
    voxelSize=${lines[1]}
    VOIs=${lines[2]}

    expFolder="${targetPatientFolder}/FastDose"
    if [ ! -d ${expFolder} ]; then
        mkdir ${expFolder}
    fi

    for ((segment=0; segment<4; segment++)); do
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
            --primaryROI "PTVSeg${segment}" \
            --bboxROI "SKIN" \
            --structureInfo "${expFolder}/StructureInfo1.csv" \
            --params "${expFolder}/params1.txt" \
            --beamlist "${targetPatientFolder}/beamlist${segment}.txt" \
            --mode 0 \
            --deviceIdx 1 \
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
done