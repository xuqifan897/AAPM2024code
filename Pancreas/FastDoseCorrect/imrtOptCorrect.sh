#!/bin/bash

exec="/data/qifan/projects/FastDose/build/bin/IMRT"
dataFolder="/data/qifan/projects/FastDose/scripts"
rootFolder="/data/qifan/projects/FastDoseWorkplace/Pancreas"

for ((i=2; i<=5; i++)); do
    patientName="Patient00${i}"
    patientFolder="${rootFolder}/plansAngleCorrect/${patientName}"
    expFolder="${patientFolder}/FastDose"
    planFolder="${expFolder}/plan1"
    if [ ! -d ${planFolder} ]; then
        mkdir ${planFolder}
    fi

    inputFolder="${rootFolder}/${patientName}/FastDose/prep_output"
    dimFile="${inputFolder}/dimension.txt"
    readarray -t lines < ${dimFile}
    phantomDim=${lines[0]}
    voxelSize=${lines[1]}
    VOIs=${lines[2]}

    logFile="${expFolder}/optimize.log"
    outputFolder="${expFolder}/doseMat1 ${expFolder}/doseMat2"

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
        --beamlist "${patientFolder}/beamlist.txt" \
        --mode 1 \
        --deviceIdx 3 \
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