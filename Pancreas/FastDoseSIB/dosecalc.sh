#!/bin/bash
exec="/data/qifan/projects/FastDose/build/bin/IMRT"
dataFolder="/data/qifan/projects/FastDose/scripts"
targetFolder="/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB"

for ((i=1; i<=5; i++)); do
    patientName="Patient00${i}"
    patientFolder="${targetFolder}/${patientName}"
    FastDoseFolder="${patientFolder}/FastDose"
    inputFolder="${FastDoseFolder}/prep_output"
    dimFile="${inputFolder}/dimension.txt"
    readarray -t lines < ${dimFile}
    phantomDim=${lines[0]}
    voxelSize=${lines[1]}
    VOIs=${lines[2]}
    
    for ((segment=1; segment<=2; segment++)); do
        outputFolder="${FastDoseFolder}/doseMat${segment}"
        if [ ! -d ${outputFolder} ]; then
            mkdir ${outputFolder}
        fi
        logFile="${FastDoseFolder}/doseMat${segment}.log"
        OMP_NUM_THREADS=64 ${exec} \
            --phantomDim ${phantomDim} \
            --voxelSize ${voxelSize} \
            --SAD 100 \
            --density "${inputFolder}/density.raw" \
            --structures ${VOIs} \
            --masks "${inputFolder}/roi_list.h5" \
            --primaryROI "ROI" \
            --bboxROI "SKIN" \
            --structureInfo "${expFolder}/StructureInfo.csv" \
            --params "${expFolder}/params.txt" \
            --beamlist "${FastDoseFolder}/beamlist${segment}.txt" \
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