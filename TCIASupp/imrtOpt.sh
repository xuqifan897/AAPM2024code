#!/bin/bash

PatientName="190"
exec="/data/qifan/projects/FastDose/build/bin/IMRT"
dataFolder="/data/qifan/projects/FastDose/scripts"
RootFolder="/data/qifan/projects/FastDoseWorkplace/TCIASupp"
FastDoseFolder="${RootFolder}/${PatientName}/FastDose"

inputFolder="${FastDoseFolder}/prep_output"
planFolder="${FastDoseFolder}/plan1"

if [ ! -d ${inputFolder} ]; then
    echo "The folder ${inputFolder} doesn't exist."
fi
if [ ! -d ${planFolder} ]; then
    mkdir ${planFolder}
fi

outputFolders=""
for i in 0 1 2 3; do
    LocalOutputFolder="${FastDoseFolder}/doseMatPTVSeg${i}"
    if [ ! -d ${LocalOutputFolder} ]; then
        echo "The folder \"${LocalOutputFolder}\" doesn't exist."
        break
    fi
    outputFolders+=" ${LocalOutputFolder}"
done

dimFile="${inputFolder}/dimension.txt"
readarray -t lines < ${dimFile}
phantomDim=${lines[0]}
voxelSize=${lines[1]}
VOIs=${lines[2]}

logFile="${FastDoseFolder}/optimize.log"

OMP_NUM_THREADS=64 ${exec} \
    --phantomDim ${phantomDim} \
    --voxelSize ${voxelSize} \
    --SAD 100 \
    --density "${inputFolder}/density.raw" \
    --structures ${VOIs} \
    --masks "${inputFolder}/roi_list.h5" \
    --primaryROI "PTVMerge.bin" \
    --bboxROI "SKIN" \
    --structureInfo "${FastDoseFolder}/StructureInfo.csv" \
    --params "${FastDoseFolder}/params.txt" \
    --beamlist - \
    --mode 1 \
    --deviceIdx 3 \
    --spectrum "${dataFolder}/spec_6mv.spec" \
    --kernel "${dataFolder}/kernel_exp_6mv.txt" \
    --fluenceDim 20 \
    --subFluenceDim 16 \
    --outputFolder ${outputFolders} \
    --planFolder ${planFolder} \
    --nBeamsReserve 3144 \
    --EstNonZeroElementsPerMat 12000000 \
| tee ${logFile}