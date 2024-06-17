#!/bin/bash

PatientName="002"
exec="/data/qifan/projects/FastDose/build/bin/IMRT"
dataFolder="/data/qifan/projects/FastDose/scripts"
RootFolder="/data/qifan/projects/FastDoseWorkplace/TCIAAdd"
FastDoseFolder="${RootFolder}/${PatientName}/FastDose"

planNo=7
inputFolder="${FastDoseFolder}/prep_output"
planFolder="${FastDoseFolder}/plan${planNo}"

if [ ! -d ${inputFolder} ]; then
    echo "The folder ${inputFolder} doesn't exist."
fi
if [ ! -d ${planFolder} ]; then
    mkdir ${planFolder}
fi

outputFolders=""
for i in 0 1 2 3; do
    for j in 0 1; do
        LocalOutputFolder="${FastDoseFolder}/dosecalcSeg${i}Split${j}"
        if [ ! -d ${LocalOutputFolder} ]; then
            echo "The folder \"${LocalOutputFolder}\" doesn't exist."
        fi
        outputFolders+=" ${LocalOutputFolder}"
    done
done

dimFile="${inputFolder}/dimension.txt"
readarray -t lines < ${dimFile}
phantomDim=${lines[0]}
voxelSize=${lines[1]}
VOIs=${lines[2]}

logFile="${FastDoseFolder}/optimize.log"

${exec} \
    --phantomDim ${phantomDim} \
    --voxelSize ${voxelSize} \
    --SAD 100 \
    --density "${inputFolder}/density.raw" \
    --structures ${VOIs} \
    --masks "${inputFolder}/roi_list.h5" \
    --primaryROI "PTVMerge.bin" \
    --bboxROI "SKIN" \
    --structureInfo "${FastDoseFolder}/StructureInfo${planNo}.csv" \
    --params "${FastDoseFolder}/params${planNo}.txt" \
    --beamlist - \
    --mode 1 \
    --deviceIdx 1 \
    --spectrum "${dataFolder}/spec_6mv.spec" \
    --kernel "${dataFolder}/kernel_exp_6mv.txt" \
    --fluenceDim 20 \
    --subFluenceDim 16 \
    --outputFolder ${outputFolders} \
    --planFolder ${planFolder} \
    --nBeamsReserve 1280 \
    --EstNonZeroElementsPerMat 12000000 \
| tee ${logFile}

# tbreak /data/qifan/projects/FastDose/IMRTOpt/src/IMRTEigenLoad.cu:135