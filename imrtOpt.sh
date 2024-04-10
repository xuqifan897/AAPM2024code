#!/bin/bash

execFolder="/data/qifan/projects/FastDose/build/bin"
inputFolder="/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_output"
outputFolder="/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_result_v1"
if [ ! -d ${outputFolder} ]; then
    mkdir ${outputFolder}
fi;
dimFile="${inputFolder}/dimension.txt"
readarray -t lines < ${dimFile}
phantomDim=${lines[0]}
voxelSize=${lines[1]}
VOIs=${lines[2]}

OMP_NUM_THREADS=64 ${execFolder}/IMRT \
    --phantomDim ${phantomDim} \
    --voxelSize ${voxelSize} \
    --SAD 100 \
    --density "${inputFolder}/density.raw" \
    --structures ${VOIs} \
    --masks "${inputFolder}/roi_list.h5" \
    --primaryROI "PTV_ENLARGED" \
    --bboxROI "Skin" \
    --structureInfo ./StructureInfo.csv \
    --params ./params.txt \
    --beamlist ./beamlist.txt \
    --mode 1 \
    --deviceIdx 0 \
    --spectrum ./spec_6mv.spec \
    --kernel ./kernel_exp_6mv.txt \
    --subFluenceDim 16 \
    --outputFolder ${outputFolder} \
    --nBeamsReserve 452