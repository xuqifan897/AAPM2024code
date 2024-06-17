#!/bin/bash

# exec="/data/qifan/projects/FastDose/build/bin/preprocess"
# rootFolder="/data/qifan/projects/FastDoseWorkplace/TCIAAdd"
# patients="002 003 009 013 070 125 132 190"
# for patient in ${patients}; do
#     PatientFolder="${rootFolder}/${patient}"
#     FastDoseFolder="${PatientFolder}/FastDose"
#     structuresFile="${FastDoseFolder}/structures.json"
#     InputFolder="${FastDoseFolder}/prep_output"
#     maskFolder="${PatientFolder}/PlanMask"

#     if [ ! -d ${FastDoseFolder} ]; then
#         mkdir ${FastDoseFolder}
#     fi
#     if [ ! -d ${InputFolder} ]; then
#         mkdir ${InputFolder}
#     fi

#     ShapeFile="${PatientFolder}/metadata.txt"
#     readarray -t lines < ${ShapeFile}
#     phantomDim="${lines[0]}"
#     read -a dim_array <<< "${phantomDim}"
#     reversed_array=()
#     for ((i=${#dim_array[@]}-1; i>=0; i--)); do
#         reversed_array+=("${dim_array[i]}")
#     done
#     reversed_phantomDim="${reversed_array[*]}"

#     phantomPath="${PatientFolder}/density_raw.bin"

#     ${exec} \
#         --mode 1 \
#         --structuresFile ${structuresFile} \
#         --ptv_name "PTVMerge" \
#         --bbox_name "SKIN" \
#         --voxelSize 0.25 \
#         --inputFolder ${InputFolder} \
#         --shape ${reversed_phantomDim} \
#         --phantomPath ${phantomPath} \
#         --RescaleSlope 1.0 \
#         --RescaleIntercept -1024.0 \
#         --maskFolder ${maskFolder}
# done


exec="/data/qifan/projects/FastDose/build/bin/preprocess"
rootFolder="/data/qifan/projects/FastDoseWorkplace/TCIAAdd"
# patients="002 003 009 013 070 125 132 190"

patient="009"
PatientFolder="${rootFolder}/${patient}"
FastDoseFolder="${PatientFolder}/FastDose"
structuresFile="${FastDoseFolder}/structuresCrop.json"
InputFolder="${FastDoseFolder}/prep_output"
maskFolder="${PatientFolder}/PlanMask"

if [ ! -d ${FastDoseFolder} ]; then
    mkdir ${FastDoseFolder}
fi
if [ ! -d ${InputFolder} ]; then
    mkdir ${InputFolder}
fi

ShapeFile="${PatientFolder}/metadata.txt"
readarray -t lines < ${ShapeFile}
phantomDim="${lines[0]}"
read -a dim_array <<< "${phantomDim}"
reversed_array=()
for ((i=${#dim_array[@]}-1; i>=0; i--)); do
    reversed_array+=("${dim_array[i]}")
done
reversed_phantomDim="${reversed_array[*]}"

phantomPath="${PatientFolder}/density_raw.bin"

${exec} \
    --mode 1 \
    --structuresFile ${structuresFile} \
    --ptv_name "PTVMerge" \
    --bbox_name "SKIN" \
    --voxelSize 0.25 \
    --inputFolder ${InputFolder} \
    --shape ${reversed_phantomDim} \
    --phantomPath ${phantomPath} \
    --RescaleSlope 1.0 \
    --RescaleIntercept -1024.0 \
    --maskFolder ${maskFolder}