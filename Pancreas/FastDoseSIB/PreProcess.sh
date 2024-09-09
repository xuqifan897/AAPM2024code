#!/bin/bash
exec="/data/qifan/projects/FastDose/build/bin/preprocess"
rootFolder="/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB"
dimensions=("220 220 160" "200 200 182" "280 280 111" "240 240 155" "200 200 128")
numPatients=5
for ((i=0; i<${numPatients}; i++)); do
    # echo ${dimensions[$i]}
    patientName="Patient00$(($i+1))"
    patientFolder="${rootFolder}/${patientName}"
    FastDoseFolder="${patientFolder}/FastDose"
    structuresFile="${FastDoseFolder}/structures.json"
    InputFolder="${FastDoseFolder}/prep_output"
    maskFolder="${patientFolder}/InputMask"
    phantomPath="${patientFolder}/density_raw.bin"

    if [ ! -d ${FastDoseFolder} ]; then
        mkdir ${FastDoseFolder}
    fi
    if [ ! -d ${InputFolder} ]; then
        mkdir ${InputFolder}
    fi

    ${exec} \
        --mode 1 \
        --structuresFile ${structuresFile} \
        --ptv_name "ROI" \
        --bbox_name "SKIN" \
        --voxelSize 0.25 \
        --inputFolder ${InputFolder} \
        --shape ${dimensions[$i]} \
        --phantomPath ${phantomPath} \
        --RescaleSlope 1.0 \
        --RescaleIntercept -1000 \
        --maskFolder ${maskFolder}
done