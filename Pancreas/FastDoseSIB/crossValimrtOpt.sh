#!bin/bash

exec="/data/qifan/projects/FastDose/build/bin/IMRT"
dataFolder="/data/qifan/projects/FastDose/scripts"
rootFolder="/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB"

for ((i=1; i<=5; i++)); do
    patientName="Patient00${i}"
    patientFolder="${rootFolder}/${patientName}"
    expFolder="${patientFolder}/FastDose"
    
    inputFolder="${patientFolder}/FastDose/prep_output"
    dimFile="${inputFolder}/dimension.txt"
    readarray -t lines < ${dimFile}
    phantomDim=${lines[0]}
    voxelSize=${lines[1]}
    VOIs=${lines[2]}

    crossValFolder="${patientFolder}/crossVal"

    logFile="${crossValFolder}/optimize.log"
    outputFolder="${crossValFolder}/doseMat_UHPP"
    referenceDose="${patientFolder}/doseNorm.bin"
    planFolder="${crossValFolder}/plan1_UHPP"
    if [ ! -d ${planFolder} ]; then
        mkdir ${planFolder}
    fi

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
        --beamlist "${expFolder}/beamlist.txt" \
        --mode 1 \
        --deviceIdx 3 \
        --spectrum "${dataFolder}/spec_6mv.spec" \
        --kernel "${dataFolder}/kernel_exp_6mv.txt" \
        --fluenceDim 20 \
        --subFluenceDim 16 \
        --outputFolder ${outputFolder} \
        --planFolder ${planFolder} \
        --nBeamsReserve 20 \
        --EstNonZeroElementsPerMat 6000000 \
        --SIB true \
        --referenceDose ${referenceDose} \
    | tee ${logFile}
done