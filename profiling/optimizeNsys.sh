#!bin/bash

exec="/data/qifan/projects/FastDose/build/bin/IMRT"
dataFolder="/data/qifan/projects/FastDose/scripts"
rootFolder="/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB"

resultFolder="/data/qifan/projects/FastDoseWorkplace/profiling"
planFolder="${resultFolder}/IMRTOpt"
if [ ! -d ${resultFolder} ]; then
    mkdir ${resultFolder}
fi
outputFile="${resultFolder}/IMRTOpt.nsys-rep"

patientFolder="/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB/Patient001"
FastDoseFolder="${patientFolder}/FastDose"
inputFolder="${FastDoseFolder}/prep_output"
dimFile="${inputFolder}/dimension.txt"
readarray -t lines < ${dimFile}
phantomDim=${lines[0]}
voxelSize=${lines[1]}
VOIs=${lines[2]}
doseMatFolder="${FastDoseFolder}/doseMat1 ${FastDoseFolder}/doseMat2"
referenceDose="${patientFolder}/doseNorm.bin"

OMP_NUM_THREADS=64 /usr/local/cuda/bin/nsys profile \
    --output ${outputFile} \
    ${exec} \
        --phantomDim ${phantomDim} \
        --voxelSize ${voxelSize} \
        --SAD 100 \
        --density "${inputFolder}/density.raw" \
        --structures ${VOIs} \
        --masks "${inputFolder}/roi_list.h5" \
        --primaryROI "ROI" \
        --bboxROI "SKIN" \
        --structureInfo "${FastDoseFolder}/StructureInfo.csv" \
        --params "${FastDoseFolder}/params.txt" \
        --beamlist "${FastDoseFolder}/beamlist.txt" \
        --mode 1 \
        --deviceIdx 3 \
        --spectrum "${dataFolder}/spec_6mv.spec" \
        --kernel "${dataFolder}/kernel_exp_6mv.txt" \
        --fluenceDim 20 \
        --subFluenceDim 16 \
        --outputFolder ${doseMatFolder} \
        --planFolder ${planFolder} \
        --nBeamsReserve 200 \
        --EstNonZeroElementsPerMat 6000000 \
        --SIB true \
        --referenceDose ${referenceDose}