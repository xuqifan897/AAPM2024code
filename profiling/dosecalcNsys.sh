#/!bin/bash

exec="/data/qifan/projects/FastDose/build/bin/IMRT"
dataFolder="/data/qifan/projects/FastDose/scripts"

resultFolder="/data/qifan/projects/FastDoseWorkplace/profiling"
outputFolder="${resultFolder}/IMRT_dosecalc"
if [ ! -d ${outputFolder} ]; then
    mkdir ${outputFolder}
fi
outputFile="${resultFolder}/IMRT_dosecalc.qdrep"

patientFolder="/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB/Patient001"
FastDoseFolder="${patientFolder}/FastDose"
inputFolder="${FastDoseFolder}/prep_output"
dimFile="${inputFolder}/dimension.txt"
readarray -t lines < ${dimFile}
phantomDim=${lines[0]}
voxelSize=${lines[1]}
VOIs=${lines[2]}

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
        --structureInfo "${expFolder}/StructureInfo.csv" \
        --params "${expFolder}/params.txt" \
        --beamlist "${resultFolder}/beamlist.txt" \
        --mode 0 \
        --deviceIdx 0 \
        --spectrum "${dataFolder}/spec_6mv.spec" \
        --kernel "${dataFolder}/kernel_exp_6mv.txt" \
        --fluenceDim 20 \
        --subFluenceDim 16 \
        --outputFolder ${outputFolder} \
        --planFolder - \
        --nBeamsReserve 10 \
        --EstNonZeroElementsPerMat 10000000