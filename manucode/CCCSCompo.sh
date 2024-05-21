#/bin/bash

exec="/data/qifan/projects/FastDose/build/bin/IMRT"
dataFolder="/data/qifan/projects/FastDose/scripts"
outputFolder="/data/qifan/projects/FastDoseWorkplace/DoseBench/composite"
inputFolder="${outputFolder}/prep_result"

if [ ! -d ${inputFolder} ]; then
    echo "The folder ${inputFolder} doesn't exist."
fi

if [ ! -d ${outputFolder} ]; then
    echo "The folder ${outputFolder} doesn't exist."
fi

dimFile="${inputFolder}/dimension.txt"
readarray -t lines < ${dimFile}
phantomDim=${lines[0]}
voxelSize=${lines[1]}
VOIs=${lines[2]}

${exec} --help

OMP_NUM_THREADS=64 ${exec} \
    --phantomDim ${phantomDim} \
    --voxelSize ${voxelSize} \
    --SAD 100 \
    --density "${inputFolder}/density.raw" \
    --structures ${VOIs} \
    --masks "${inputFolder}/roi_list.h5" \
    --primaryROI "PTV" \
    --bboxROI "SKIN" \
    --structureInfo "-" \
    --params "-" \
    --beamlist "${outputFolder}/beamlist.txt" \
    --mode 0 \
    --deviceIdx 2 \
    --spectrum "${dataFolder}/spec_6mv.spec" \
    --kernel "${dataFolder}/kernel_exp_6mv.txt" \
    --fluenceDim 10 \
    --subFluenceDim 16 \
    --subFluenceOn 4 \
    --longSpacing 0.25 \
    --outputFolder ${outputFolder} \
    --planFolder "-" \
    --nBeamsReserve 1 \
    --EstNonZeroElementsPerMat 6000000


