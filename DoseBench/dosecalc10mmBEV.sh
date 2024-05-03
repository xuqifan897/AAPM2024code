#!/bin/bash

exec="/data/qifan/projects/FastDose/build/bin/singleBeamBEV"
dataFolder="/data/qifan/projects/FastDose/scripts"
rootFolder="/data/qifan/projects/FastDoseWorkplace/DoseBench/water/prep_output"

${exec} \
    --phantomDim 100 100 100 \
    --voxelSize 0.25 0.25 0.25 \
    --SAD 100.0 \
    --density "${rootFolder}/density.raw" \
    --deviceIdx 1 \
    --spectrum "${dataFolder}/spec_6mv.spec" \
    --kernel "${dataFolder}/kernel_exp_6mv.txt" \
    --subFluenceRes 0.125 \
    --subFluenceDim 24 \
    --subFluenceOn 8 \
    --outputFile "/data/qifan/projects/FastDoseWorkplace/DoseBench/water/width10mm/BEVdose248.bin"