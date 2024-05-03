#!/bin/bash

exec="/data/qifan/projects/FastDose/build/bin/singleBeamBEV"
dataFolder="/data/qifan/projects/FastDose/scripts"
densityFile="/data/qifan/projects/FastDoseWorkplace/DoseBench/slab/slabDensity.raw"

${exec} \
    --phantomDim 256 256 256 \
    --voxelSize 0.1 0.1 0.1 \
    --SAD 100.0 \
    --density ${densityFile} \
    --deviceIdx 1 \
    --spectrum "${dataFolder}/spec_6mv.spec" \
    --kernel "${dataFolder}/kernel_exp_6mv.txt" \
    --subFluenceRes 0.25 \
    --subFluenceDim 24 \
    --subFluenceOn 8 \
    --outputFile "/data/qifan/projects/FastDoseWorkplace/DoseBench/slab/width20mm/BEVdose248.bin"