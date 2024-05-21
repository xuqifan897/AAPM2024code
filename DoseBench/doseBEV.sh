#!/bin/bash

exec="/data/qifan/projects/FastDose/build/bin/singleBeamBEV"
dataFolder="/data/qifan/projects/FastDose/scripts"
rootFolder="/data/qifan/projects/FastDoseWorkplace/DoseBench/MCSuperSampling"

# phantom="slab"
phantom="water"

# width=0.5
# width=1.0
width=2.0

if [ ${phantom} == "slab" ]; then
    phantomDim="256 256 256"
    voxelSize="0.1 0.1 0.1"
    densityFile="${rootFolder}/slabDensity.raw"
elif [ ${phantom} == "water" ]; then
    phantomDim="100 100 100"
    voxelSize="0.25 0.25 0.25"
    densityFile="${rootFolder}/waterDensity.bin"
fi

if [ ${width} == 0.5 ]; then
    subFluenceRes=0.08333
    subFluenceDim=24
    subFluenceOn=6
elif [ ${width} == 1.0 ]; then
    subFluenceRes=0.125
    subFluenceDim=24
    subFluenceOn=8;
elif [ ${width} == 2.0 ]; then
    subFluenceRes=0.25
    subFluenceDim=24
    subFluenceOn=8
fi

outputFile="${rootFolder}/BEVDose_${phantom}_${width}.bin"

${exec} \
    --phantomDim ${phantomDim} \
    --voxelSize ${voxelSize} \
    --SAD 100.0 \
    --density ${densityFile} \
    --deviceIdx 2 \
    --spectrum "${dataFolder}/spec_6mv.spec" \
    --kernel "${dataFolder}/kernel_exp_6mv.txt" \
    --subFluenceRes ${subFluenceRes} \
    --subFluenceDim ${subFluenceDim} \
    --subFluenceOn ${subFluenceOn} \
    --outputFile ${outputFile}