#!/bin/bash

ProjectFolder="/data/qifan/projects/FastDose"
exec="${ProjectFolder}/build/bin/cubeScore"

resultFolder="/data/qifan/projects/FastDoseWorkplace/DoseBench/MCSuperSampling"
if [ ! -d ${resultFolder} ]; then
    mkdir ${resultFolder}
fi

# phantom="slab"
phantom="water"

if [[ ${phantom} == "slab" ]]; then
    for width in 0.5 1.0 2.0; do
        halfWidth=$(echo "scale=2; ${width} / 2" | bc)
        logFile="${resultFolder}/slabWidth_${width}.log"
        ${exec} \
            --SpectrumFile "${ProjectFolder}/cubeScore/spectrum.csv" \
            --SlabPhantomFile "${ProjectFolder}/cubeScore/SlabPhantom.csv" \
            --MaterialFile "${ProjectFolder}/cubeScore/material.csv" \
            --OutputFile "${resultFolder}/MCDose_width_${width}cm.bin" \
            --FluenceSize ${halfWidth} \
            --nParticles 1000000000 \
            --logFreq 10000000 \
            | tee ${logFile}
    done
elif [[ ${phantom} == "water" ]]; then
    for width in 0.5 1.0 2.0; do
        halfWidth=$(echo "scale=2; ${width} / 2" | bc)
        logFile="${resultFolder}/waterWidth_${width}.log"
        ${exec} \
            --SpectrumFile "${ProjectFolder}/cubeScore/spectrum.csv" \
            --SlabPhantomFile "${ProjectFolder}/cubeScore/WaterPhantom.csv" \
            --MaterialFile "${ProjectFolder}/cubeScore/WaterMaterial.csv" \
            --OutputFile "${resultFolder}/MCDoseWater_width_${width}cm.bin" \
            --FluenceSize ${halfWidth} \
            --nParticles 1000000000 \
            --logFreq 10000000 \
            | tee ${logFile}
    done
fi