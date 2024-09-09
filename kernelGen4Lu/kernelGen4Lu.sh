#!/bin/bash

exec="/data/qifan/projects/FastDose/build/bin/KernelGen"

# ${exec} \
#     --help

outputFolder="/data/qifan/projects/FastDoseWorkplace/kernelGen4Lu"
spectrumFile="${outputFolder}/spec_225kv_1mm.spec"
logFile="${outputFolder}/kernelGen.log"
nohup ${exec} \
    --outputFolder $outputFolder \
    --spectrumFile $spectrumFile \
    --nParticles 1000000000 \
    --radiusRes 0.005 \
    --heightRes 0.005 \
    --radiusDim 100 \
    --heightDim 400 \
    --marginTail 20 \
    --marginHead 100 \
    --logFreq 100000 \
    2>&1 > $logFile &