#!/bin/bash

exec="/data/qifan/projects/FastDose/build/bin/boxScore"
resultFolder="/data/qifan/projects/FastDoseWorkplace/DoseBench/slab/MC5mm"
beamletSize=0.25

for ((i=0; i<16; i++)); do
    ${exec} \
    ./build/bin/boxScore \
    --nParticles 10000000 \
    --dimXY 99 \
    --beamlet-size ${beamletSize} \
    --resultFolder ${resultFolder} \
    --iteration ${i}
done