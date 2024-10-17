#!/bin/bash

# This script is to test how much speedup would it be
# to set the number of sparsification threads to 32

export DOSECALC_DATA="/data/qifan/projects/BeamOpt/CCCS/data"
dosecalc_exe="/data/qifan/projects/BeamOpt/CCCS/build/dosecalc-beamlet/dosecalc-beamlet"
sparsity="1e-4"

folder="/data/qifan/projects/FastDoseWorkplace/profiling/Ryan_dosecalc_sr32"
for seg in 0 1 2 3; do
    subFolder="${folder}/preprocess${seg}"
    cd ${subFolder}
    logFile="${subFolder}/dosecalc.log"
    ( time ${dosecalc_exe} \
        --sparsity-thresh=${sparsity} \
        --ndevices=1 \
        --device=2 \
        --temp_dir=${subFolder} \
        --srworkers=32 ) \
    2>&1 | tee ${logFile}
done