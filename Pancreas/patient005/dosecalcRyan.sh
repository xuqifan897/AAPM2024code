#!/bin/bash
export DOSECALC_DATA="/data/qifan/projects/BeamOpt/CCCS/data"

dosecalc_exe="/data/qifan/projects/BeamOpt/CCCS/build/dosecalc-beamlet/dosecalc-beamlet"
expFolder="/data/qifan/projects/FastDoseWorkplace/Pancreas/Patient005/QihuiRyan"
temp_dir="${expFolder}/preprocess"
logFile="${expFolder}/dosecalc-beamlet.log"
sparsity='1e-4'
device=1

cd ${expFolder}
( time ${dosecalc_exe} \
    --sparsity-threshold=${sparsity} \
    --ndevices=1 \
    --device="${device}" \
    --temp_dir="${temp_dir}") \
    2>&1 | tee ${logFile}

# gdb --args ${dosecalc_exe} \
#     --sparsity-threshold=${sparsity} \
#     --ndevices=1 \
#     --device="${device}" \
#     --temp_dir="${temp_dir}" \
#     --verbose

# tbreak /data/qifan/projects/BeamOpt/CCCS/dosecalc-beamlet/main.cu:174