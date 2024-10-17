#!/bin/bash

export DOSECALC_DATA="/data/qifan/projects/BeamOpt/CCCS/data"
dosecalc_exe="/data/qifan/projects/BeamOpt/CCCS/build/dosecalc-beamlet/dosecalc-beamlet"
sourceFolder="/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB"

voxelsize="0.25"  # [units: cm]
sparsity="1e-4"  # probably don't need to change ever
device=1
stride=4
nPatients=5
deviceIdx=1

resultFolder="/data/qifan/projects/FastDoseWorkplace/profiling"
resultFile="${resultFolder}/Ryan_dosecalc_sr4.nsys-rep"
workFolder="${resultFolder}/Ryan_dosecalc"

cd ${workFolder}
/usr/local/cuda/bin/nsys profile \
    --output ${resultFile} \
    ${dosecalc_exe} \
        --sparsity-threshold=${sparsity} \
        --ndevices=1 \
        --device=0 \
        --temp_dir=${workFolder} \
        --srworkers=4

if false; then
    cuda-gdb --args ${dosecalc_exe} \
        --sparsity-threshold=${sparsity} \
        --ndevices=1 \
        --device=0 \
        --temp_dir=${workFolder} \
        --srworkers=4

    # tbreak /data/qifan/projects/BeamOpt/CCCS/dosecalc-beamlet/main.cu:260
fi