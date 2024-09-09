#!/bin/bash

export DOSECALC_DATA="/data/qifan/projects/BeamOpt/CCCS/data"
dosecalc_exe="/data/qifan/projects/BeamOpt/CCCS/build/dosecalc-beamlet/dosecalc-beamlet"
sourceFolder="/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB"

voxelsize="0.25"  # [units: cm]
sparsity="1e-4"  # probably don't need to change ever
device=1
stride=4
nPatients=5
deviceIdx=$1

for ((idx=(($1+1)); idx<=5; idx=${idx}+${stride})); do
    patientName="Patient00$idx"
    sourcePatientFolder="${sourceFolder}/${patientName}"
    expFolder="${sourcePatientFolder}/QihuiRyan"
    prepFolder="${expFolder}/preprocess"
    logFile="${expFolder}/preprocess.log"

    cd ${expFolder}
    ( time ${dosecalc_exe} \
        --sparsity-threshold=${sparsity} \
        --ndevices=1 \
        --device=${deviceIdx} \
        --temp_dir=${prepFolder}) \
    2>&1 | tee ${logFile}
done