#!/bin/bash

export DOSECALC_DATA="/data/qifan/projects/BeamOpt/CCCS/data"
dosecalc_exe="/data/qifan/projects/BeamOpt/CCCS/build/dosecalc-beamlet/dosecalc-beamlet"
sourceFolder="/data/qifan/projects/FastDoseWorkplace/Pancreas"
targetFolder="${sourceFolder}/plansAngleCorrect"

voxelsize="0.25"  # [units: cm]
sparsity="1e-4"  # probably don't need to change ever
device=1

for ((i=1; i<=5; i++)); do
    patientName="Patient00${i}"
    targetPatientFolder="${targetFolder}/${patientName}"
    expFolder="${targetPatientFolder}/QihuiRyan"
    prepFolder="${expFolder}/preprocess"

    # dosecalc
    logFile="${expFolder}/dosecalc.log"
    cd ${expFolder}
    ( time ${dosecalc_exe} \
        --sparsity-threshold=${sparsity} \
        --ndevices=1 \
        --device=${device} \
        --temp_dir=${prepFolder}) \
    2>&1 | tee ${logFile}
done