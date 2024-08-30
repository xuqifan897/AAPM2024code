#!/bin/bash

export DOSECALC_DATA="/data/qifan/projects/BeamOpt/CCCS/data"
dosecalc_exe="/data/qifan/projects/BeamOpt/CCCS/build/dosecalc-beamlet/dosecalc-beamlet"
sourceFolder="/data/qifan/projects/FastDoseWorkplace/TCIAAdd"
targetFolder="${sourceFolder}/plansAngleCorrect"

Idx=$1
stride=4
deviceIdx=$(( $Idx % ${stride} ))
numPatients=8
patientList=(002 003 009 013 070 125 132 190)
while [ $Idx -lt ${numPatients} ]; do
    patient=${patientList[$Idx]}
    echo $patient
    for seg in 0 1 2 3; do
        folder="${targetFolder}/${patient}/QihuiRyan/preprocess${seg}"
        if [ ! -d ${folder} ]; then
            echo "${folder} doesn't exist"
            exit
        fi
        logFile="${folder}/dosecalc.log"
        cd ${folder}
        ( time ${dosecalc_exe} \
            --sparsity-threshold=${sparsity} \
            --ndevices=1 \
            --device=${deviceIdx} \
            --temp_dir=${folder} ) \
        2>&1 | tee ${logFile}
    done
    Idx=$(( ${Idx}+${stride} ))
done