#!/bin/bash

sharedData="/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/sharedData"
# must match MIM structure names
BBox="Skin"  # bounding box - dose is calculated in this box (speedup runtime)
dicomdata="${sharedData}/data"
configfile="${PWD}/config.json"
beamlist="${PWD}/beamlist.txt"
structures="${PWD}/structures.json"

# Quality settings
voxelsize='0.25'  # [units: cm]
sparsity='1e-4'  # probably don't need to change ever

export DOSECALC_DATA="/data/qifan/projects/BeamOpt/CCCS/data"

expFolder="/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_bench"
if [ ! -d ${expFolder} ]; then
    mkdir ${expFolder}
fi
preprocess_exe="/data/qifan/projects/BeamOpt/CCCS/build/dosecalc-preprocess/dosecalc-preprocess"
dosecalc_exe="/data/qifan/projects/BeamOpt/CCCS/build/dosecalc-beamlet/dosecalc-beamlet"
cd ${expFolder}

device=1

# call preprocess, save a log of the output automatically
( time ${preprocess_exe} \
    --dicom=${dicomdata} \
    --beamlist=${beamlist} \
    --structures=${structures} \
    --config=${configfile} \
    --bbox-roi=${BBox} \
    --voxsize=${voxelsize} \
    --device=${device} \
    --verbose ) \
    2>&1 | tee "dosecalc-preprocess.log"

echo -e "\n\n=================================================================================\n\n"

# call dosecalc-beamlet, save a log of the output automatically
( time ${dosecalc_exe} \
    --sparsity-threshold=${sparsity} \
    --ndevices=1 \
    --device=${device}) \
    2>&1 | tee "dosecalc-beamlet.log"

echo -e "\n\n=================================================================================\n\n"