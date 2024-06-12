#!/bin/bash

SourceRoot="/data/qifan/projects/FastDoseWorkplace/TCIASupp"
TargetRoot="/data/qifan/projects/FastDoseWorkplace/TCIAAdd"
patients="002 003 009 013 070 125 132 190"
for patient in ${patients}; do
    SourceFile1="${SourceRoot}/${patient}/dose.bin"
    SourceFile2="${SourceRoot}/${patient}/metadata.txt"
    TargetFolder="${TargetRoot}/${patient}"
    cp ${SourceFile1} ${TargetFolder}
    cp ${SourceFile2} ${TargetFolder}
done