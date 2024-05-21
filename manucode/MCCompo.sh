
ProjectFolder="/data/qifan/projects/FastDose"
exec="${ProjectFolder}/build/bin/cubeScore"

resultFolder="/data/qifan/projects/FastDoseWorkplace/DoseBench/composite"
if [ ! -d ${resultFolder} ]; then
    echo "The folder ${resultFolder} doesn't exist."
fi

logFile="${resultFolder}/MCWaterWidth5cm.log"
${exec} \
    --SpectrumFile "${ProjectFolder}/cubeScore/spectrum.csv" \
    --SlabPhantomFile "${ProjectFolder}/cubeScore/WaterPhantom.csv" \
    --MaterialFile "${ProjectFolder}/cubeScore/WaterMaterial.csv" \
    --OutputFile "${resultFolder}/MCWaterWidth5cm.bin" \
    --FluenceSize 2.5 \
    --nParticles 1000000000 \
    --logFreq 10000000 \
    | tee ${logFile}