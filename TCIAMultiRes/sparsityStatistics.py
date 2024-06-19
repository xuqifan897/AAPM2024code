import os
import numpy as np
import matplotlib.pyplot as plt

threshold = 0.5

def showStat():
    valueBuffer = "/data/qifan/projects/FastDoseWorkplace/TCIAAdd/002" \
        "/FastDose/dosecalcSeg0Split0/doseMatFolder/valuesBuffer.bin"
    valueBuffer = np.fromfile(valueBuffer, dtype=np.float32)
    np.random.shuffle(valueBuffer)

    nSamples = 10000
    valueBuffer = valueBuffer[:nSamples]
    valueBuffer = np.sort(valueBuffer)
    xAxis = np.arange(nSamples)
    plt.plot(xAxis, valueBuffer)
    plt.xlabel("# sample")
    plt.ylabel("dose value")
    image = "/data/qifan/projects/AAPM2024/TCIASpar/sparseStat.png"
    plt.title("Voxel dose statistics")
    plt.savefig(image)
    plt.clf()

    content = "| dose | percentile |\n|-|-|"
    idx = 0
    dose_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    for i in dose_values:
        while True:
            if idx == nSamples:
                break
            if valueBuffer[idx] >= i:
                percentile = idx / nSamples
                newLine = "| {} | {} |".format(i, percentile)
                content = content + "\n" + newLine
                idx += 1
                break
            idx += 1
    file = "/data/qifan/projects/AAPM2024/TCIASpar/sparseStat2.md"
    with open(file, "w") as f:
        f.write(content)


def evalThresh():
    """
    This function evaluates the effect of different sparsification thresholds
    """
    patientFolder = "/data/qifan/projects/FastDoseWorkplace/TCIAAdd/002"
    doseMatFolder = os.path.join(patientFolder, "FastDose", "dosecalcSeg0Split0", "doseMatFolder")

    metadata = os.path.join(patientFolder, "metadata.txt")
    with open(metadata, "r") as f:
        metadata = f.readline()
    dimension = metadata.replace(" ", ", ")
    dimension = eval(dimension)
    nVoxels = dimension[0] * dimension[1] * dimension[2]

    offsetsBuffer = os.path.join(doseMatFolder, "offsetsBuffer.bin")
    columnsBuffer = os.path.join(doseMatFolder, "columnsBuffer.bin")
    valuesBuffer = os.path.join(doseMatFolder, "valuesBuffer.bin")
    NonZeroElements = os.path.join(doseMatFolder, "NonZeroElements.bin")
    numRowsPerMat = os.path.join(doseMatFolder, "numRowsPerMat.bin")

    offsetsBuffer = np.fromfile(offsetsBuffer, dtype=np.uint64)
    columnsBuffer = np.fromfile(columnsBuffer, dtype=np.uint64)
    valuesBuffer = np.fromfile(valuesBuffer, dtype=np.float32)
    NonZeroElements = np.fromfile(NonZeroElements, dtype=np.uint64)
    numRowsPerMat = np.fromfile(numRowsPerMat, dtype=np.uint64)

    numMatrices = numRowsPerMat.size
    assert numMatrices == NonZeroElements.size

    resultFolder = os.path.join(patientFolder, "beamDose_thresh{}".format(threshold))
    if not os.path.isdir(resultFolder):
        os.mkdir(resultFolder)

    offsetsIdx = 0
    columnsIdx = 0
    for idx in range(numMatrices):
        # early stop
        if idx == 10:
            break

        localRows = numRowsPerMat[idx]
        localNNZ = NonZeroElements[idx]

        offsetsIdxEnd = int(offsetsIdx + localRows + 1)
        localOffsets = offsetsBuffer[offsetsIdx: offsetsIdxEnd]

        columnsIdxEnd = int(columnsIdx + localNNZ)
        localColumns = columnsBuffer[columnsIdx: columnsIdxEnd]
        localValues = valuesBuffer[columnsIdx: columnsIdxEnd].copy()

        # apply pruning
        localValues[localValues < threshold] = 0

        offsetsIdx = offsetsIdxEnd
        columnsIdx = columnsIdxEnd

        shape = (localRows, nVoxels)
        resultMat = np.zeros(shape, dtype=np.float32)

        for j in range(localRows):
            idxBegin = localOffsets[j]
            idxEnd = localOffsets[j + 1]
            for k in range(idxBegin, idxEnd):
                column_value = localColumns[k]
                value_value = localValues[k]
                resultMat[j, column_value] = value_value
        
        resultMat = np.sum(resultMat, axis=0)
        resultMat = np.reshape(resultMat, dimension)
        matFile = os.path.join(resultFolder, "doseBeam{:03d}.npy".format(idx))
        np.save(matFile, resultMat)
        print(matFile)


def beamDoseView():
    patientFolder = "/data/qifan/projects/FastDoseWorkplace/TCIAAdd/002"
    beamDoseFolder = os.path.join(patientFolder, "beamDose_thresh{}".format(threshold))
    doseViewFolder = os.path.join(patientFolder, "beamDoseView_thresh{}".format(threshold))
    if not os.path.isdir(doseViewFolder):
        os.mkdir(doseViewFolder)

    metadata = os.path.join(patientFolder, "metadata.txt")
    with open(metadata, "r") as f:
        metadata = f.readline()
    dimension = metadata.replace(" ", ", ")
    dimension = eval(dimension)

    densityArrayFile = os.path.join(patientFolder, "density_raw.bin")
    densityArray = np.fromfile(densityArrayFile, dtype=np.uint16)
    densityArray = np.reshape(densityArray, dimension)

    PTVMaskFile = os.path.join(patientFolder, "PlanMask", "PTV70.bin")
    PTVMask = np.fromfile(PTVMaskFile, dtype=np.uint8)
    PTVMask = PTVMask.astype(bool)
    PTVMask = np.reshape(PTVMask, dimension)
    
    for i in range(10):
        file = os.path.join(beamDoseFolder, "doseBeam{:03d}.npy".format(i))
        doseArray = np.load(file)

        figureFolder = os.path.join(doseViewFolder, "doseView{:03d}".format(i))
        if not os.path.isdir(figureFolder):
            os.mkdir(figureFolder)
        for j in range(doseArray.shape[0]):
            plt.imshow(densityArray[j, :, :], cmap="gray", vmin=500, vmax=1500)
            plt.imshow(doseArray[j, :, :], cmap="jet", alpha=0.3)
            plt.colorbar()
            figureFile = os.path.join(figureFolder, "{:03d}.png".format(j))
            plt.savefig(figureFile)
            plt.clf()
            print(figureFile)


if __name__ == "__main__":
    # showStat()
    # evalThresh()
    beamDoseView()