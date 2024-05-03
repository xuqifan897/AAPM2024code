import os
import numpy as np
import json
import matplotlib.pyplot as plt

def waterPhantomGen():
    """This function generates the water phantom"""

    if True:
        # show example
        exampleFile = "/data/qifan/projects/FastDoseWorkplace/Pancreas/Patient001/density_raw.bin"
        exampleArray = np.fromfile(exampleFile, dtype=np.uint16)
        print(np.min(exampleArray), np.max(exampleArray))
    
    # let the size of the water phantom be (x, y, z): (25cm, 25cm, 25cm)
    # let the resolution be: (0.25cm, 0.25cm, 0.25cm)
    # dimension: (100, 100, 100)

    shape = (100, 100, 100)
    densityArray = np.ones(shape, dtype=np.uint16) * 1000
    densityFile = "/data/qifan/projects/FastDoseWorkplace/DoseBench/water/waterDensity.bin"
    densityArray.tofile(densityFile)

    inputMaskFolder = "/data/qifan/projects/FastDoseWorkplace/DoseBench/water/InputMask"
    if not os.path.isdir(inputMaskFolder):
        os.mkdir(inputMaskFolder)
    mask = np.ones(shape, dtype=np.uint8)
    PTVFile = os.path.join(inputMaskFolder, "PTV.bin")
    mask.tofile(PTVFile)
    SKINFile = os.path.join(inputMaskFolder, "SKIN.bin")
    mask.tofile(SKINFile)


def structuresFileGen():
    content = {
        "prescription": 20,
        "ptv": "PTV",
        "oar": ["SKIN"]
    }
    content = json.dumps(content, indent=4)
    file = "/data/qifan/projects/FastDoseWorkplace/DoseBench/water/structures.json"
    with open(file, "w") as f:
        f.write(content)


def doseInterpret5mm():
    expFolder = "/data/qifan/projects/FastDoseWorkplace/DoseBench/water/width5mm"
    doseFolder = os.path.join(expFolder, "doseMat", "doseMatFolder")
    offsetsBufferFile = os.path.join(doseFolder, "offsetsBuffer.bin")
    columnsBufferFile = os.path.join(doseFolder, "columnsBuffer.bin")
    valuesBufferFile = os.path.join(doseFolder, "valuesBuffer.bin")
    nBeams = 81
    phantomDim = (100, 100, 100)
    offsets = np.fromfile(offsetsBufferFile, dtype=np.uint64)
    columns = np.fromfile(columnsBufferFile, dtype=np.uint64)
    values = np.fromfile(valuesBufferFile, dtype=np.float32)

    phantomNumElements = phantomDim[0] * phantomDim[1] * phantomDim[2]
    doseArrayShape = (nBeams, phantomNumElements)
    doseArray = np.zeros(doseArrayShape, dtype=np.float32)

    assert offsets.size == nBeams + 1, "Something wrong!"
    for i in range(nBeams):
        idx_start = offsets[i]
        idx_end = offsets[i+1]
        for j in range(idx_start, idx_end):
            column_idx = columns[j]
            value = values[j]
            doseArray[i, column_idx] = value
    
    doseViewFolder = os.path.join(expFolder, "doseView")
    if not os.path.isdir(doseViewFolder):
        os.mkdir(doseViewFolder)
    for i in range(nBeams):
        doseMat = doseArray[i, :]
        doseMat = np.reshape(doseMat, phantomDim)
        doseMatSum = np.sum(doseMat, axis=0)
        doseFile = os.path.join(doseViewFolder, "beam{:03d}.png".format(i))
        plt.imsave(doseFile, doseMatSum)
        print(doseFile)


def showCentralBeam():
    expFolder = "/data/qifan/projects/FastDoseWorkplace/DoseBench/water/width5mm"
    doseFolder = os.path.join(expFolder, "doseMat", "doseMatFolder")
    offsetsBufferFile = os.path.join(doseFolder, "offsetsBuffer.bin")
    columnsBufferFile = os.path.join(doseFolder, "columnsBuffer.bin")
    valuesBufferFile = os.path.join(doseFolder, "valuesBuffer.bin")
    nBeams = 81
    phantomDim = (100, 100, 100)
    offsets = np.fromfile(offsetsBufferFile, dtype=np.uint64)
    columns = np.fromfile(columnsBufferFile, dtype=np.uint64)
    values = np.fromfile(valuesBufferFile, dtype=np.float32)
    centralIdx = int((nBeams - 1) / 2)

    phantomNumElements = phantomDim[0] * phantomDim[1] * phantomDim[2]
    centralBeamDoseMat = np.zeros(phantomNumElements, dtype=np.float32)
    idx_begin = offsets[centralIdx]
    idx_end = offsets[centralIdx+1]
    for i in range(idx_begin, idx_end):
        column_idx = columns[i]
        value = values[i]
        centralBeamDoseMat[column_idx] = value
    centralBeamDoseMat = np.reshape(centralBeamDoseMat, phantomDim)

    centerline = centralBeamDoseMat[50, :, 50]
    depth = np.linspace(0, 25, 100)
    plt.plot(depth, centerline)
    plt.xlabel("depth (cm)")
    plt.ylabel("dose (a.u.)")
    figureFile = os.path.join(expFolder, "depthDose5mm.png")
    plt.savefig(figureFile)
    plt.clf()


def showBEVProfile():
    width = "20mm"
    workFolder = "/data/qifan/projects/FastDoseWorkplace/DoseBench/water/width{}".format(width)

    if width == "5mm":
        doseFile = os.path.join(workFolder, "BEVdose246.bin")
        doseShape = (100, 24, 24)
        res = (0.25, 0.08333, 0.08333)
    elif width == "10mm":
        doseFile = os.path.join(workFolder, "BEVdose248.bin")
        doseShape = (100, 24, 24)
        res = (0.25, 0.125, 0.125)
    elif width == "20mm":
        doseFile = os.path.join(workFolder, "BEVdose248.bin")
        doseShape = (100, 24, 24)
        res = (0.25, 0.25, 0.25)

    doseArray = np.fromfile(doseFile, dtype=np.float32)
    doseArray = np.reshape(doseArray, doseShape)
    doseArray /= np.max(doseArray)

    centerIdx = int(doseShape[1] / 2)
    centerline = doseArray[:, centerIdx, centerIdx].copy()
    centerline *= 100

    if width == "20mm":
        centerline[55] = (centerline[54] + centerline[56]) / 2

    depthDose = np.linspace(0, 25, doseShape[0])
    plt.plot(depthDose, centerline)
    plt.xlabel("depth (cm)")
    plt.ylabel("dose (normalized)")
    plt.title("Percent depth dose at beam width {}".format(width))
    figureFile = os.path.join(workFolder, "depthDose{}.png".format(width))
    plt.savefig(figureFile)
    plt.clf()

    isocenterIdx = int(doseShape[0] / 2)
    lateralProfile = doseArray[isocenterIdx, centerIdx, :]
    halfDim = int(doseShape[2] / 2)
    halfSize = halfDim * res[2]
    lateralDistance = np.linspace(-halfSize, halfSize, doseShape[2])
    plt.plot(lateralDistance, lateralProfile)
    plt.xlim(-2.0, 2.0)
    plt.xlabel("lateral distance (cm)")
    plt.ylabel("dose (normalized)")
    plt.title("Lateral dose profile at isocenter plane, beam width {}".format(width))
    figureFile = os.path.join(workFolder, "lateralDose{}.png".format(width))
    plt.savefig(figureFile)
    plt.clf()


if __name__ == "__main__":
    # waterPhantomGen()
    # structuresFileGen()
    # doseInterpret5mm()
    # showCentralBeam()
    showBEVProfile()