import os
import numpy as np
import json
import matplotlib.pyplot as plt

def slabPhantomGen():
    """This function generates the slab phantom"""
    dimension = (256, 256, 256)
    array = np.zeros(dimension, dtype=np.float32)
    resolution = (0.1, 0.1, 0.1)
    DensityLUT = {
        "Adipose": 0.92,
        "Muscle": 1.04,
        "Bone": 1.85,
        "Lung": 0.25
    }
    Material = [
        ("Adipose", 1),
        ("Muscle", 1),
        ("Bone", 1),
        ("Muscle", 1),
        ("Lung", 6),
        ("Muscle", 1),
        ("Bone", 1),
        ("Adipose", 1),
        ("Bone", 1),
        ("Muscle", 1),
        ("Adipose", 1)
    ]
    Material = [(a, 16*b) for a, b in Material]
    base = 0
    for mat, thick in Material:
        density = DensityLUT[mat]
        array[:, base:base+thick, :] = density
        base += thick
    densityFile = "/data/qifan/projects/FastDoseWorkplace/DoseBench/slab/slabDensity.raw"
    array.tofile(densityFile)
    print(densityFile)


def showBEVProfile():
    width = "20mm"
    workFolder = "/data/qifan/projects/FastDoseWorkplace/DoseBench/slab/width{}".format(width)

    if width == "5mm":
        doseFile = os.path.join(workFolder, "BEVdose246.bin")
        doseShape = (102, 24, 24)
        res = (0.25, 0.08333, 0.08333)
    elif width == "10mm":
        doseFile = os.path.join(workFolder, "BEVdose248.bin")
        doseShape = (102, 24, 24)
        res = (0.25, 0.125, 0.125)
    elif width == "20mm":
        doseFile = os.path.join(workFolder, "BEVdose248.bin")
        doseShape = (102, 24, 24)
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
    # slabPhantomGen()
    showBEVProfile()