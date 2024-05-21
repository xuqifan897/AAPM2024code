import os
import numpy as np
import matplotlib.pyplot as plt
import glob

densityMap = None
dimension = (99, 99, 256)  # (z, y, x)

def densityMapInit():
    global densityMap
    matMap = {"Adipose": 0.92, "Muscle": 1.04, "Bone": 1.85, "Lung": 0.25}
    Materials = [
        ("Adipose", 16),
        ("Muscle", 16),
        ("Bone", 16),
        ("Muscle", 16),
        ("Lung", 96),
        ("Muscle", 16),
        ("Bone", 16),
        ("Adipose", 16),
        ("Bone", 16),
        ("Muscle", 16),
        ("Adipose", 16)
    ]
    totalSlices = 0
    for material, thickness in Materials:
        totalSlices += thickness
    densityMap = np.zeros(totalSlices)
    offset = 0
    for material, thickness in Materials:
        density = matMap[material]
        densityMap[offset: offset+thickness] = density
        offset += thickness

def showCoarse():
    """
    This script examines the correctness of the MC data obtained.
    """
    
    # Show partial Edep dose
    folder = "/data/qifan/projects/FastDoseWorkplace/DoseBench/MCSuperSampling"
    pattern = os.path.join(folder, "*.bin")
    EdepFiles = glob.glob(pattern)
    
    for EdepFile in EdepFiles:
        leading = EdepFile.split(".")[:-1]
        leading = ".".join(leading)

        EdepMat = np.fromfile(EdepFile, dtype=np.double)
        EdepMat = np.reshape(EdepMat, dimension)
        # show centerline dose
        centerIdx = int((dimension[0]-1)/2), int((dimension[1]-1)/2)
        centerline = EdepMat[centerIdx[0], centerIdx[1], :].copy()
        centerline /= densityMap
        depth = np.arange(dimension[2]) * 0.1  # cm
        plt.plot(depth, centerline)
        plt.xlabel("Depth")
        plt.ylabel("Energy deposition")

        figureFile = leading + ".png"
        plt.savefig(figureFile)
        plt.clf()
        print(figureFile)


if __name__ == "__main__":
    densityMapInit()
    showCoarse()