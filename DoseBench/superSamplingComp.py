import os
import numpy as np
import matplotlib.pyplot as plt

phantom="slab"
# phantom="water"

width=0.5
# width=1.0
# width=2.0

rootFolder = "/data/qifan/projects/FastDoseWorkplace/DoseBench/MCSuperSampling"
subFluenceDim = 24
subFluenceRes = None
subFluenceOn = None
dimension_MC = (99, 99, 256)
densityMap = None
densityMapRes = 0.1
longitudinalSpacing = 0.25

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

def paramsInit():
    global subFluenceRes
    global subFluenceOn
    if width == 0.5:
        subFluenceRes = 0.08333
        subFluenceOn = 6
    elif width == 1.0:
        subFluenceRes = 0.125
        subFluenceOn = 8
    elif width == 2.0:
        subFluenceRes = 2.0
        subFluenceOn = 8


def dosePlotSlab():
    assert phantom == "slab", "Incorrect phantom type"
    MCMatFile = os.path.join(rootFolder, "MCDose_width_{}cm.bin".format(width))
    MCArray = np.fromfile(MCMatFile, dtype=np.double)
    MCArray = np.reshape(MCArray, dimension_MC)
    densityMap_local = np.expand_dims(densityMap, axis=(0, 1))
    MCArray = MCArray / densityMap_local  # convert energy deposition to dose
    MCArray /= np.max(MCArray)

    CCCSMatFile = os.path.join(rootFolder, "BEVDose_{}_{}.bin".format(phantom, width))
    CCCSArray = np.fromfile(CCCSMatFile, dtype=np.float32)
    nElements = CCCSArray.size
    assert nElements % (subFluenceDim * subFluenceDim) == 0, "Dimension incorrect"
    dimZ = nElements // (subFluenceDim * subFluenceDim)
    CCCSShape = (dimZ, subFluenceDim, subFluenceDim)
    CCCSArray = np.reshape(CCCSArray, CCCSShape)

    CCCSDepth = (np.arange(dimZ) + 0.5) * longitudinalSpacing
    densityMapDepth = (np.arange(densityMap.size) + 0.5) * densityMapRes
    CCCSArray = CCCSArray
    CCCSArray /= np.max(CCCSArray)

    # plot depth dose curve
    CCCSCenterIdx = subFluenceDim // 2
    CCCSCenterline = CCCSArray[:, CCCSCenterIdx, CCCSCenterIdx]
    plt.plot(CCCSDepth, CCCSCenterline, label="CCCS")

    MCCenterIdx = (dimension_MC[0] - 1) // 2
    MCCenterline = MCArray[MCCenterIdx, MCCenterIdx, :]
    plt.plot(densityMapDepth, MCCenterline, label="Monte Carlo")
    plt.legend()
    plt.title("Depth dose curve ({}cm)".format(width))
    plt.tight_layout()
    figureFile = os.path.join(rootFolder, "DepthDoseCurve_{}cm.png".format(width))
    plt.savefig(figureFile)
    plt.clf()


if __name__ == "__main__":
    densityMapInit()
    paramsInit()
    dosePlotSlab()