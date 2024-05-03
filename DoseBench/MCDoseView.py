import os
import numpy as np
import matplotlib.pyplot as plt

densityMap = None
width = "20mm"

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
    densityMap = np.expand_dims(densityMap, axis=(1, 2))


def getMCArray():
    rootFolder = "/data/qifan/projects/FastDoseWorkplace/DoseBench/slab"
    expFolder = os.path.join(rootFolder, "MC{}".format(width))
    sliceShape = (16, 99, 99)
    numSlices = 16

    globalShape = (sliceShape[0] * 16, sliceShape[1], sliceShape[2])
    globalArray = np.zeros(globalShape, dtype=np.double)
    for i in range(numSlices):
        slicePath = os.path.join(expFolder, "SD{:03d}.bin".format(i+1))
        localArray = np.fromfile(slicePath, dtype=np.double)
        localArray = np.reshape(localArray, sliceShape)
        idx_start = i * sliceShape[0]
        idx_end = idx_start + sliceShape[0]
        globalArray[idx_start: idx_end, :, :] = localArray
    globalArray /= densityMap
    resolution = np.array((0.1, 0.1, 0.1))
    return globalArray, resolution


def getCCCSArray():
    rootFolder = "/data/qifan/projects/FastDoseWorkplace/DoseBench/slab"
    expFolder = os.path.join(rootFolder, "width{}".format(width))

    if width == "5mm":
        doseFile = os.path.join(expFolder, "BEVdose246.bin")
        doseShape = (102, 24, 24)
        res = (0.25, 0.08333, 0.08333)
    elif width == "10mm":
        doseFile = os.path.join(expFolder, "BEVdose248.bin")
        doseShape = (102, 24, 24)
        res = (0.25, 0.125, 0.125)
    elif width == "20mm":
        doseFile = os.path.join(expFolder, "BEVdose248.bin")
        doseShape = (102, 24, 24)
        res = (0.25, 0.25, 0.25)
    
    doseArray = np.fromfile(doseFile, dtype=np.float32)
    doseArray = np.reshape(doseArray, doseShape)
    return doseArray, np.array(res)


def MCDoseView():
    rootFolder = "/data/qifan/projects/FastDoseWorkplace/DoseBench/slab"
    expFolder = os.path.join(rootFolder, "MC{}".format(width))

    CCCSArray, CCCSRes = getCCCSArray()
    CCCSShape = np.array(CCCSArray.shape)
    CCCSArray *= 100 / np.max(CCCSArray)

    MCArray, MCRes = getMCArray()
    MCShape = np.array(MCArray.shape)
    MCArray *= 100 / np.max(MCArray)

    # draw DDP
    CCCSDepth = np.arange(CCCSShape[0]) + 0.5
    CCCSDepth *= CCCSRes[0]
    CCCSCenterIdx = int((CCCSShape[1]-1)/2)
    CCCSCenterline = CCCSArray[:, CCCSCenterIdx, CCCSCenterIdx].copy()
    plt.plot(CCCSDepth, CCCSCenterline, label="CCCS")

    MCDepth = np.arange(MCShape[0]) + 0.5
    MCDepth *= MCRes[0]
    MCCenterIdx = int((MCShape[1]-1)/2)
    MCCenterline = MCArray[:, MCCenterIdx, MCCenterIdx].copy()
    factor = doseNormalize(CCCSDepth, CCCSCenterline, MCDepth, MCCenterline)
    MCArray *= factor
    MCCenterline = MCArray[:, MCCenterIdx, MCCenterIdx].copy()
    plt.plot(MCDepth, MCCenterline, label="Monte Carlo")
    
    plt.xlabel("depth (cm)")
    plt.ylabel("percent depth dose")
    plt.legend()
    plt.title("Depth Dose Comparison at Beam Width {}".format(width))
    figureFile = os.path.join(expFolder, "DepthDoseComp{}.png".format(width))
    plt.savefig(figureFile)
    plt.clf()


    # draw lateral profile
    CCCSDepthIdx = int((CCCSShape[0]-1)/2)
    CCCSProfilie = CCCSArray[CCCSDepthIdx, CCCSCenterIdx, :]
    CCCSLateralDisplacement = np.arange(CCCSShape[2]) - (CCCSShape[2]-1) / 2
    CCCSLateralDisplacement *= CCCSRes[2]
    plt.plot(CCCSLateralDisplacement, CCCSProfilie, label="CCCS")

    MCDepthIdx = int((MCShape[0]-1)/2)
    MCProfile = MCArray[MCDepthIdx, MCCenterIdx, :]
    MCLateralDisplacement = np.arange(MCShape[2]) - (MCShape[2]-1) / 2
    MCLateralDisplacement *= MCRes[2]
    plt.plot(MCLateralDisplacement, MCProfile, label="Monte Carlo")

    plt.xlabel("off-axis distance (cm)")
    plt.ylabel("percent max dose")
    plt.xlim(-2.0, 2.0)
    plt.ylim(0.0, 50.0)
    plt.legend()
    plt.title("Lateral Dose Profile at Beam Width {}".format(width))
    figureFile = os.path.join(expFolder, "LateralDoseComp{}.png".format(width))
    plt.savefig(figureFile)
    plt.clf()


def doseNormalize(depthRef, doseRef, depthTarget, doseTarget):
    interpolatedTarget = np.interp(depthRef, depthTarget, doseTarget)
    coefficient = np.sum(interpolatedTarget * doseRef) / np.sum(interpolatedTarget ** 2)
    return coefficient


if __name__ == "__main__":
    densityMapInit()
    MCDoseView()