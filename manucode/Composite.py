import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

CompoFolder = "/data/qifan/projects/FastDoseWorkplace/DoseBench/composite"
prepFolder = os.path.join(CompoFolder, "prep_result")
ManuFiguresFolder = "/data/qifan/projects/AAPM2024/manufigures"


def phantomGen():
    dimension = (99, 256, 99)
    resolution = (0.1, 0.1, 0.1)
    
    densityArray = np.ones(dimension, dtype=np.uint16) * 1000
    densityFile = os.path.join(CompoFolder, "density_raw.bin")
    densityArray.tofile(densityFile)

    InputMaskFolder = os.path.join(CompoFolder, "InputMask")
    if not os.path.isdir(InputMaskFolder):
        os.mkdir(InputMaskFolder)
    fullMask = np.ones(dimension, dtype=np.uint8)
    PTVFile = os.path.join(InputMaskFolder, "PTV.bin")
    BODYFile = os.path.join(InputMaskFolder, "BODY.bin")
    fullMask.tofile(PTVFile)
    fullMask.tofile(BODYFile)


def genCompo():
    # get CCCS dose
    doseMatFolder = os.path.join(CompoFolder, "doseMatFolder")
    dimFile = os.path.join(prepFolder, "dimension.txt")
    with open(dimFile, "r") as f:
        lines = f.readlines()
    dimension = lines[0]
    dimension = dimension.replace(" ", ",")
    dimension = eval(dimension)
    DimProd = dimension[0] * dimension[1] * dimension[2]
    
    offsetsBufferFile = os.path.join(doseMatFolder, "offsetsBuffer.bin")
    columnsBufferFile = os.path.join(doseMatFolder, "columnsBuffer.bin")
    valuesBufferFile = os.path.join(doseMatFolder, "valuesBuffer.bin")
    offsetsBuffer = np.fromfile(offsetsBufferFile, dtype=np.uint64)
    columnsBuffer = np.fromfile(columnsBufferFile, dtype=np.uint64)
    valuesBuffer = np.fromfile(valuesBufferFile, dtype=np.float32)
    assert columnsBuffer.size == valuesBuffer.size
    
    nBeamlets = offsetsBuffer.size - 1
    matrix_global = None
    for i in range(nBeamlets):
        idx_begin = offsetsBuffer[i]
        idx_end = offsetsBuffer[i+1]
        columns_local = columnsBuffer[idx_begin: idx_end]
        values_local = valuesBuffer[idx_begin: idx_end]
        matrix_local = np.zeros(DimProd, dtype=np.float32)
        matrix_local[columns_local] = values_local
        if matrix_global is None:
            matrix_global = matrix_local
        else:
            matrix_global += matrix_local

    CCCSCompositeFile = os.path.join(CompoFolder, "composite.bin")
    matrix_global.tofile(CCCSCompositeFile)
    print(CCCSCompositeFile)


def lateralProfilePlot():
    dimFile = os.path.join(prepFolder, "dimension.txt")
    with open(dimFile, "r") as f:
        lines = f.readlines()
    dimension = lines[0]
    dimension = dimension.replace(" ", ",")
    dimension = eval(dimension)
    dimension = np.array(dimension)

    VoxelSize = lines[1]
    VoxelSize = VoxelSize.replace(" ", ",")
    VoxelSize = eval(VoxelSize)
    VoxelSize = np.array(VoxelSize)

    CCCSCompositeFile = os.path.join(CompoFolder, "composite.bin")
    CompoCCCS = np.fromfile(CCCSCompositeFile, dtype=np.float32)
    CompoCCCS = np.reshape(CompoCCCS, dimension)

    width = 3
    kernel = np.ones((width, 1, width)) / width**2
    CompoCCCS = convolve(CompoCCCS, kernel, mode="constant", cval=0)
    CenterlineIdx = int((dimension[2] - 1) / 2)
    CCCS_centerline = CompoCCCS[CenterlineIdx, :, CenterlineIdx]

    maxDose = np.max(CCCS_centerline)
    CompoCCCS *= 100 / maxDose
    CCCS_centerline = CompoCCCS[CenterlineIdx, :, CenterlineIdx]

    if True:
        CCCS_depth = (np.arange(dimension[1]) + 0.5) * VoxelSize[1]
        plt.plot(CCCS_depth, CCCS_centerline)
        figureFile = os.path.join(ManuFiguresFolder, "CCCS_width5cm_ddc.png")
        plt.savefig(figureFile)
        plt.clf()

    depthList = [10, 15, 20]
    lateralDisplacement = np.arange(dimension[2]) - (dimension[2] - 1) / 2
    lateralDisplacement *= VoxelSize[2]
    for depth in depthList:
        depthIdx = int(depth / VoxelSize[1])
        lateralProfile = CompoCCCS[CenterlineIdx, depthIdx, :]
        plt.plot(lateralDisplacement, lateralProfile, label="depth {}mm".format(depth))
    plt.legend()
    plt.xlabel("Off-axis Distance(cm)")
    plt.ylabel("Dose (% of Dmax)")
    figureFile = os.path.join(ManuFiguresFolder, "CCCS_width5cm_lateral.png")
    plt.savefig(figureFile)
    plt.clf()
    print(figureFile)


def CCCS_MC_Combine():
    dimFile = os.path.join(prepFolder, "dimension.txt")
    with open(dimFile, "r") as f:
        lines = f.readlines()
    dimension = lines[0]
    dimension = dimension.replace(" ", ",")
    dimension = eval(dimension)
    dimension = np.array(dimension)

    VoxelSize = lines[1]
    VoxelSize = VoxelSize.replace(" ", ",")
    VoxelSize = eval(VoxelSize)
    VoxelSize = np.array(VoxelSize)

    CCCSCompositeFile = os.path.join(CompoFolder, "composite.bin")
    CompoCCCS = np.fromfile(CCCSCompositeFile, dtype=np.float32)
    CompoCCCS = np.reshape(CompoCCCS, dimension)

    width = 2
    kernel = np.ones((width, 1, width)) / (2 * width + 1)**2
    CompoCCCS = convolve(CompoCCCS, kernel, mode="constant", cval=0)
    CenterlineIdx = int((dimension[2] - 1) / 2)
    CCCS_centerline = CompoCCCS[CenterlineIdx, :, CenterlineIdx]
    maxDose = np.max(CCCS_centerline)
    CompoCCCS = CompoCCCS * 100 / maxDose


    MCFile = os.path.join(CompoFolder, "MCWaterWidth5cm.bin")
    MCDim = (99, 99, 256)
    VoxelSizeMC = (0.1, 0.1, 0.1)
    MCArray = np.fromfile(MCFile, dtype=np.float64)
    MCArray = np.reshape(MCArray, MCDim)
    MC_centerline_idx = int((MCDim[0] - 1) / 2)
    MC_centerline = MCArray[MC_centerline_idx, MC_centerline_idx, :]
    MC_max_dose = np.max(MC_centerline)
    MCArray = MCArray * 100 / MC_max_dose

    # plot partial dose
    CCCS_partial = np.sum(CompoCCCS, axis=(0, 2)) * VoxelSize[0] * VoxelSize[2]
    MC_partial = np.sum(MCArray, axis=(0, 1)) * VoxelSizeMC[0] * VoxelSizeMC[1]
    CCCS_top = np.max(CCCS_partial)
    MC_top = np.max(MC_partial)
    CompoCCCS *= MC_top / CCCS_top
    
    if True:
        CCCS_depth = (np.arange(dimension[1]) + 0.5) * VoxelSize[1]
        MC_depth = (np.arange(MCDim[2]) + 0.5) * VoxelSizeMC[2]
        CCCS_partial = np.sum(CompoCCCS, axis=(0, 2)) * VoxelSize[0] * VoxelSize[2]
        MC_partial = np.sum(MCArray, axis=(0, 1)) * VoxelSizeMC[0] * VoxelSizeMC[1]
        plt.plot(CCCS_depth, CCCS_partial)
        plt.plot(MC_depth, MC_partial)
        figureFile = os.path.join(ManuFiguresFolder, "MC_CCCS_Partial.png")
        plt.savefig(figureFile)
        plt.clf()

    depthList = [5, 10, 15]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # fig, axes = plt.subplots(1, 3)
    for i, depth in enumerate(depthList):
        depth_idx_CCCS = int(depth / VoxelSize[1])
        CCCS_lateral_profile = CompoCCCS[CenterlineIdx, depth_idx_CCCS, :]
        CCCS_lateral_displacement = (np.arange(dimension[2]) - (dimension[2] - 1) / 2) * VoxelSize[2]
        axes[i].plot(CCCS_lateral_displacement, CCCS_lateral_profile, label="CCCS", linewidth=2)

        depth_idx_MC = int(depth / VoxelSizeMC[2])
        MC_lateral_profile = MCArray[MC_centerline_idx, :, depth_idx_MC]
        MC_lateral_displacement = (np.arange(MCDim[1]) - (MCDim[1] - 1) / 2) * VoxelSizeMC[1]
        axes[i].plot(MC_lateral_displacement, MC_lateral_profile, label="Monte Carlo", linestyle="--", linewidth=2)
    
        axes[i].set_xlabel("Off-axis Distance (cm)", fontsize=16)
        axes[i].set_ylabel("Dose (% of Dmax)", fontsize=16)
        axes[i].tick_params(axis="x", labelsize=14)
        axes[i].tick_params(axis="y", labelsize=14)
        axes[i].set_title("{}cm depth".format(depth), fontsize=16)
        axes[i].legend(loc="upper right", fontsize=12)
    
    plt.tight_layout()
    figureFile = os.path.join(ManuFiguresFolder, "Composite.png")
    plt.savefig(figureFile)
    plt.clf()


def CCCS_MC_Combine_correct():
    # Now we adjusted the phantoms in both Monte Carlo simulation and the
    # CCCS dose calculation to the same resolution to simplify.
    doseMatFolder = os.path.join(CompoFolder, "doseMatFolder")
    dimFile = os.path.join(prepFolder, "dimension.txt")
    with open(dimFile, "r") as f:
        lines = f.readlines()
    dimension = lines[0]
    dimension = dimension.replace(" ", ",")
    dimension = eval(dimension)
    DimProd = dimension[0] * dimension[1] * dimension[2]
    offsetsBufferFile = os.path.join(doseMatFolder, "offsetsBuffer.bin")
    columnsBufferFile = os.path.join(doseMatFolder, "columnsBuffer.bin")
    valuesBufferFile = os.path.join(doseMatFolder, "valuesBuffer.bin")
    offsetsBuffer = np.fromfile(offsetsBufferFile, dtype=np.uint64)
    columnsBuffer = np.fromfile(columnsBufferFile, dtype=np.uint64)
    valuesBuffer = np.fromfile(valuesBufferFile, dtype=np.float32)
    assert columnsBuffer.size == valuesBuffer.size

    nBeamlets = offsetsBuffer.size - 1
    matrix_global = None
    for i in range(nBeamlets):
        idx_begin = offsetsBuffer[i]
        idx_end = offsetsBuffer[i+1]
        columns_local = columnsBuffer[idx_begin: idx_end]
        values_local = valuesBuffer[idx_begin: idx_end]
        matrix_local = np.zeros(DimProd, dtype=np.float32)
        matrix_local[columns_local] = values_local
        if matrix_global is None:
            matrix_global = matrix_local
        else:
            matrix_global += matrix_local
    matrix_global = np.reshape(matrix_global, dimension)
    matrix_global = np.transpose(matrix_global, axes=(0, 2, 1))

    # load Monte Carlo dose
    MCFile = os.path.join(CompoFolder, "MCWaterWidth5cm.bin")
    MCDim = (99, 99, 256)
    voxelRes = 0.1  # cm
    MCArray = np.fromfile(MCFile, dtype=np.float64)
    MCArray = np.reshape(MCArray, MCDim)

    assert matrix_global.shape == MCArray.shape
    # matrix global has a smaller effective field than MCArray
    scale = np.sum(matrix_global * MCArray) / (np.sum(np.square(matrix_global)))
    matrix_global *= scale

    centerline_idx = int((MCDim[0] - 1) / 2)
    halfWidth = 5
    boundaryValue = 0.5
    avgKernel = np.ones((2*halfWidth+1, 2*halfWidth+1, 1))
    avgKernel[0, :, :] *= boundaryValue
    avgKernel[-1, :, :] *= boundaryValue
    avgKernel[:, 0, :] *= boundaryValue
    avgKernel[:, -1, :] *= boundaryValue
    avgKernel /= np.sum(avgKernel)

    matrix_global_smooth = convolve(matrix_global, avgKernel, mode="constant", cval=0)
    matrix_global_centerline = matrix_global_smooth[centerline_idx, centerline_idx, :]
    matrix_global_norm = np.max(matrix_global_centerline)
    matrix_global_smooth *= 100 / matrix_global_norm
    matrix_global *= 100 / matrix_global_norm

    MCArray_smooth = convolve(MCArray, avgKernel, mode="constant", cval=0)
    MCArray_centerline = MCArray_smooth[centerline_idx, centerline_idx, :]
    MCArray_norm = np.max(MCArray_centerline)
    MCArray_smooth *= 100 / MCArray_norm
    MCArray *= 100 / MCArray_norm

    if False:
        # plot depth dose curve
        matrix_global_centerline = matrix_global_smooth[centerline_idx, centerline_idx, :]
        MCArray_centerline = MCArray_smooth[centerline_idx, centerline_idx, :]
        depth = (np.arange(MCDim[2]) + 0.5) * voxelRes
        plt.plot(depth, matrix_global_centerline, label="CCCS")
        plt.plot(depth, MCArray_centerline, label="Monte Carlo")
        plt.xlabel("Depth (cm)")
        plt.ylabel("Dose (% of Dmax)")
        plt.legend()
        plt.tight_layout()
        figureFile = os.path.join(ManuFiguresFolder, "MC_CCCS_Partial.png")
        plt.savefig(figureFile)

    halfWidth = 4
    boundaryValue = 0.5
    avgKernel = np.ones((2*halfWidth+1, 2*halfWidth+1, 1))
    avgKernel[0, :, :] *= boundaryValue
    avgKernel[-1, :, :] *= boundaryValue
    avgKernel[:, 0, :] *= boundaryValue
    avgKernel[:, -1, :] *= boundaryValue
    avgKernel /= np.sum(avgKernel)
    matrix_global_smooth = convolve(matrix_global, avgKernel, mode="constant", cval=0)
    MCArray_smooth = convolve(MCArray, avgKernel, mode="constant", cval=0)

    depthList = [8, 16, 24] # cm
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, depth in enumerate(depthList):
        depthIdx = int(depth / voxelRes)
        if True:
            CCCS_lateral_profile = matrix_global_smooth[centerline_idx, :, depthIdx]
            MC_lateral_profile = MCArray_smooth[centerline_idx, :, depthIdx]
        else:
            CCCS_lateral_profile = matrix_global[centerline_idx, :, depthIdx]
            MC_lateral_profile = MCArray[centerline_idx, :, depthIdx]
        offAxisDistance = (np.arange(MCDim[1]) - (MCDim[1] - 1) / 2) * voxelRes
        axes[i].plot(offAxisDistance, CCCS_lateral_profile, label="CCCS", linewidth=2)
        axes[i].plot(offAxisDistance, MC_lateral_profile, label="Monte Carlo", linewidth=2, linestyle="--")
        axes[i].set_xlabel("Off-axis Distance (cm)", fontsize=16)
        axes[i].set_ylabel("Dose (% of Dmax)", fontsize=16)
        axes[i].set_title("Depth {} cm".format(depth), fontsize=16)
        axes[i].tick_params(axis="x", labelsize=14)
        axes[i].tick_params(axis="y", labelsize=14)
        if i == 2:
            axes[i].legend(loc="upper right", fontsize=16)
        axes[i].set_ylim(0, 80)
    plt.tight_layout()
    figureFile = os.path.join(ManuFiguresFolder, "Composite.png")
    plt.savefig(figureFile)
    figureFile = os.path.join(ManuFiguresFolder, "Composite.eps")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()
        

if __name__ == "__main__":
    # phantomGen()
    # genCompo()
    # lateralProfilePlot()
    # CCCS_MC_Combine()
    CCCS_MC_Combine_correct()