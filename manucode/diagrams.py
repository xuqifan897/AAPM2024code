import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import nrrd
from scipy.interpolate import RegularGridInterpolator

figuresFolder = "/data/qifan/AAPM2024/manufigures"
topicFolder = os.path.join(figuresFolder, "ultraFastChen")
if not os.path.isdir(topicFolder):
    os.mkdir(topicFolder)
pixelSize = 1.08  # mm

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())

def getSlice():
    """
    This function creates a single axial slice of a CT phantom
    """
    sliceIdx = 10
    valueMin = 500
    valueMax = 1500

    CTArray = "/data/qifan/FastDoseWorkplace/TCIAAdd/002/CT.nrrd"
    CTArray, header = nrrd.read(CTArray)
    CTArray += 1024
    CTSlice = CTArray[:, :, sliceIdx]
    CTSlice = np.transpose(CTSlice)
    # CTSlice = CTSlice[:380, :]

    sliceFile = os.path.join(topicFolder, "sliceOrg.png")
    plt.imsave(sliceFile, CTSlice, cmap="gray", vmin=500, vmax=1500)


def drawBeam():
    SAD = 1000  # mm
    SAD /= pixelSize  # pixel
    fmapSize = 100  # mm
    fmapSize /= pixelSize  # pixel
    angle = np.pi/6  # rad
    halfSize = 256
    color = colors[2]

    canvas = os.path.join(topicFolder, "sliceOrg.png")
    canvas = plt.imread(canvas)

    # calculate the source location
    sourceCoords = np.array((np.sin(angle), np.cos(angle))) * SAD
    leftPointCoords = np.array((-np.cos(angle), np.sin(angle))) * fmapSize / 2
    rightPointCoords = - leftPointCoords

    def adjustImageCoords(coordsOrg):
        x = coordsOrg[0] + halfSize
        y = halfSize - coordsOrg[1]
        return np.array((x, y))
    
    halfLengthBegin = 130  # mm
    halfLengthEnd = 170  # mm
    point12_1 = sourceCoords * halfLengthBegin / SAD + leftPointCoords * (SAD - halfLengthBegin) / SAD
    point12_2 = - sourceCoords * halfLengthEnd / SAD + leftPointCoords * (SAD + halfLengthEnd) / SAD
    point34_1 = sourceCoords * halfLengthBegin / SAD + rightPointCoords * (SAD - halfLengthBegin) / SAD
    point34_2 = - sourceCoords * halfLengthEnd / SAD + rightPointCoords * (SAD + halfLengthEnd) / SAD

    # prepare for transverse lines
    nLines = 16
    displacement12 = (point12_2 - point12_1) / (nLines - 1)
    displacement34 = (point34_2 - point34_1) / (nLines - 1)
    transverseLines = []
    for i in range(nLines):
        leftPoint = point12_1 + displacement12 * i
        leftPoint = adjustImageCoords(leftPoint)
        rightPoint = point34_1 + displacement34 * i
        rightPoint = adjustImageCoords(rightPoint)
        transverseLines.append([leftPoint, rightPoint])

    sourceCoords = adjustImageCoords(sourceCoords)
    leftPointCoords = adjustImageCoords(leftPointCoords)
    rightPointCoords = adjustImageCoords(rightPointCoords)
    point12_1 = adjustImageCoords(point12_1)
    point12_2 = adjustImageCoords(point12_2)
    point34_1 = adjustImageCoords(point34_1)
    point34_2 = adjustImageCoords(point34_2)

    point1, point2 = calcBoundaryPoints(sourceCoords, leftPointCoords, halfSize)
    point3, point4 = calcBoundaryPoints(sourceCoords, rightPointCoords, halfSize)
    fig, ax = plt.subplots(figsize=(2*halfSize/100, 2*halfSize/100), dpi=100)
    ax.imshow(canvas)

    ax.plot((point1[0], point2[0]), (point1[1], point2[1]), color, linestyle="--")
    ax.plot((point3[0], point4[0]), (point3[1], point4[1]), color, linestyle="--")

    for pointLeft, pointRight in transverseLines:
        ax.plot((pointLeft[0], pointRight[0]), (pointLeft[1], pointRight[1]), color=color, linestyle="--")

    longitudinalLines = []
    nLongiLines = 3
    for i in range(1, nLongiLines):
        pointUp = transverseLines[0][0] * (1 - i / nLongiLines) + transverseLines[0][1] * (i / nLongiLines)
        pointDown = transverseLines[-1][0] * (1 - i / nLongiLines) + transverseLines[-1][1] * (i / nLongiLines)
        longitudinalLines.append([pointUp, pointDown])
    for pointUp, pointDown in longitudinalLines:
        ax.plot((pointUp[0], pointDown[0]), (pointUp[1], pointDown[1]), color=color, linestyle="--")

    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figureFile = os.path.join(topicFolder, "Cart2BEV.png")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()


def calcBoundaryPoints(point1, point2, halfSize):
    diff = point2 - point1
    # calculate the intersection of all the lines with the boundary of the box
    eps = 1e-4
    boundaries = []
    if np.abs(diff[0]) >= eps:
        k1 = (0 - point1[0]) / diff[0]
        boundary1 = point1 + k1 * diff
        k2 = (2*halfSize - point1[0]) / diff[0]
        boundary2 = point1 + k2 * diff
        boundaries.append(boundary1)
        boundaries.append(boundary2)
    if np.abs(diff[1]) >= eps:
        k3 = (0 - point1[1]) / diff[1]
        boundary3 = point1 + k3 * diff
        k4 = (2*halfSize - point1[1]) / diff[1]
        boundary4 = point1 + k4 * diff
        boundaries.append(boundary3)
        boundaries.append(boundary4)

    print(boundaries)
    boundaries = [np.array((x, y)) for x, y in boundaries if x>=-eps
        and x<=2*halfSize+eps and y>=-eps and y<=2*halfSize+eps]
    boundaries = [np.clip(a, eps, 2*halfSize-eps-1) for a in boundaries]
    print(boundaries)
    if len(boundaries) == 0:
        return False
    else:
        return boundaries[0], boundaries[1]
    

def DensityBEV():
    """
    This function generates the density interpolated to the BEV coordinate frame
    """
    sliceOrg = os.path.join(topicFolder, "sliceOrg.png")
    sliceOrg = plt.imread(sliceOrg)
    
    SAD = 1000  # mm
    SAD /= pixelSize  # pixel
    fmapSize = 100  # mm
    fmapSize /= pixelSize  # pixel
    angle = np.pi/6  # rad
    halfSize = 256
    color = colors[2]

    canvas = os.path.join(topicFolder, "sliceOrg.png")
    canvas = plt.imread(canvas)

    halfLengthBegin = 130  # mm
    halfLengthEnd = 170  # mm
    voxelSize = 1

    # calculate the sampling coordinates
    longitudinalDim = int((halfLengthBegin + halfLengthEnd) / voxelSize)
    transverseDim = int(fmapSize / voxelSize)
    coords = np.zeros((longitudinalDim, transverseDim, 2), dtype=float)
    coords_height = np.arange(longitudinalDim) - halfLengthBegin
    coords_height = np.expand_dims(coords_height, axis=1)
    coords[:, :, 0] = coords_height
    coords_width = np.arange(transverseDim) - (transverseDim - 1) / 2
    coords_width = np.expand_dims(coords_width, axis=0)
    coords[:, :, 1] = coords_width
    
    # transform the coords in BEV to PVCS
    def transformFunc(sourceCoords, coords_org, angle, SAD):
        sourceToPointDistance = coords_org[:, :, 0] + SAD
        scalingFactor = sourceToPointDistance / SAD
        transverseDistance = coords_org[:, :, 1] * scalingFactor
        resultCoordsHeight = np.cos(angle) * sourceToPointDistance + np.sin(angle) * transverseDistance
        resultCoordsHeight += sourceCoords[0]
        resultCoordsHeight = np.expand_dims(resultCoordsHeight, axis=2)
        resultCoordsWidth = -np.sin(angle) * sourceToPointDistance + np.cos(angle) * transverseDistance
        resultCoordsWidth += sourceCoords[1]
        resultCoordsWidth = np.expand_dims(resultCoordsWidth, axis=2)
        resultCoords = np.concatenate((resultCoordsHeight, resultCoordsWidth), axis=2)
        return resultCoords
    
    sourceCoords = np.array((halfSize, halfSize)) + SAD * np.array((-np.cos(angle), np.sin(angle)))
    resultCoords = transformFunc(sourceCoords, coords, angle, SAD)
    resultCoords = np.reshape(resultCoords, (longitudinalDim * transverseDim, 2))
    
    # interpolate
    layers = []
    for i in range(4):
        # there are four channels of the image
        channel = sliceOrg[:, :, i]
        interpolator = RegularGridInterpolator((np.arange(2*halfSize), np.arange(2*halfSize)), channel)
        channelValue = interpolator(resultCoords)
        channelValue = np.reshape(channelValue, (longitudinalDim, transverseDim))
        channelValue = np.expand_dims(channelValue, axis=2)
        layers.append(channelValue)
    newImage = np.concatenate(layers, axis=2)
    newImageFile = os.path.join(topicFolder, "densityBEV.png")
    plt.imsave(newImageFile, newImage)
    plt.clf()
    
    fig, ax = plt.subplots(figsize=(transverseDim/100, longitudinalDim/100), dpi=100)
    ax.imshow(newImage)

    nArrows = 4
    for i in range(1, nArrows):
        xDisplacement = i / nArrows * transverseDim
        pointBegin = (xDisplacement, 0)
        pointEnd = (xDisplacement, longitudinalDim)
        arrow = FancyArrowPatch(pointBegin, pointEnd, arrowstyle="->", mutation_scale=30,
            linestyle="--", linewidth=2, color=color)
        ax.add_patch(arrow)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    newImageFile = os.path.join(topicFolder, "BEVRayTracing.png")
    plt.savefig(newImageFile)
    plt.close(fig)
    plt.clf()


def TermaBEV():
    """
    This function draws the Terma in the BEV coordinate system
    """
    sliceIdx = 10

    CTArray = "/data/qifan/FastDoseWorkplace/TCIAAdd/002/CT.nrrd"
    CTArray, header = nrrd.read(CTArray)
    CTArray += 1024
    CTSlice = CTArray[:, :, sliceIdx]
    CTSlice = np.transpose(CTSlice)

    # interpolate from CTSlice to get the density view in BEV
    halfLengthBegin = 130  # mm
    halfLengthEnd = 170  # mm
    fmapSize = 100 / pixelSize
    halfSize = 256
    SAD = 1000 / pixelSize
    angle = np.pi / 6

    longitudinalDim = (halfLengthBegin + halfLengthEnd)
    transverseDim = int(fmapSize)

    # calculate the sampling coordinates
    coords = np.zeros((longitudinalDim, transverseDim, 2), dtype=float)
    coords_height = np.arange(longitudinalDim) - halfLengthBegin
    coords_height = np.expand_dims(coords_height, axis=1)
    coords_width = np.arange(transverseDim) - (transverseDim - 1) / 2
    coords_width = np.expand_dims(coords_width, axis=0)
    coords[:, :, 0] = coords_height
    coords[:, :, 1] = coords_width
    
    # transform the coords in BEV to PVCS
    def transformFunc(sourceCoords, coords_org, angle, SAD):
        sourceToPointDistance = coords_org[:, :, 0] + SAD
        scalingFactor = sourceToPointDistance / SAD
        transverseDistance = coords_org[:, :, 1] * scalingFactor
        resultCoordsHeight = np.cos(angle) * sourceToPointDistance + np.sin(angle) * transverseDistance
        resultCoordsHeight += sourceCoords[0]
        resultCoordsHeight = np.expand_dims(resultCoordsHeight, axis=2)
        resultCoordsWidth = -np.sin(angle) * sourceToPointDistance + np.cos(angle) * transverseDistance
        resultCoordsWidth += sourceCoords[1]
        resultCoordsWidth = np.expand_dims(resultCoordsWidth, axis=2)
        resultCoords = np.concatenate((resultCoordsHeight, resultCoordsWidth), axis=2)
        return resultCoords
    
    sourceCoords = np.array((halfSize, halfSize)) + SAD * np.array((-np.cos(angle), np.sin(angle)))
    resultCoords = transformFunc(sourceCoords, coords, angle, SAD)
    resultCoords = np.reshape(resultCoords, (longitudinalDim * transverseDim, 2))
    interpolator = RegularGridInterpolator((np.arange(2*halfSize), np.arange(2*halfSize)), CTSlice)
    densityBEV = interpolator(resultCoords)
    densityBEV = np.reshape(densityBEV, (longitudinalDim, transverseDim))
    densityBEV /= 1000  # density relative to water

    if False:
        CTSliceFile = os.path.join(topicFolder, "densitySlice.png")
        plt.imsave(CTSliceFile, densityBEV, cmap="gray")
        return
    
    spectrumFile = "/data/qifan/FastDose/scripts/spec_6mv.spec"
    with open(spectrumFile, "r") as f:
        spectrumFile = f.readlines()
    numEntries = len(spectrumFile)
    energy = np.zeros(numEntries, dtype=float)
    fluence = energy.copy()
    mu_en = energy.copy()
    mu = energy.copy()
    for i in range(numEntries):
        line = spectrumFile[i]
        line = line.split(" ")[:-1]
        line = [eval(a) for a in line]
        energy[i], fluence[i], mu[i], mu_en[i] = line
    
    # calculate radiological path
    print(np.min(densityBEV), np.max(densityBEV))
    step_size = (np.arange(transverseDim) - (transverseDim - 1) / 2) * pixelSize  # mm
    SAD_mm = SAD * pixelSize
    step_size = np.sqrt(np.square(step_size) + SAD_mm ** 2)
    step_size *= pixelSize / SAD_mm  # mm
    step_size *= 0.1  # cm
    
    # Initialize radiological path length
    radPath = np.zeros_like(densityBEV)
    for i in range(1, radPath.shape[0]):
        densityRow = densityBEV[i, :]
        radPath[i, :] = radPath[i-1, :] + densityRow * step_size
    
    # calculate terma value
    termaArray = None
    for e in range(numEntries):
        this_fluence = fluence[e]
        this_energy = energy[e]
        this_mu = mu[e]
        if termaArray is None:
            termaArray = this_fluence * this_energy * this_mu * np.exp(- this_mu * radPath)
        else:
            termaArray += this_fluence * this_energy * this_mu * np.exp(- this_mu * radPath)

    termaFile = os.path.join(topicFolder, "TermaBEV.png")
    fig, ax = plt.subplots(figsize=(densityBEV.shape[1]/100, densityBEV.shape[0]/100), dpi=100)
    ax.imshow(densityBEV, cmap="gray", vmin=0.5, vmax=1.5)
    ax.imshow(termaArray, cmap="jet", vmin=0, vmax=np.max(termaArray), alpha=0.8)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(termaFile)
    plt.close(fig)
    plt.clf()


def TermaPVCS():
    """
    This function draws the Terma in the BEV coordinate system
    """
    sliceIdx = 10

    CTArray = "/data/qifan/FastDoseWorkplace/TCIAAdd/002/CT.nrrd"
    CTArray, header = nrrd.read(CTArray)
    CTArray += 1024
    CTSlice = CTArray[:, :, sliceIdx]
    CTSlice = np.transpose(CTSlice)

    # interpolate from CTSlice to get the density view in BEV
    halfLengthBegin = 130  # mm
    halfLengthEnd = 170  # mm
    fmapSize = 100 / pixelSize
    halfSize = 256
    SAD = 1000 / pixelSize
    angle = np.pi / 6

    longitudinalDim = (halfLengthBegin + halfLengthEnd)
    transverseDim = int(fmapSize)

    # calculate the sampling coordinates
    coords = np.zeros((longitudinalDim, transverseDim, 2), dtype=float)
    coords_height = np.arange(longitudinalDim) - halfLengthBegin
    coords_height = np.expand_dims(coords_height, axis=1)
    coords_width = np.arange(transverseDim) - (transverseDim - 1) / 2
    coords_width = np.expand_dims(coords_width, axis=0)
    coords[:, :, 0] = coords_height
    coords[:, :, 1] = coords_width
    
    # transform the coords in BEV to PVCS
    def transformFunc(sourceCoords, coords_org, angle, SAD):
        sourceToPointDistance = coords_org[:, :, 0] + SAD
        scalingFactor = sourceToPointDistance / SAD
        transverseDistance = coords_org[:, :, 1] * scalingFactor
        resultCoordsHeight = np.cos(angle) * sourceToPointDistance + np.sin(angle) * transverseDistance
        resultCoordsHeight += sourceCoords[0]
        resultCoordsHeight = np.expand_dims(resultCoordsHeight, axis=2)
        resultCoordsWidth = -np.sin(angle) * sourceToPointDistance + np.cos(angle) * transverseDistance
        resultCoordsWidth += sourceCoords[1]
        resultCoordsWidth = np.expand_dims(resultCoordsWidth, axis=2)
        resultCoords = np.concatenate((resultCoordsHeight, resultCoordsWidth), axis=2)
        return resultCoords
    
    sourceCoords = np.array((halfSize, halfSize)) + SAD * np.array((-np.cos(angle), np.sin(angle)))
    resultCoords = transformFunc(sourceCoords, coords, angle, SAD)
    resultCoords = np.reshape(resultCoords, (longitudinalDim * transverseDim, 2))
    interpolator = RegularGridInterpolator((np.arange(2*halfSize), np.arange(2*halfSize)), CTSlice)
    densityBEV = interpolator(resultCoords)
    densityBEV = np.reshape(densityBEV, (longitudinalDim, transverseDim))
    densityBEV /= 1000  # density relative to water
    
    spectrumFile = "/data/qifan/FastDose/scripts/spec_6mv.spec"
    with open(spectrumFile, "r") as f:
        spectrumFile = f.readlines()
    numEntries = len(spectrumFile)
    energy = np.zeros(numEntries, dtype=float)
    fluence = energy.copy()
    mu_en = energy.copy()
    mu = energy.copy()
    for i in range(numEntries):
        line = spectrumFile[i]
        line = line.split(" ")[:-1]
        line = [eval(a) for a in line]
        energy[i], fluence[i], mu[i], mu_en[i] = line
    
    # calculate radiological path
    print(np.min(densityBEV), np.max(densityBEV))
    step_size = (np.arange(transverseDim) - (transverseDim - 1) / 2) * pixelSize  # mm
    SAD_mm = SAD * pixelSize
    step_size = np.sqrt(np.square(step_size) + SAD_mm ** 2)
    step_size *= pixelSize / SAD_mm  # mm
    step_size *= 0.1  # cm
    
    # Initialize radiological path length
    radPath = np.zeros_like(densityBEV)
    for i in range(1, radPath.shape[0]):
        densityRow = densityBEV[i, :]
        radPath[i, :] = radPath[i-1, :] + densityRow * step_size
    
    # calculate terma value
    termaArray = None
    for e in range(numEntries):
        this_fluence = fluence[e]
        this_energy = energy[e]
        this_mu = mu[e]
        if termaArray is None:
            termaArray = this_fluence * this_energy * this_mu * np.exp(- this_mu * radPath)
        else:
            termaArray += this_fluence * this_energy * this_mu * np.exp(- this_mu * radPath)
    
    # interpolate BEV terma to PVCS
    coordsPVCS = np.zeros((2*halfSize, 2*halfSize, 2))
    coordsPVCSHeight = np.arange(2*halfSize)
    coordsPVCSHeight = np.expand_dims(coordsPVCSHeight, axis=1)
    coordsPVCS[:, :, 0] = coordsPVCSHeight
    coordsPVCSWidth = np.arange(2*halfSize)
    coordsPVCSWidth = np.expand_dims(coordsPVCSWidth, axis=0)
    coordsPVCS[:, :, 1] = coordsPVCSWidth
    sourceCoords = np.expand_dims(sourceCoords, axis=(0, 1))
    coordsPVCS -= sourceCoords  # relative to the source
    longiAxis = np.array((np.cos(angle), -np.sin(angle)))
    longiAxis = np.expand_dims(longiAxis, axis=(0, 1))
    longiCoord = np.sum(coordsPVCS * longiAxis, axis=2)
    longiCoord = np.expand_dims(longiCoord, axis=2)

    perpendicular = coordsPVCS - longiCoord * longiAxis
    perpenAxis = np.array((np.sin(angle), np.cos(angle)))
    perpenAxis = np.expand_dims(perpenAxis, axis=(0, 1))
    perpenProj = np.sum(perpendicular * perpenAxis, axis=2)
    perpenProj = np.expand_dims(perpenProj, axis=2)
    
    perpenCoord = perpenProj / longiCoord * SAD
    longiCoord -= (SAD - halfLengthBegin)
    perpenCoord += (transverseDim - 1) / 2
    BEVCoordsPVCS = np.concatenate((longiCoord, perpenCoord), axis=2)
    BEVCoordsPVCS = np.reshape(BEVCoordsPVCS, (4 * halfSize ** 2, 2))
    
    BEVDoseGrid = RegularGridInterpolator((np.arange(longitudinalDim),
        np.arange(transverseDim)), termaArray, bounds_error=False, fill_value=0)
    PVCSValues = BEVDoseGrid(BEVCoordsPVCS)
    PVCSValues = np.reshape(PVCSValues, (2*halfSize, 2*halfSize))

    SKINMask = "/data/qifan/FastDoseWorkplace/TCIAAdd/002/RTSTRUCT.nrrd"
    SKINMask = getMask(SKINMask)
    SKINMask = SKINMask[sliceIdx, :, :]
    
    fig, ax = plt.subplots(figsize=(2*halfSize/100, 2*halfSize/100), dpi=100)
    ax.imshow(CTSlice, cmap="gray", vmin=500, vmax=1500)
    ax.imshow(PVCSValues, cmap="jet", vmin=0, vmax=np.max(PVCSValues),
        alpha = 0.8 * np.logical_and(SKINMask > 0, PVCSValues > 0))
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figureFile = os.path.join(topicFolder, "TermaPVCS.png")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()


def getMask(maskFile):
    maskArray, maskHeader = nrrd.read(maskFile)
    name = "SKIN"
    idx = 0

    Layer = None
    LabelValue = None
    while True:
        keyBegin = "Segment{}_".format(idx)
        idx += 1
        key = keyBegin + "Name"
        if key not in maskHeader:
            break
        name = maskHeader[key]
        if name == "SKIN":
            Layer = int(maskHeader[keyBegin + "Layer"])
            LabelValue = int(maskHeader[keyBegin + "LabelValue"])
            break
    assert Layer is not None and LabelValue is not None
    SKINMask = maskArray[Layer, :, :, :] == LabelValue
    SKINMask = np.transpose(SKINMask, axes=(2, 1, 0))
    return SKINMask


def figMerge():
    Cart2BEV = os.path.join(topicFolder, "Cart2BEV.png")
    Cart2BEV = plt.imread(Cart2BEV)
    BEVRayTracing = os.path.join(topicFolder, "BEVRayTracing.png")
    BEVRayTracing = plt.imread(BEVRayTracing)
    TermaBEV = os.path.join(topicFolder, "TermaBEV.png")
    TermaBEV = plt.imread(TermaBEV)
    TermaPVCS = os.path.join(topicFolder, "TermaPVCS.png")
    TermaPVCS = plt.imread(TermaPVCS)
    print(Cart2BEV.shape, BEVRayTracing.shape, TermaBEV.shape, TermaPVCS.shape)

    boundingBoxSize = (512, 128, 4)
    boundingBox = np.zeros(boundingBoxSize, dtype=BEVRayTracing.dtype)
    boundingBox[:, :, -1] = 1
    idx_begin_height = int((boundingBoxSize[0] - BEVRayTracing.shape[0]) / 2)
    idx_begin_width = int((boundingBoxSize[1] - BEVRayTracing.shape[1]) / 2)
    boundingBox[idx_begin_height: idx_begin_height+BEVRayTracing.shape[0],
        idx_begin_width: idx_begin_width+BEVRayTracing.shape[1], :] = BEVRayTracing
    
    boundingBox1 = np.zeros_like(boundingBox)
    boundingBox1[:, :, -1] = 1
    boundingBox1[idx_begin_height: idx_begin_height+BEVRayTracing.shape[0],
        idx_begin_width: idx_begin_width+BEVRayTracing.shape[1], :] = TermaBEV
    
    image = np.concatenate([Cart2BEV, boundingBox, boundingBox1, TermaPVCS], axis=1)
    imageFile = os.path.join(topicFolder, "ultraFastChen.png")
    
    fig, ax = plt.subplots(figsize=(image.shape[1]/100, image.shape[0]/100), dpi=100)
    ax.imshow(image)
    offset1 = 10
    offset2 = 40
    fontsize = 20
    ax.text(offset1, offset2, "(a)", color="white", fontsize=fontsize)
    ax.text(512 + offset1, offset2, "(b)", color="white", fontsize=fontsize)
    ax.text(768 + offset1, offset2, "(c)", color="white", fontsize=fontsize)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(imageFile)
    plt.close(fig)
    plt.clf()


if __name__ == "__main__":
    # getSlice()
    # drawBeam()
    # DensityBEV()
    # TermaBEV()
    # TermaPVCS()
    figMerge()