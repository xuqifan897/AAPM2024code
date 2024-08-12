import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import nrrd
from scipy.interpolate import RegularGridInterpolator
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.transform import resize

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())


spectrumFile = "/data/qifan/FastDose/scripts/spec_6mv.spec"
colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
figuresFolder = "/data/qifan/AAPM2024/manufigures"
topicFolder = os.path.join(figuresFolder, "BeamletDosecalc")
if not os.path.isdir(topicFolder):
    os.mkdir(topicFolder)

densitySlice = None
densityMask = None
beamAngleGlobal = np.pi / 6
isoCenterGlobal = None
resolution = 1.08  # mm
halfXDim = int(50 / resolution)
SAD = 1000 / resolution

# after calculation, we discovered the following parameters
yLowerLim = -127
yUpperLim = 155

def densitySliceInit():
    global densitySlice, densityMask, isoCenterGlobal
    CTFile = "/data/qifan/FastDoseWorkplace/TCIAAdd/002/CT.nrrd"
    CTArray, CTHeader = nrrd.read(CTFile)
    CTArray = np.transpose(CTArray, axes=(2, 1, 0))
    CTArray += 1024
    sliceIdx = 10
    densitySlice = CTArray[sliceIdx, :, :]
    isoCenterGlobal = np.array(densitySlice.shape) / 2

    MaskFile = "/data/qifan/FastDoseWorkplace/TCIAAdd/002/RTSTRUCT.nrrd"
    MaskArray, MaskHeader = nrrd.read(MaskFile)
    idx = 0
    while True:
        keyHead = "Segment{}_".format(idx)
        idx += 1
        nameKey = keyHead + "Name"
        if nameKey in MaskHeader:
            name = MaskHeader[nameKey]
            if name == "SKIN":
                LayerKey = keyHead + "Layer"
                Layer = int(MaskHeader[LayerKey])
                LabelKey = keyHead + "LabelValue"
                Label = int(MaskHeader[LabelKey])
                mask_array = MaskArray[Layer] == Label
                mask_array = np.transpose(mask_array, axes=(2, 1, 0))
                densityMask = mask_array[sliceIdx, :, :]
                break
        else:
            break

    if False:
        plt.imshow(densitySlice, cmap="gray", vmin=500, vmax=1500)
        plt.imshow(densityMask, alpha=0.3)
        densitySliceFile = os.path.join(topicFolder, "densitySlice.png")
        plt.savefig(densitySliceFile)
        plt.clf()


def convertFromCartesianToBEV(coordsCartesian, isocenter, angle, lowerX, lowerY):
    # coordsCartesian: (n, 2)
    # isocenter: (2, )
    # angle: float, rad
    # limMin: int(float)
    # limMax: int(float)
    # here we use the coords order (width, height)

    axis = np.array((-np.sin(angle), np.cos(angle)))
    axis = np.expand_dims(axis, axis=0)
    paraAxis = np.array((np.cos(angle), np.sin(angle)))
    paraAxis = np.expand_dims(paraAxis, axis=0)
    IsoCenter = np.expand_dims(isocenter, axis=0)
    coordsCartesian_minus_isocenter = coordsCartesian - IsoCenter
    projectionAlongAxis = np.sum(coordsCartesian_minus_isocenter * axis,
        axis=1, keepdims=True)
    projectionAlongParaAxis = np.sum(coordsCartesian_minus_isocenter * paraAxis,
        axis=1, keepdims=True)
    
    point2SourceDistance = projectionAlongAxis + SAD
    BEV_x_coords = projectionAlongParaAxis * SAD / point2SourceDistance - lowerX
    BEV_y_coords = projectionAlongAxis - lowerY
    result = np.concatenate((BEV_x_coords, BEV_y_coords), axis=1)
    return result


def convertFromBEVToCartesian(coordsBEV, isocenter, angle, lowerX, lowerY):
    axis = np.array((-np.sin(angle), np.cos(angle)))
    axis = np.expand_dims(axis, axis=0)
    paraAxis = np.array((np.cos(angle), np.sin(angle)))
    paraAxis = np.expand_dims(paraAxis, axis=0)

    componentX = coordsBEV[:, 0] + lowerX
    componentY = coordsBEV[:, 1] + lowerY
    componentX = componentX * (componentY + SAD) / SAD
    
    componentX = np.expand_dims(componentX, axis=1)
    componentY = np.expand_dims(componentY, axis=1)

    coords = componentX * paraAxis + componentY * axis

    IsoCenter = np.expand_dims(isocenter, axis=0)
    coords = coords + IsoCenter
    return coords


def calcBEVLim():
    # calculate the active area
    densitySliceShape = densitySlice.shape
    coordsX = np.arange(densitySliceShape[1])
    coordsX = np.expand_dims(coordsX, axis=0)
    coordsY = np.arange(densitySliceShape[0])
    coordsY = np.expand_dims(coordsY, axis=1)
    coords = np.zeros(densitySliceShape + (2,))
    coords[:, :, 0] = coordsX
    coords[:, :, 1] = coordsY

    nElements = densitySliceShape[0] * densitySliceShape[1]
    coords = np.reshape(coords, (nElements, 2))
    densityMaskLinear = np.reshape(densityMask, (nElements,))
    bodyCoordsCartesian = coords[densityMaskLinear, :]
    bodyCoordsBEV = convertFromCartesianToBEV(
        bodyCoordsCartesian, isoCenterGlobal, beamAngleGlobal, 0, 0)
    
    bodyCoordsBEV_componentX = bodyCoordsBEV[:, 0]
    validFlag = np.logical_and(bodyCoordsBEV_componentX >= - halfXDim,
        bodyCoordsBEV_componentX <= halfXDim)

    validBEVComponentsY = bodyCoordsBEV[:, 1]
    validBEVComponentsY = validBEVComponentsY[validFlag]
    print(np.min(validBEVComponentsY), np.max(validBEVComponentsY))


def getBEVMat():
    """
    This function calculates the BEV coordinates in the Cartesian
    coordinate system, and get the density value
    """
    BEVMatHeight = yUpperLim - yLowerLim
    BEVMatWidth = 2 * halfXDim
    BEVMatShape = (BEVMatHeight, BEVMatWidth)
    BEVCoordsX = np.arange(BEVMatWidth)
    BEVCoordsX = np.expand_dims(BEVCoordsX, axis=0)
    BEVCoordsY = np.arange(BEVMatHeight)
    BEVCoordsY = np.expand_dims(BEVCoordsY, axis=1)
    BEVCoords = np.zeros(BEVMatShape + (2,))
    BEVCoords[:, :, 0] = BEVCoordsX
    BEVCoords[:, :, 1] = BEVCoordsY

    BEVCoords = np.reshape(BEVCoords, (BEVMatHeight * BEVMatWidth, 2))
    CartesianCoords = convertFromBEVToCartesian(
        BEVCoords, isoCenterGlobal, beamAngleGlobal, -halfXDim, yLowerLim)
    
    densitySliceInterpolator = RegularGridInterpolator(
        (np.arange(densitySlice.shape[1]), np.arange(densitySlice.shape[0])),
        np.transpose(densitySlice, axes=(1, 0)))
    BEVDensity = densitySliceInterpolator(CartesianCoords)
    BEVDensity = np.reshape(BEVDensity, BEVMatShape)

    if True:
        BEVDensityFile = os.path.join(topicFolder, "BEVDensity.npy")
        np.save(BEVDensityFile, BEVDensity)
    else:
        BEVDensityFile = os.path.join(topicFolder, "BEVDensity.png")
        plt.imsave(BEVDensityFile, BEVDensity, cmap="gray", vmin=500, vmax=1500)


def drawTermaOnPVCS():
    # firstly, we calculate the four edges of the BEV bounding box
    BEVMatHeight = yUpperLim - yLowerLim
    BEVMatWidth = 2 * halfXDim

    fig, ax = plt.subplots(figsize=(densitySlice.shape[0]/100, densitySlice.shape[1]/100), dpi=100)
    numSegmentsX = 3
    incrementalX = BEVMatWidth / numSegmentsX
    numSegmentsY = 15
    incrementalY = BEVMatHeight / numSegmentsY
    leftPoints = np.zeros((numSegmentsY + 1, 2))
    leftPoints[:, 1] = np.arange(numSegmentsY + 1) * incrementalY
    leftPoints[:, 0] = 0
    rightPoints = leftPoints.copy()
    rightPoints[:, 0] = BEVMatWidth

    ax.imshow(densitySlice, cmap="gray", vmin=500, vmax=1500)

    boundingBoxColor = colors[2]
    leftPointsCartesian = convertFromBEVToCartesian(
        leftPoints, np.array(densitySlice.shape)/2, beamAngleGlobal, -halfXDim, yLowerLim)
    rightPointsCartesian = convertFromBEVToCartesian(
        rightPoints, np.array(densitySlice.shape)/2, beamAngleGlobal, -halfXDim, yLowerLim)
    for i in range(numSegmentsY + 1):
        start = leftPointsCartesian[i, :]
        end = rightPointsCartesian[i, :]
        ax.plot((start[0], end[0]), (start[1], end[1]), linestyle="--", color=boundingBoxColor)

    topPoints = np.zeros((numSegmentsX + 1, 2))
    topPoints[:, 0] = np.arange(numSegmentsX + 1) * incrementalX
    topPoints[:, 1] = 0
    bottomPoints = topPoints.copy()
    bottomPoints[:, 1] = BEVMatHeight

    topPointsCartesian = convertFromBEVToCartesian(
        topPoints, np.array(densitySlice.shape)/2, beamAngleGlobal, -halfXDim, yLowerLim)
    bottomPointsCartesian = convertFromBEVToCartesian(
        bottomPoints, np.array(densitySlice.shape)/2, beamAngleGlobal, -halfXDim, yLowerLim)
    for i in range(numSegmentsX + 1):
        start = topPointsCartesian[i, :]
        end = bottomPointsCartesian[i, :]
        ax.plot((start[0], end[0]), (start[1], end[1]), linestyle="--", color=boundingBoxColor)

    global spectrumFile
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

    # calculate TERMA
    BEVDensityArray = os.path.join(topicFolder, "BEVDensity.npy")
    BEVDensity = np.load(BEVDensityArray)
    BEVDensity = BEVDensity / 1024  # relative to water
    BEVTerma = np.zeros_like(BEVDensity)
    pathLength = np.ones((BEVMatWidth,))
    voxelSize = 0.108  # cm

    for i in range(BEVMatHeight):
        for mu_value in mu:
            BEVTerma[i, :] += np.exp(-pathLength * voxelSize * mu_value)
        pathLength += BEVDensity[i, :]

    idxStart = int(BEVMatWidth / 3)
    idxEnd = int(BEVMatWidth * 2 / 3)
    BEVTerma[:, :idxStart] = 0
    BEVTerma[:, idxEnd:] = 0

    # interpolate the Cartesian Terma from BEV Terma
    densitySliceShape = densitySlice.shape
    coordsX = np.arange(densitySliceShape[1])
    coordsX = np.expand_dims(coordsX, axis=0)
    coordsY = np.arange(densitySliceShape[0])
    coordsY = np.expand_dims(coordsY, axis=1)
    coords = np.zeros(densitySliceShape + (2,))
    coords[:, :, 0] = coordsX
    coords[:, :, 1] = coordsY
    coords = np.reshape(coords, (densitySliceShape[0] * densitySliceShape[1], 2))
    coordsBEV = convertFromCartesianToBEV(coords, np.array(densitySliceShape)/2,
        beamAngleGlobal, -halfXDim, yLowerLim)
    BEVTermaInterpolator = RegularGridInterpolator(
        (np.arange(BEVMatWidth), np.arange(BEVMatHeight)), np.transpose(BEVTerma),
        bounds_error=False, fill_value=0)
    CartesianTerma = BEVTermaInterpolator(coordsBEV)
    CartesianTerma = np.reshape(CartesianTerma, densitySliceShape)
    CartesianTerma[np.logical_not(densityMask)] = 0

    if False:
        plt.imshow(CartesianTerma, cmap="jet", vmin=0, vmax=np.max(CartesianTerma),
            alpha=0.8*(CartesianTerma>1e-4))
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figurePath = os.path.join(topicFolder, "CartesianDensity.png")
    plt.savefig(figurePath)
    plt.close(fig)
    plt.clf()

    BEVTermaFile = os.path.join(topicFolder, "BEVTerma.npy")
    np.save(BEVTermaFile, BEVTerma)


def drawTermaBEV():
    BEVMatHeight = yUpperLim - yLowerLim
    BEVMatWidth = 2 * halfXDim
    boundingColor = colors[2]

    densityBEV = os.path.join(topicFolder, "BEVDensity.npy")
    densityBEV = np.load(densityBEV)
    TermaBEV = os.path.join(topicFolder, "BEVTerma.npy")
    TermaBEV = np.load(TermaBEV)

    fig, ax = plt.subplots(figsize=(densityBEV.shape[1]/20, densityBEV.shape[0]/20), dpi=100)
    ax.imshow(densityBEV, cmap="gray", vmin=500, vmax=1500)
    ax.imshow(TermaBEV, cmap="jet", vmin=0, vmax=np.max(TermaBEV), alpha=0.5*(TermaBEV > 1e-4))

    numSegmentsX = 3
    incrementalX = BEVMatWidth / numSegmentsX
    numSegmentsY = 15
    incrementalY = BEVMatHeight / numSegmentsY

    eps = 1
    LineWidth = 4
    for i in range(1, numSegmentsY):
        x_points = (eps, BEVMatWidth - eps)
        y_points = (i * incrementalY, i * incrementalY)
        ax.plot(x_points, y_points, linestyle="--", color=boundingColor, linewidth=LineWidth)
    
    for i in range(1, numSegmentsX):
        x_points = (i * incrementalX - eps, i * incrementalX - eps)
        y_points = (eps, BEVMatHeight - eps)
        ax.plot(x_points, y_points, linestyle="--", color=boundingColor, linewidth=LineWidth)
    
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figurePath = os.path.join(topicFolder, "BEVDensity.png")
    plt.savefig(figurePath)
    plt.close(fig)
    plt.clf()


def VaryingTheta():
    """
    This figure draws the figure of rays with varying angle but fixed location
    """
    densityBEVFile = os.path.join(topicFolder, "BEVDensity.npy")
    densityBEV = np.load(densityBEVFile)
    figureHeight, figureWidth = densityBEV.shape[:2]
    
    # now we make a square crop of the density
    # yOffset = figureHeight - figureWidth
    yOffset = 0
    densityCrop = densityBEV[yOffset: figureWidth+yOffset, :]

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.imshow(densityCrop, cmap="gray", vmin=500, vmax=1500)


    # draw the rays
    eps = 1
    for i in range(4):
        thetaAngle = (i + 0.5) * np.pi / 8
        direction = np.array((np.sin(thetaAngle), np.cos(thetaAngle)))
        sourcePoint = np.array((figureWidth / 2, 0))
        scaleLimit = figureWidth / direction[1]
        # calculate the intersection points of X
        scaleList = [0]
        j = 1
        while True:
            currentScale = figureWidth * (j / 2) / direction[0]
            j += 2
            if currentScale > scaleLimit:
                break
            scaleList.append(currentScale)
        scaleList.append(scaleLimit)

        # print(sourcePoint, scaleList)
        # return
        
        for j in range(len(scaleList) - 1):
            scaleBegin = scaleList[j]
            scaleEnd = scaleList[j + 1]
            pointBegin = sourcePoint + scaleBegin * direction
            pointBegin[0] = pointBegin[0] % figureWidth
            pointEnd = sourcePoint + scaleEnd * direction
            pointEnd[0] = (pointEnd[0] - eps) % figureWidth
            ax.plot((pointBegin[0], pointEnd[0]), (pointBegin[1], pointEnd[1]-eps),
                color=colors[i], linewidth=2)


    nBlocks = 5
    blockSize = figureWidth / nBlocks
    septaWidth = 3
    septaColor = "white"
    eps = 1
    for i in range(nBlocks + 1):
        if i == nBlocks:
            ax.plot((figureWidth-eps, figureWidth-eps), (0, figureWidth-eps),
                linewidth=septaWidth, color=septaColor)
            ax.plot((0, figureWidth-eps), (figureWidth-eps, figureWidth-eps),
                linewidth=septaWidth, color=septaColor)
        else:
            ax.plot((i * blockSize, i * blockSize), (eps, figureWidth-eps),
                linewidth=septaWidth, color=septaColor)
            ax.plot((0, figureWidth-eps), (i * blockSize, i * blockSize),
                linewidth=septaWidth, color=septaColor)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figureFile = os.path.join(topicFolder, "varyingAngles.png")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()


def VaryingX():
    densityBEVFile = os.path.join(topicFolder, "BEVDensity.npy")
    densityBEV = np.load(densityBEVFile)
    figureHeight, figureWidth = densityBEV.shape[:2]
    
    # now we make a square crop of the density
    # yOffset = figureHeight - figureWidth
    yOffset = 0
    densityCrop = densityBEV[yOffset: figureWidth+yOffset, :]

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.imshow(densityCrop, cmap="gray", vmin=500, vmax=1500)

    theta = 1.5 * np.pi / 8
    direction = np.array((np.sin(theta), np.cos(theta)))
    nBlocks = 5
    blockSize = figureWidth / nBlocks

    # calculate the x intersection scales
    scaleListX = []
    for j in range(1, nBlocks+1):
        localScale = j * blockSize / direction[1]
        scaleListX.append(localScale)
    scaleLimit = scaleListX[-1]
    # calculate the y intersection scales
    scaleListY = []
    j = 1
    while True:
        localScale = j / 2 * blockSize / direction[0]
        j += 2
        if localScale > scaleLimit:
            break
        scaleListY.append(localScale)
    # merge scaleListX and scaleListY
    scaleList = scaleListX + scaleListY
    scaleList.sort()
    scaleList.insert(0, 0)
    
    eps = 1
    rayWidth = 2
    for i in range(nBlocks):
        sourcePoint = np.array(((i+0.5) * blockSize, 0))
        for j in range(len(scaleList) - 1):
            pointBegin = sourcePoint + scaleList[j] * direction
            pointEnd = sourcePoint + (scaleList[j+1] - eps) * direction
            middleX = (pointBegin[0] + pointEnd[0]) / 2
            module = middleX // figureWidth
            offset = module * figureWidth
            pointBegin[0] -= offset
            pointEnd[0] -= offset
            if j % 2 == 0:
                ax.plot((pointBegin[0], pointEnd[0]), (pointBegin[1], pointEnd[1]),
                    color=colors[i], linewidth=rayWidth)
            else:
                ax.plot((pointBegin[0], pointEnd[0]), (pointBegin[1], pointEnd[1]),
                    color=colors[i], linewidth=rayWidth, linestyle="--")

    septaWidth = 3
    septaColor = "white"
    eps = 1
    for i in range(nBlocks + 1):
        if i == nBlocks:
            ax.plot((figureWidth-eps, figureWidth-eps), (0, figureWidth-eps),
                linewidth=septaWidth, color=septaColor)
            ax.plot((0, figureWidth-eps), (figureWidth-eps, figureWidth-eps),
                linewidth=septaWidth, color=septaColor)
        else:
            ax.plot((i * blockSize, i * blockSize), (eps, figureWidth-eps),
                linewidth=septaWidth, color=septaColor)
            ax.plot((0, figureWidth-eps), (i * blockSize, i * blockSize),
                linewidth=septaWidth, color=septaColor)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figureFile = os.path.join(topicFolder, "varyingDisplacement.png")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()


def RayPlots():
    """
    This function generates the ray plots in both BEV and Cartesian coordinates
    """
    BEVDensityFile = os.path.join(topicFolder, "BEVDensity.npy")
    BEVDensity = np.load(BEVDensityFile)
    BEVHeight, BEVWidth = BEVDensity.shape[: 2]
    nBlocksWidth = 3

    BlockSize = BEVWidth / nBlocksWidth
    result = {}
    stepSize = 1
    nAngles = 3
    for i in range(nAngles):
        theta = (i + 0.5) * np.pi / 8
        direction = np.array((np.sin(theta), np.cos(theta)))
        for j in range(nBlocksWidth):
            angleDisplacementList = []
            currentPoint = np.array(((j + 0.5) * BlockSize, 0))
            localLineList = [currentPoint]
            while True:
                currentPoint = currentPoint + direction * stepSize
                if currentPoint[1] > BEVHeight:
                    break

                # tell if lastPoint and currentPoint is within the same region
                if currentPoint[0] // BEVWidth == 0:
                    localLineList.append(currentPoint)
                else:
                    localLineList = [np.expand_dims(a, axis=0) for a in localLineList]
                    lineSegment = np.concatenate(localLineList, axis=0)
                    angleDisplacementList.append(lineSegment)
                    localLineList.clear()
                    currentPoint[0] -= BEVWidth
                    localLineList.append(currentPoint)

            # the last line segment
            localLineList = [np.expand_dims(a, axis=0) for a in localLineList]
            lineSegment = np.concatenate(localLineList, axis=0)
            angleDisplacementList.append(lineSegment)
            key = (i, j)
            result[key] = angleDisplacementList
    
    fig, ax = plt.subplots(figsize=(BEVWidth/25, BEVHeight/25), dpi=100)
    ax.imshow(BEVDensity, cmap="gray", vmin=500, vmax=1500)
    for i in range(nAngles):
        for j in range(nBlocksWidth):
            localResult = result[(i, j)]
            for line in localResult:
                lineLocal = line[:-3, :]
                ax.plot(lineLocal[:, 0], lineLocal[:, 1], color=colors[i], linewidth=3)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figureFile = os.path.join(topicFolder, "rayPlotsBEV.png")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()

    sliceHeight, sliceWidth = densitySlice.shape
    fig, ax = plt.subplots(figsize=(sliceWidth/20, sliceHeight/20), dpi=100)
    ax.imshow(densitySlice, cmap="gray", vmin=500, vmax=1500)
    for i in range(nAngles):
        for j in range(nBlocksWidth):
            localResult = result[(i, j)]
            for line in localResult:
                lineCartesian = convertFromBEVToCartesian(
                    line, isoCenterGlobal, beamAngleGlobal, -BEVWidth/2, yLowerLim)
                ax.plot(lineCartesian[:, 0], lineCartesian[:, 1], color=colors[i], linewidth=4)
    # plot the boundary
    pointsBoundary = np.array((
        (0, 0),
        (BEVWidth, 0),
        (BEVWidth, BEVHeight),
        (0, BEVHeight)
    ))
    pointsBoundary = convertFromBEVToCartesian(
        pointsBoundary, isoCenterGlobal, beamAngleGlobal, -BEVWidth/2, yLowerLim)
    nPoints = 4
    for i in range(nPoints):
        pointBegin = pointsBoundary[i, :]
        pointEnd = pointsBoundary[(i+1) % nPoints, :]
        ax.plot((pointBegin[0], pointEnd[0]), (pointBegin[1], pointEnd[1]),
            linestyle="--", color=colors[2], linewidth=6)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figureFile = os.path.join(topicFolder, "rayPlotsCartesian.png")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()


def varyingPhi3D():
    """
    In this function, we draw a plot of rays of the same theta but varying phi values
    """
    resultDict = {}
    numTheta = 3
    numPhi = 8
    dimX = 1
    dimY = 1
    dimZ = 4
    eps = 1e-4
    for j in range(numTheta):
        theta = (j + 0.5) * np.pi / 8
        for i in range(numPhi):
            phi = i * 2 * np.pi / numPhi
            direction = np.array((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))
            # calculate the scale factors
            scaleLimit = dimZ / direction[2]

            scaleXList = []
            scaleStrideX = dimX / (np.sign(direction[0] + eps) * (abs(direction[0]) + eps))
            scaleStrideX = abs(scaleStrideX)
            idx = 0.5
            while True:
                currentScale = idx * scaleStrideX
                idx += 1
                if currentScale > scaleLimit:
                    break
                scaleXList.append(currentScale)
            
            scaleYList = []
            scaleStrideY = dimY / (np.sign(direction[1] + eps) * (abs(direction[1]) + eps))
            scaleStrideY = abs(scaleStrideY)
            idx = 0.5
            while True:
                currentScale = idx * scaleStrideY
                idx += 1
                if currentScale > scaleLimit:
                    break
                scaleYList.append(currentScale)

            # mix scaleXList and scaleYList
            scaleList = scaleXList + scaleYList
            scaleList.append(scaleLimit)
            scaleList.append(0)
            scaleList.sort()
            resultDict[(j, i)] = scaleList
    

    pygame.init()
    display = (400, 800)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -6)
    xRotateAngle = 120  #degree
    zRotateAngle = 30

    glRotatef(xRotateAngle, 1, 0, 0)
    xRotateAngleRad = xRotateAngle * np.pi / 180
    zAxis = np.array((0, -np.sin(xRotateAngleRad), np.cos(xRotateAngleRad)))
    glRotatef(zRotateAngle, 0, 0, 1)

    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_DEPTH_TEST)
    glDepthMask(GL_FALSE)

    clock = pygame.time.Clock()
    
    for k in range(2):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        drawLattice(dimX, dimY, dimZ)
        for j in range(numTheta):
            rayColor = hex_to_rgb(colors[j+1])
            theta = (j + 0.5) * np.pi / 8
            for i in range(numPhi):
                phi = i * 2 * np.pi / numPhi
                scaleList = resultDict[(j, i)]
                drawRays(dimX, dimY, dimZ, theta, phi, scaleList, rayColor)
        pygame.display.flip()
        clock.tick(60)

    buffer = glReadPixels(0, 0, display[0], display[1], GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(buffer, dtype=np.uint8).reshape((display[1], display[0], 3))
    image = np.flipud(image)
    plt.imsave("./cube.png", image)


def drawRays(dimX, dimY, dimZ, theta, phi, scaleList, rayColor):
    startingPoint = np.array((0, 0, -dimZ/2))
    direction = np.array((np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi), np.cos(theta)))

    glColor3fv(rayColor)
    glLineWidth(2)
    glBegin(GL_LINES)
    for i in range(len(scaleList) - 1):
        scale0 = scaleList[i]
        scale1 = scaleList[i + 1]
        point0 = startingPoint + scale0 * direction
        point1 = startingPoint + scale1 * direction
        middlePoint = (point0 + point1) / 2

        middlePointX = middlePoint[0]
        middleModuleX = np.floor((middlePointX + dimX / 2) / dimX)
        point0[0] -= middleModuleX * dimX
        point1[0] -= middleModuleX * dimX
        
        middlePointY = middlePoint[1]
        middleModuleY = np.floor((middlePointY + dimY / 2) / dimY)
        point0[1] -= middleModuleY * dimY
        point1[1] -= middleModuleY * dimY

        glVertex3fv(point0)
        glVertex3fv(point1)
    glEnd()



def drawLattice(dimX, dimY, dimZ):
    point1 = (-dimX / 2, -dimY / 2, -dimZ / 2)
    point2 = (-dimX / 2, dimY / 2, -dimZ / 2)
    point3 = (dimX / 2, dimY / 2, -dimZ / 2)
    point4 = (dimX / 2, -dimY / 2, -dimZ / 2)
    point5 = (dimX / 2, -dimY / 2, dimZ / 2)
    point6 = (dimX / 2, dimY / 2, dimZ / 2)
    point7 = (-dimX / 2, dimY / 2, dimZ / 2)
    point8 = (-dimX / 2, -dimY / 2, dimZ / 2)
    points = [point1, point2, point3, point4, point5, point6, point7, point8]

    lineColor = hex_to_rgb(colors[0])
    glColor3fv(lineColor)
    glLineWidth(3)
    glBegin(GL_LINES)
    for i in range(4):
        point1 = points[i]
        point2 = points[(i + 1) % 4]
        glVertex3fv(point1)
        glVertex3fv(point2)

    for i in range(4):
        point1 = points[i + 4]
        point2 = points[(i + 1) % 4 + 4]
        glVertex3fv(point1)
        glVertex3fv(point2)

    for i in range(4):
        point1 = points[3 - i]
        point2 = points[4 + i]
        glVertex3fv(point1)
        glVertex3fv(point2)
    glEnd()
    
    faceColor = lineColor + (0.25, )
    glColor4fv(faceColor)
    glBegin(GL_QUADS)
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 6, 7),
        (1, 2, 5, 6),
        (2, 3, 4, 5),
        (3, 0, 7, 4)
    ]
    for face in faces:
        for idx in face:
            glVertex3fv(points[idx])
    glEnd()


def hex_to_rgb(hex_color):
    """Converts a color from hexadecimal format to RGB."""
    hex_color = hex_color.lstrip('#')
    result = tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))
    return result


def finalGroup():
    verticalGap = 64
    horizontalGap = 64
    
    sliceHeightTarget = 1024
    sliceHeightBegin = 232
    sliceHeightCrop = 620

    CartesianDensity = os.path.join(topicFolder, "CartesianDensity.png")
    CartesianDensity = plt.imread(CartesianDensity)
    CartesianDensity = resize(CartesianDensity, (sliceHeightTarget, sliceHeightTarget))
    CartesianDensity = CartesianDensity[
        sliceHeightBegin: sliceHeightBegin + sliceHeightCrop, :, :]
    
    rayPlotsCartesian = os.path.join(topicFolder, "rayPlotsCartesian.png")
    rayPlotsCartesian = plt.imread(rayPlotsCartesian)
    rayPlotsCartesian = resize(rayPlotsCartesian, (sliceHeightTarget, sliceHeightTarget))
    rayPlotsCartesian = rayPlotsCartesian[
        sliceHeightBegin: sliceHeightBegin + sliceHeightCrop, :, :]
    
    gap1Shape = (verticalGap, CartesianDensity.shape[1], 4)
    gap1 = np.zeros(gap1Shape, dtype=rayPlotsCartesian.dtype)
    gap1[:, :, 3] = 1.0
    columnOne = np.concatenate((CartesianDensity, gap1, rayPlotsCartesian), axis=0)

    globalHeight = columnOne.shape[0]
    horizontalGap = (globalHeight, horizontalGap, 4)
    horizontalGap = np.zeros(horizontalGap, dtype=CartesianDensity.dtype)
    horizontalGap[:, :, 3] = 1

    BEVDensity = os.path.join(topicFolder, "BEVDensity.png")
    BEVDensity = plt.imread(BEVDensity)
    BEVDensityHeight, BEVDensityWidth = BEVDensity.shape[:2]
    BEVDensityWidth = BEVDensityWidth * (globalHeight / BEVDensityHeight)
    BEVDensityWidth = int(BEVDensityWidth)
    BEVDensity = resize(BEVDensity, (globalHeight, BEVDensityWidth))

    rayPlotsBEV = os.path.join(topicFolder, "rayPlotsBEV.png")
    rayPlotsBEV = plt.imread(rayPlotsBEV)
    rayPlotsBEVHeight, rayPlotsBEVWidth = rayPlotsBEV.shape[:2]
    rayPlotsBEVWidth = rayPlotsBEVWidth * (globalHeight / rayPlotsBEVHeight)
    rayPlotsBEVWidth = int(rayPlotsBEVWidth)
    rayPlotsBEV = resize(rayPlotsBEV, (globalHeight, rayPlotsBEVWidth))

    varyingAngles = os.path.join(topicFolder, "varyingAngles.png")
    varyingAngles = plt.imread(varyingAngles)
    varyingAngles = resize(varyingAngles, (sliceHeightCrop, sliceHeightCrop))

    varyingDisplacement = os.path.join(topicFolder, "varyingDisplacement.png")
    varyingDisplacement = plt.imread(varyingDisplacement)
    varyingDisplacement = resize(varyingDisplacement, (sliceHeightCrop, sliceHeightCrop))

    column4Gap = (verticalGap, sliceHeightCrop, 4)
    column4Gap = np.zeros(column4Gap, dtype=CartesianDensity.dtype)
    column4Gap[:, :, 3] = 1.0
    column4 = np.concatenate((varyingAngles, column4Gap, varyingDisplacement), axis=0)

    varyingAngles3D = os.path.join(topicFolder, "varyingAngles3D.png")
    varyingAngles3D = plt.imread(varyingAngles3D)
    varyingAngles3DHeight, varyingAngles3DWidth = varyingAngles3D.shape[:2]
    varyingAngles3DWidthNew = int(varyingAngles3DWidth * (globalHeight / varyingAngles3DHeight))
    varyingAngles3D = resize(varyingAngles3D, (globalHeight, varyingAngles3DWidthNew))

    resultFigure = np.concatenate((columnOne, horizontalGap, BEVDensity,
        horizontalGap, rayPlotsBEV, horizontalGap, column4, horizontalGap, varyingAngles3D), axis=1)
    
    fig, ax = plt.subplots(figsize=(resultFigure.shape[1]/100, resultFigure.shape[0]/100), dpi=100)
    ax.imshow(resultFigure)

    # label the subfigures
    horizontalOffset = 10
    verticalOffset = 10
    fontsize = 48
    fontcolor = "white"
    coords_a = (horizontalOffset, verticalOffset)
    ax.text(coords_a[0], coords_a[1], "(a)", fontsize=fontsize,
            color=fontcolor, ha="left", va="top")
    
    coords_b = (horizontalOffset, verticalOffset + sliceHeightCrop + verticalGap)
    ax.text(coords_b[0], coords_b[1], "(d)", fontsize=fontsize,
            color=fontcolor, ha="left", va="top")
    
    coords_c = (horizontalOffset + columnOne.shape[1] + horizontalGap.shape[1], verticalOffset)
    ax.text(coords_c[0], coords_c[1], "(b)", fontsize=fontsize,
            color=fontcolor, ha="left", va="top")
    
    coords_d = (coords_c[0] + BEVDensity.shape[1] + horizontalGap.shape[1], verticalOffset)
    ax.text(coords_d[0], coords_d[1], "(c)", fontsize=fontsize,
            color=fontcolor, ha="left", va="top")

    coords_e = (coords_d[0] + rayPlotsBEV.shape[1] + horizontalGap.shape[1], verticalOffset)
    ax.text(coords_e[0], coords_e[1], "(e)", fontsize=fontsize,
            color=fontcolor, ha="left", va="top")

    coords_f = (coords_e[0], coords_e[1] + sliceHeightCrop)
    ax.text(coords_f[0], coords_f[1], "(f)", fontsize=fontsize,
            color=fontcolor, ha="left", va="top")

    coords_g = (coords_e[0] + column4.shape[1] + horizontalGap.shape[1], verticalOffset)
    ax.text(coords_g[0], coords_g[1], "(g)", fontsize=fontsize,
            color=fontcolor, ha="left", va="top")
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    targetFile = os.path.join(topicFolder, "BeamletDosecalcGroup.png")
    # plt.imsave(targetFile, resultFigure)
    plt.savefig(targetFile)
    plt.close(fig)
    plt.clf()

    
if __name__ == '__main__':
    # densitySliceInit()
    # calcBEVLim()
    # getBEVMat()
    # drawTermaOnPVCS()
    # drawTermaBEV()
    # VaryingTheta()
    # VaryingX()
    # RayPlots()
    # varyingPhi3D()
    finalGroup()