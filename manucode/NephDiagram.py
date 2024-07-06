import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import nrrd
from scipy.interpolate import RegularGridInterpolator
import string

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
figuresFolder = "/data/qifan/AAPM2024/manufigures"
topicFolder = os.path.join(figuresFolder, "CCCSNeph")
if not os.path.isdir(topicFolder):
    os.mkdir(topicFolder)

SAD = 1000  # mm
beamAngle = - np.pi / 3
fluenceDim = 8
fluenceSize = 20  # mm
fluenceGap = 0
fluenceStride = fluenceSize + fluenceGap * 2
voxelSize = 1.08  # mm
mu = 0.02  # used for mm

convolutionMargin = 10  # voxels

densitySlice = None
bodyMaskSlice = None
isoCoords = None  # voxels

def densitySliceInit():
    global densitySlice, bodyMaskSlice, isoCoords
    CTFile = "/data/qifan/FastDoseWorkplace/TCIAAdd/002/CT.nrrd"
    CTArray, CTHeader = nrrd.read(CTFile)
    CTArray = np.transpose(CTArray, axes=(2, 1, 0))
    CTArray += 1024
    CTArray /= 1024  # 1.0 is the water density
    sliceIdx = 10
    densitySlice = CTArray[sliceIdx, :, :]
    isoCoords = np.array(densitySlice.shape)
    isoCoords = isoCoords / 2

    BodyMaskFile = "/data/qifan/FastDoseWorkplace/TCIAAdd/002/RTSTRUCT.nrrd"
    maskArray, maskHeader = nrrd.read(BodyMaskFile)
    idx = 0
    while True:
        nameHeader = "Segment{}_".format(idx)
        idx += 1
        key = nameHeader + "Name"
        if key not in maskHeader:
            break
        name = maskHeader[key]
        if name == "SKIN":
            layer = int(maskHeader[nameHeader + "Layer"])
            label = int(maskHeader[nameHeader + "LabelValue"])
            bodyMaskArray = maskArray[layer, :, :, :] == label
            bodyMaskArray = np.transpose(bodyMaskArray, axes=(2, 1, 0))
            bodyMaskSlice = bodyMaskArray[sliceIdx, :, :]
            break


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return np.array(tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)))


def TermaCartesian():
    """
    This function generates the images
    """
    height, width = densitySlice.shape
    sourceAxis = np.array((np.cos(beamAngle), np.sin(beamAngle)))
    sourceAxis = np.expand_dims(sourceAxis, axis=(0, 1))
    lateralAxis = np.array((- np.sin(beamAngle), np.cos(beamAngle)))
    lateralAxis = np.expand_dims(lateralAxis, axis=(0, 1))

    voxelCoordsShape = (height, width, 2)
    voxelCoords = np.zeros(voxelCoordsShape, float)
    voxelCoordsX = np.arange(width) + 0.5
    voxelCoordsX = np.expand_dims(voxelCoordsX, axis=0)
    voxelCoordsY = np.arange(height) + 0.5
    voxelCoordsY = np.expand_dims(voxelCoordsY, axis=1)
    voxelCoords[:, :, 0] = voxelCoordsX
    voxelCoords[:, :, 1] = voxelCoordsY
    isoCoordsExpand = np.expand_dims(isoCoords, axis=(0, 1))
    voxelCoords -= isoCoordsExpand

    sourceAxisProjection = np.sum(voxelCoords * sourceAxis, axis=2)
    distanceToSource = SAD / voxelSize - sourceAxisProjection

    lateralAxisProjection = np.sum(voxelCoords * lateralAxis, axis=2)
    lateralAxisNormalized = lateralAxisProjection / distanceToSource * SAD / voxelSize

    fig, ax = plt.subplots(figsize=(height/100, width/100), dpi=100)
    ax.imshow(densitySlice, cmap="gray", vmin=0.5, vmax=1.5)

    OnList = []
    for i in range(fluenceDim):
        offsetStart = (- fluenceDim / 2  + i) * fluenceStride + fluenceGap
        offsetStart /= voxelSize
        offsetEnd = (- fluenceDim / 2 + (i + 1)) * fluenceStride - fluenceGap
        offsetEnd /= voxelSize
        voxelsOn = np.logical_and(lateralAxisNormalized >= offsetStart,
            lateralAxisNormalized <= offsetEnd)
        OnList.append(voxelsOn)
        voxelsOnColor = np.expand_dims(voxelsOn, axis=2)
        color = colors[i]
        color = hex_to_rgb(color)
        color = np.expand_dims(color, axis=(0, 1))
        voxelsOnColor = (voxelsOnColor * color).astype(np.uint8)
        onFlag = np.logical_and(voxelsOn, bodyMaskSlice)
        transparencyChannel = np.zeros((height, width, 1), dtype=np.uint8)
        transparencyChannel[onFlag] = np.uint8(255 * 0.5)
        voxelsOnColor = np.concatenate((voxelsOnColor, transparencyChannel), axis=2)
        ax.imshow(voxelsOnColor)
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    imageFigure = os.path.join(topicFolder, "CartesianRay.png")
    plt.savefig(imageFigure)
    plt.close(fig)
    plt.clf()


def contextBundle():
    height, width = densitySlice.shape
    sourceAxis = np.array((np.cos(beamAngle), np.sin(beamAngle)))
    sourceAxis = np.expand_dims(sourceAxis, axis=(0, 1))
    lateralAxis = np.array((- np.sin(beamAngle), np.cos(beamAngle)))
    lateralAxis = np.expand_dims(lateralAxis, axis=(0, 1))

    voxelCoordsShape = (height, width, 2)
    voxelCoords = np.zeros(voxelCoordsShape, float)
    voxelCoordsX = np.arange(width) + 0.5
    voxelCoordsX = np.expand_dims(voxelCoordsX, axis=0)
    voxelCoordsY = np.arange(height) + 0.5
    voxelCoordsY = np.expand_dims(voxelCoordsY, axis=1)
    voxelCoords[:, :, 0] = voxelCoordsX
    voxelCoords[:, :, 1] = voxelCoordsY
    isoCoordsExpand = np.expand_dims(isoCoords, axis=(0, 1))
    voxelCoords2Isocenter = voxelCoords - isoCoordsExpand

    sourceAxisProjection = np.sum(voxelCoords2Isocenter * sourceAxis, axis=2)
    distanceToSource = SAD / voxelSize - sourceAxisProjection

    lateralAxisProjection = np.sum(voxelCoords2Isocenter * lateralAxis, axis=2)
    lateralAxisNormalized = lateralAxisProjection / distanceToSource * SAD / voxelSize

    sourceCoords = isoCoords + sourceAxis * SAD / voxelSize
    longitudinalRange = None
    lateralRange = None
    paramsList = []
    debugView = False
    for i in range(fluenceDim):
        offsetStart = (- fluenceDim / 2  + i) * fluenceStride + fluenceGap
        offsetStart /= voxelSize
        offsetEnd = (- fluenceDim / 2 + (i + 1)) * fluenceStride - fluenceGap
        offsetEnd /= voxelSize
        voxelsOn = np.logical_and(lateralAxisNormalized >= offsetStart,
            lateralAxisNormalized <= offsetEnd)
        voxelsOn = np.logical_and(voxelsOn, bodyMaskSlice)

        
        if debugView:
            voxelsOnColor = np.expand_dims(voxelsOn, axis=2)
            color = colors[i]
            color = hex_to_rgb(color)
            color = np.expand_dims(color, axis=(0, 1))
            voxelsOnColor = (voxelsOnColor * color).astype(np.uint8)
            onFlag = np.logical_and(voxelsOn, bodyMaskSlice)
            transparencyChannel = np.zeros((height, width, 1), dtype=np.uint8)
            transparencyChannel[onFlag] = np.uint8(255 * 0.5)
            voxelsOnColor = np.concatenate((voxelsOnColor, transparencyChannel), axis=2)
            ax.imshow(voxelsOnColor)

        
        beamletIsocenter = isoCoords + (- fluenceDim / 2 + i + 0.5) \
            * fluenceStride / voxelSize * lateralAxis.squeeze()
        
        if debugView:
            circle = patches.Circle(beamletIsocenter, 3)
            ax.add_patch(circle)

        beamletAxis = sourceCoords - beamletIsocenter
        beamletAxis /= np.linalg.norm(beamletAxis)
        beamletAxis = beamletAxis.squeeze()
        beamletLateral = np.array((-beamletAxis[1], beamletAxis[0]))
        beamletAxis = np.expand_dims(beamletAxis, axis=(0, 1))
        beamletLateral = np.expand_dims(beamletLateral, axis=(0, 1))

        voxelCoords2BeamletIsocenter = voxelCoords - beamletIsocenter
        beamletAxisProjection = np.sum(voxelCoords2BeamletIsocenter * beamletAxis, axis=2)
        beamletAxisProjection = beamletAxisProjection[voxelsOn]
        beamletLateralProjection = np.sum(voxelCoords2BeamletIsocenter * beamletLateral, axis=2)
        beamletLateralProjection = beamletLateralProjection[voxelsOn]

        minAxisProj, maxAxisProj = np.min(beamletAxisProjection), np.max(beamletAxisProjection)
        minLateralProj, maxLateralProj = np.min(beamletLateralProjection), np.max(beamletLateralProjection)
        
        # add margin to it
        localLongitudinalRange = maxAxisProj - minAxisProj + 2 * convolutionMargin
        if longitudinalRange is None:
            longitudinalRange = localLongitudinalRange
        else:
            longitudinalRange = max(longitudinalRange, localLongitudinalRange)
            
        localLateralRange = maxLateralProj - minLateralProj + 2 * convolutionMargin
        if lateralRange is None:
            lateralRange = localLateralRange
        else:
            lateralRange = max(lateralRange, localLateralRange)

        updatedIsocenter = beamletIsocenter + (minAxisProj + maxAxisProj) / 2 * beamletAxis.squeeze()
        paramsLocal = [updatedIsocenter, beamletAxis]
        paramsList.append(paramsLocal)
        if debugView:
            circle = patches.Circle(updatedIsocenter, 3, color=colors[1])
            ax.add_patch(circle)

    
    if debugView:
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        imageFigure = os.path.join(topicFolder, "contextBundle.png")
        plt.savefig(imageFigure)
        plt.close(fig)
        plt.clf()
    
    densityInterpolator = RegularGridInterpolator(
        (np.arange(height), np.arange(width)), densitySlice)

    # extract the bounding boxes
    contextList = []
    contextDimension = (int(lateralRange), int(longitudinalRange))
    contextCoordsX = (np.arange(contextDimension[0]) + 0.5) / contextDimension[0]
    contextCoordsX = np.expand_dims(contextCoordsX, axis=(1, 2))
    contextCoordsX = np.repeat(contextCoordsX, contextDimension[1], axis=1)

    contextCoordsY = (np.arange(contextDimension[1]) + 0.5) / contextDimension[1]
    contextCoordsY = np.expand_dims(contextCoordsY, axis=(0, 2))
    contextCoordsY = np.repeat(contextCoordsY, contextDimension[0], axis=0)
    for i in range(fluenceDim):
        # calculate the four corner points of the rectangle
        centerPoint, beamletAxis = paramsList[i]
        beamletAxis = beamletAxis.squeeze()
        # print(beamletAxis)
        lateralAxis = np.array((-beamletAxis[1], beamletAxis[0]))
        
        centerPoint__ = np.expand_dims(centerPoint, axis=(0, 1))
        beamletAxis__ = np.expand_dims(beamletAxis, axis=(0, 1))
        lateralAxis__ = np.expand_dims(lateralAxis, axis=(0, 1))
        CartesianCoords = centerPoint__ + (contextCoordsX - 0.5) * lateralAxis__ * contextDimension[0] \
            - (contextCoordsY - 0.5) * beamletAxis__ * contextDimension[1]
        
        CartesianCoords = np.reshape(CartesianCoords, (contextDimension[0] * contextDimension[1], 2))
        CartesianCoords = np.flip(CartesianCoords, axis=1)
        newDensity = densityInterpolator(CartesianCoords)
        newDensity = np.reshape(newDensity, contextDimension)
        newDensity = np.transpose(newDensity, axes=(1, 0))
        contextList.append(newDensity)
    
    contextBundle = np.concatenate(contextList, axis=1)
    colorMasks = []
    alphaValue = np.uint8(0.5 * 255)
    for i in range(fluenceDim):
        color = colors[i]
        color = hex_to_rgb(color)
        color = np.insert(color, 3, alphaValue)
        color = color.astype(np.uint8)
        color = np.expand_dims(color, axis=(0, 1))
        color = np.repeat(color, contextDimension[1], axis=0)
        color = np.repeat(color, contextDimension[0], axis=1)
        colorMasks.append(color)
    colorMask = np.concatenate(colorMasks, axis=1)

    fig, ax = plt.subplots(figsize=(contextBundle.shape[1]/100, contextBundle.shape[0]/100), dpi=100)
    ax.imshow(contextBundle, cmap="gray", vmin=0.5, vmax=1.5)
    ax.imshow(colorMask)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figureFile = os.path.join(topicFolder, "contextBundle.png")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()


def contextTerma():
    """
    This function overlays the density and TERMA information into the context array
    """    
    height, width = densitySlice.shape
    sourceAxis = np.array((np.cos(beamAngle), np.sin(beamAngle)))
    sourceCoords = isoCoords + sourceAxis * SAD / voxelSize

    densityInterpolator = RegularGridInterpolator((np.arange(width), np.arange(height)),
        np.transpose(densitySlice, axes=(1, 0)))
    bodyInterpolator = RegularGridInterpolator((np.arange(width), np.arange(height)),
        np.transpose(bodyMaskSlice.astype(float), axes=(1, 0)))
    TermaArray = np.zeros((height, width), dtype=float)
    for i in range(height):
        for j in range(width):
            flag = bodyMaskSlice[i, j]
            if not flag:
                continue
            pointCoords = np.array((j, i)) + 0.5
            direction = sourceCoords - pointCoords
            direction /= np.linalg.norm(direction)
            idx = 0
            # ray tracing
            radiologicalPathLength = 0
            while True:
                coords = pointCoords + idx * direction
                idx += 1
                valid = bodyInterpolator(coords)[0]
                if valid <= 0:
                    break
                radiologicalPathLength += densityInterpolator(coords)[0]
            TermaArray[i, j] = np.exp(- radiologicalPathLength * voxelSize * mu)
        print("Row {}".format(i))
    TermaArrayFile = os.path.join(topicFolder, "termaArray.npy")
    np.save(TermaArrayFile, TermaArray)


def overlayDensityTerma():
    height, width = densitySlice.shape
    termaArray = os.path.join(topicFolder, "termaArray.npy")
    termaArray = np.load(termaArray)
    radiologicalPath = - np.log(termaArray)
    radPathFiltered = radiologicalPath[bodyMaskSlice]
    if False:
        print(np.min(radPathFiltered), np.max(radPathFiltered))
        return
    
    # scale radiological path
    mu_new = 0.00707
    termaArray = np.power(termaArray, mu_new / mu)

    fig, ax = plt.subplots(figsize=(height/100, width/100), dpi=100)
    ax.imshow(densitySlice, cmap="gray", vmin=0.5, vmax=1.5)
    ax.imshow(termaArray, cmap="jet", vmin=0, vmax=np.max(termaArray), alpha=0.3)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figureFile = os.path.join(topicFolder, "densityTerma.png")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()


def drawTermaInTheContext():
    """
    This function draws the TERMA irradiation in the BEV context array
    """
    height, width = densitySlice.shape
    TermaFullArray = os.path.join(topicFolder, "termaArray.npy")
    TermaFullArray = np.load(TermaFullArray)
    mu_new = 0.00707
    # adjust the mu value
    TermaFullArray = np.power(TermaFullArray, mu_new / mu)
    # then consider the inverse square law
    sourceAxis = np.array((np.cos(beamAngle), np.sin(beamAngle)))
    sourceCoords_1d = isoCoords + sourceAxis * SAD / voxelSize
    sourceCoords_3d = np.expand_dims(sourceCoords_1d, axis=(0, 1))
    voxelCoordsShape = (height, width, 2)
    voxelCoords = np.zeros(voxelCoordsShape, float)
    voxelCoordsX = np.arange(width) + 0.5
    voxelCoordsX = np.expand_dims(voxelCoordsX, axis=0)
    voxelCoordsY = np.arange(height) + 0.5
    voxelCoordsY = np.expand_dims(voxelCoordsY, axis=1)
    voxelCoords[:, :, 0] = voxelCoordsX
    voxelCoords[:, :, 1] = voxelCoordsY
    distance = voxelCoords - sourceCoords_3d
    distance = distance[:, :, 0] ** 2 + distance[:, :, 1] ** 2
    factor = (SAD / voxelSize) ** 2 / distance
    TermaFullArray *= factor

    sourceAxis = np.array((np.cos(beamAngle), np.sin(beamAngle)))
    sourceAxis = np.expand_dims(sourceAxis, axis=(0, 1))
    lateralAxis = np.array((- np.sin(beamAngle), np.cos(beamAngle)))
    lateralAxis = np.expand_dims(lateralAxis, axis=(0, 1))

    
    isoCoordsExpand = np.expand_dims(isoCoords, axis=(0, 1))
    voxelCoords -= isoCoordsExpand

    sourceAxisProjection = np.sum(voxelCoords * sourceAxis, axis=2)
    distanceToSource = SAD / voxelSize - sourceAxisProjection

    lateralAxisProjection = np.sum(voxelCoords * lateralAxis, axis=2)
    lateralAxisNormalized = lateralAxisProjection / distanceToSource * SAD / voxelSize
    termaContextList = []
    for i in range(fluenceDim):
        offsetStart = (- fluenceDim / 2  + i) * fluenceStride + fluenceGap
        offsetStart /= voxelSize
        offsetEnd = (- fluenceDim / 2 + (i + 1)) * fluenceStride - fluenceGap
        offsetEnd /= voxelSize
        voxelsOn = np.logical_and(lateralAxisNormalized >= offsetStart,
            lateralAxisNormalized <= offsetEnd)
        localTermaArray = np.zeros((height, width), float)
        localTermaArray[voxelsOn] = TermaFullArray[voxelsOn]
        termaContextList.append(localTermaArray)
    

    voxelCoordsShape = (height, width, 2)
    voxelCoords = np.zeros(voxelCoordsShape, float)
    voxelCoordsX = np.arange(width) + 0.5
    voxelCoordsX = np.expand_dims(voxelCoordsX, axis=0)
    voxelCoordsY = np.arange(height) + 0.5
    voxelCoordsY = np.expand_dims(voxelCoordsY, axis=1)
    voxelCoords[:, :, 0] = voxelCoordsX
    voxelCoords[:, :, 1] = voxelCoordsY
    longitudinalRange = None
    lateralRange = None
    sourceCoords = isoCoords + sourceAxis * SAD / voxelSize
    paramsList = []
    for i in range(fluenceDim):
        offsetStart = (- fluenceDim / 2  + i) * fluenceStride + fluenceGap
        offsetStart /= voxelSize
        offsetEnd = (- fluenceDim / 2 + (i + 1)) * fluenceStride - fluenceGap
        offsetEnd /= voxelSize
        voxelsOn = np.logical_and(lateralAxisNormalized >= offsetStart,
            lateralAxisNormalized <= offsetEnd)
        voxelsOn = np.logical_and(voxelsOn, bodyMaskSlice)

        beamletIsocenter = isoCoords + (- fluenceDim / 2 + i + 0.5) \
            * fluenceStride / voxelSize * lateralAxis.squeeze()
        beamletAxis = sourceCoords - beamletIsocenter
        beamletAxis /= np.linalg.norm(beamletAxis)
        beamletAxis = beamletAxis.squeeze()
        beamletLateral = np.array((-beamletAxis[1], beamletAxis[0]))
        beamletAxis = np.expand_dims(beamletAxis, axis=(0, 1))
        beamletLateral = np.expand_dims(beamletLateral, axis=(0, 1))

        voxelCoords2BeamletIsocenter = voxelCoords - beamletIsocenter
        beamletAxisProjection = np.sum(voxelCoords2BeamletIsocenter * beamletAxis, axis=2)
        beamletAxisProjection = beamletAxisProjection[voxelsOn]
        beamletLateralProjection = np.sum(voxelCoords2BeamletIsocenter * beamletLateral, axis=2)
        beamletLateralProjection = beamletLateralProjection[voxelsOn]

        minAxisProj, maxAxisProj = np.min(beamletAxisProjection), np.max(beamletAxisProjection)
        minLateralProj, maxLateralProj = np.min(beamletLateralProjection), np.max(beamletLateralProjection)

        localLongitudinalRange = maxAxisProj - minAxisProj + 2 * convolutionMargin
        if longitudinalRange is None:
            longitudinalRange = localLongitudinalRange
        else:
            longitudinalRange = max(longitudinalRange, localLongitudinalRange)
            
        localLateralRange = maxLateralProj - minLateralProj + 2 * convolutionMargin
        if lateralRange is None:
            lateralRange = localLateralRange
        else:
            lateralRange = max(lateralRange, localLateralRange)

        updatedIsocenter = beamletIsocenter + (minAxisProj + maxAxisProj) / 2 * beamletAxis.squeeze()
        paramsLocal = [updatedIsocenter, beamletAxis]
        paramsList.append(paramsLocal)

    densityInterpolator = RegularGridInterpolator((np.arange(height), np.arange(width)), densitySlice)

    densityContextList = []
    termaContextResults = []
    contextDimension = (int(lateralRange), int(longitudinalRange))
    contextCoordsX = (np.arange(contextDimension[0]) + 0.5) / contextDimension[0]
    contextCoordsX = np.expand_dims(contextCoordsX, axis=(1, 2))
    contextCoordsX = np.repeat(contextCoordsX, contextDimension[1], axis=1)

    contextCoordsY = (np.arange(contextDimension[1]) + 0.5) / contextDimension[1]
    contextCoordsY = np.expand_dims(contextCoordsY, axis=(0, 2))
    contextCoordsY = np.repeat(contextCoordsY, contextDimension[0], axis=0)
    for i in range(fluenceDim):
        # calculate the four corner points of the rectangle
        centerPoint, beamletAxis = paramsList[i]
        beamletAxis = beamletAxis.squeeze()
        # print(beamletAxis)
        lateralAxis = np.array((-beamletAxis[1], beamletAxis[0]))
        
        centerPoint__ = np.expand_dims(centerPoint, axis=(0, 1))
        beamletAxis__ = np.expand_dims(beamletAxis, axis=(0, 1))
        lateralAxis__ = np.expand_dims(lateralAxis, axis=(0, 1))
        CartesianCoords = centerPoint__ + (contextCoordsX - 0.5) * lateralAxis__ * contextDimension[0] \
            - (contextCoordsY - 0.5) * beamletAxis__ * contextDimension[1]
        
        CartesianCoords = np.reshape(CartesianCoords, (contextDimension[0] * contextDimension[1], 2))
        CartesianCoords = np.flip(CartesianCoords, axis=1)
        newDensity = densityInterpolator(CartesianCoords)
        newDensity = np.reshape(newDensity, contextDimension)
        newDensity = np.transpose(newDensity, axes=(1, 0))
        densityContextList.append(newDensity)

        localTermaArray = termaContextList[i]
        localTermaInterpolator = RegularGridInterpolator(
            (np.arange(height), np.arange(width)), localTermaArray)
        newTerma = localTermaInterpolator(CartesianCoords)
        newTerma = np.reshape(newTerma, contextDimension)
        newTerma = np.transpose(newTerma, axes=(1, 0))
        termaContextResults.append(newTerma)
    
    densityCanvas = np.concatenate(densityContextList, axis=1)
    termaCanvas = np.concatenate(termaContextResults, axis=1)

    debugView = True
    if debugView:
        fig, ax = plt.subplots(figsize=(densityCanvas.shape[1]/100, densityCanvas.shape[0]/100), dpi=100)
        ax.imshow(densityCanvas, cmap="gray", vmin=0.5, vmax=1.5)
        ax.imshow(termaCanvas, cmap="jet", vmin=0, vmax=np.max(TermaFullArray), alpha=0.3*(termaCanvas>1e-5))
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        figureFile = os.path.join(topicFolder, "densityTerma.png")
        plt.savefig(figureFile)
        plt.close(fig)
        plt.clf()
    else:
        densityCanvasFile = os.path.join(topicFolder, "densityCanvas.npy")
        np.save(densityCanvasFile, densityCanvas)
        termaCanvasFile = os.path.join(topicFolder, "termaCanvas.npy")
        np.save(termaCanvasFile, termaCanvas)


def REVPreConvolution():
    """
    This function draws the dose convolution step in the REV coordinate system.
    In this function, we define the X anx Y axes to be along the width and height directions,
    respectively
    """
    Array_A = np.array((2.1791e+00, 1.7472e+00, 9.7930e-01, 5.1485e-01))
    Array_a = np.array((2.1791e+00, 1.9211e+00, 2.5750e+00, 3.9910e+00))
    Array_B = np.array((2.4508e-02, 2.2600e-02, 1.3471e-02, 7.8529e-03))
    Array_b = np.array((5.8029e-02, 5.7971e-02, 9.5684e-02, 1.1161e-01))

    densityCanvas = os.path.join(topicFolder, "densityCanvas.npy")
    densityCanvas = np.load(densityCanvas)
    termaCanvas = os.path.join(topicFolder, "termaCanvas.npy")
    termaCanvas = np.load(termaCanvas)
    assert densityCanvas.shape == termaCanvas.shape
    height, width = densityCanvas.shape
    densityCanvas = RegularGridInterpolator((np.arange(width), np.arange(height)),
        np.transpose(densityCanvas, axes=(1, 0)), bounds_error=False, fill_value=0)
    termaCanvas = RegularGridInterpolator((np.arange(width), np.arange(height)),
        np.transpose(termaCanvas, axes=(1, 0)), bounds_error=False, fill_value=0)

    nAngles = 4
    canvasHeight = 0
    canvasWidth = 0
    for i in range(nAngles):
        angle = np.pi / 2 * (i + 0.5) / nAngles
        resultCoordinates = []
        for j in range(2):
            for k in range(2):
                xValue = j * width
                yValue = k * height
                coords_new = np.array((xValue * np.cos(angle) - yValue * np.sin(angle),
                    xValue * np.sin(angle) + yValue * np.cos(angle)))
                resultCoordinates.append(coords_new)
        xComponents = [a[0] for a in resultCoordinates]
        xRange = max(*xComponents) - min(*xComponents)
        yComponents = [a[1] for a in resultCoordinates]
        yRange = max(*yComponents) - min(*yComponents)
        canvasWidth = max(canvasWidth, xRange)
        canvasHeight = max(canvasHeight, yRange)

    canvasHeight = int(np.ceil(canvasHeight))
    canvasWidth = int(np.ceil(canvasWidth))
    fullCanvasShape = (canvasHeight, canvasWidth)
    
    # prepare the background board
    backgroundBoardShape = fullCanvasShape + (1,)
    backgroundBoard = np.ones(backgroundBoardShape, dtype=np.uint8)
    backgroundColor = hex_to_rgb(colors[2])
    backgroundColor = lighten_color(backgroundColor, 0.3)
    backgroundColor = np.expand_dims(backgroundColor, axis=(0, 1))
    backgroundBoard = backgroundBoard * backgroundColor
    backgroundBoard = backgroundBoard.astype(np.uint8)

    for i in range(nAngles):
        angle = np.pi / 2 * (i + 0.5) / nAngles
        # calculate again the four coordinates of the original canvas
        resultCoordinates = []
        for j in range(2):
            for k in range(2):
                xValue = j * width
                yValue = k * height
                coords_new = np.array((xValue * np.cos(angle) - yValue * np.sin(angle),
                    xValue * np.sin(angle) + yValue * np.cos(angle)))
                resultCoordinates.append(coords_new)
        # calculate the minimum values of the coords
        xComponents = [a[0] for a in resultCoordinates]
        xBase = min(*xComponents)
        yComponents = [a[1] for a in resultCoordinates]
        yBase = min(*yComponents)

        # then calculate the coordinates of the fullCanvas
        fullCanvasCoordsX = np.arange(canvasWidth) + xBase
        fullCanvasCoordsX = np.expand_dims(fullCanvasCoordsX, axis=(0, 2))
        fullCanvasCoordsX = np.repeat(fullCanvasCoordsX, canvasHeight, axis=0)
        fullCanvasCoordsY = np.arange(canvasHeight) + yBase
        fullCanvasCoordsY = np.expand_dims(fullCanvasCoordsY, axis=(1, 2))
        fullCanvasCoordsY = np.repeat(fullCanvasCoordsY, canvasWidth, axis=1)
        fullCanvasCoords = np.concatenate((fullCanvasCoordsX, fullCanvasCoordsY), axis=2)
        
        orgAxisX = np.array((np.cos(angle), np.sin(angle)))
        orgAxisX = np.expand_dims(orgAxisX, axis=(0, 1))
        orgAxisY = np.array((-np.sin(angle), np.cos(angle)))
        orgAxisY = np.expand_dims(orgAxisY, axis=(0, 1))
        fullCanvasOrgCoordsX = np.sum(fullCanvasCoords * orgAxisX, axis=2, keepdims=True)
        fullCanvasOrgCoordsY = np.sum(fullCanvasCoords * orgAxisY, axis=2, keepdims=True)
        fullCanvasOrgCoords = np.concatenate((fullCanvasOrgCoordsX, fullCanvasOrgCoordsY), axis=2)
        validArea = np.logical_and(np.logical_and(fullCanvasOrgCoordsX >= 0, fullCanvasOrgCoordsX <= width),
            np.logical_and(fullCanvasOrgCoordsY >= 0, fullCanvasOrgCoordsY <= height))
        validArea = validArea.squeeze()
        validArea = validArea.astype(float)
        
        fullCanvasOrgCoords = np.reshape(fullCanvasOrgCoords, (fullCanvasShape[0] * fullCanvasShape[1], 2))
        fullCanvasDensity = densityCanvas(fullCanvasOrgCoords)
        fullCanvasDensity = np.reshape(fullCanvasDensity, fullCanvasShape)
        fullCanvasTerma = termaCanvas(fullCanvasOrgCoords)
        fullCanvasTerma = np.reshape(fullCanvasTerma, fullCanvasShape)
        
        fig, ax = plt.subplots(figsize=(fullCanvasShape[0]/100, fullCanvasShape[1]/100), dpi=100)
        ax.imshow(backgroundBoard)
        ax.imshow(fullCanvasDensity, cmap="gray", vmin=0.5, vmax=1.5, alpha=validArea)
        ax.imshow(fullCanvasTerma, cmap="jet", vmin=0, vmax=np.max(fullCanvasTerma), alpha=0.3*(fullCanvasTerma>0))
        
        # add an arrow indicating the convolution direction
        ArrowStartCoords = (0.9 * fullCanvasShape[1], 0.2 * fullCanvasShape[0])
        ArrowEndCoords = (0.9 * fullCanvasShape[1], 0.8 * fullCanvasShape[0])
        arrow = patches.FancyArrowPatch(
            ArrowStartCoords, ArrowEndCoords, arrowstyle="<->", mutation_scale=30,
            color=colors[0], linewidth=2, linestyle="--")
        ax.add_patch(arrow)
        ax.text(fullCanvasShape[1] * 0.95, fullCanvasShape[0] * 0.5,
            "Convolution", rotation=90, fontsize=15, ha="center", va="center",
            color=colors[0])

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        figureFile = os.path.join(topicFolder, "convolveREV{}.png".format(i))
        plt.savefig(figureFile)
        plt.close(fig)
        plt.clf()


def lighten_color(color, factor):
    assert factor >= 0 and factor <= 1
    color = color.astype(float)
    color = color + (255 - color) * factor
    color = color.astype(np.uint8)
    return color


def REVPostConvolution():
    # obtain the kernel file
    kernelFile = "/data/qifan/FastDose/scripts/kernel_exp_6mv.txt"
    Array_A = []
    Array_a = []
    Array_B = []
    Array_b = []
    with open(kernelFile, "r") as f:
        lines = f.readlines()
    for i in range(1, 5):
        line = lines[i]
        entries = line.split("  ")
        entries = [float(a) for a in entries]
        Array_A.append(entries[2])
        Array_a.append(entries[3])
        Array_B.append(entries[4])
        Array_b.append(entries[5])
    Array_A = np.array(Array_A)
    Array_a = np.array(Array_a)
    Array_B = np.array(Array_B)
    Array_b = np.array(Array_b)
    
    densityCanvas = os.path.join(topicFolder, "densityCanvas.npy")
    densityCanvas = np.load(densityCanvas)
    termaCanvas = os.path.join(topicFolder, "termaCanvas.npy")
    termaCanvas = np.load(termaCanvas)

    # process terma
    voxelSize = 0.108  # cm
    mu_new = 0.00707
    height, width = termaCanvas.shape
    termaCanvas = np.power(termaCanvas, mu_new / mu)
    sourceAxis = np.array((np.cos(beamAngle), np.sin(beamAngle)))
    sourceCoords_1d = isoCoords + sourceAxis * SAD / voxelSize
    sourceCoords_3d = np.expand_dims(sourceCoords_1d, axis=(0, 1))
    voxelCoordsShape = (height, width, 2)
    voxelCoords = np.zeros(voxelCoordsShape, float)
    voxelCoordsX = np.arange(width) + 0.5
    voxelCoordsX = np.expand_dims(voxelCoordsX, axis=0)
    voxelCoordsY = np.arange(height) + 0.5
    voxelCoordsY = np.expand_dims(voxelCoordsY, axis=1)
    voxelCoords[:, :, 0] = voxelCoordsX
    voxelCoords[:, :, 1] = voxelCoordsY
    distance = voxelCoords - sourceCoords_3d
    distance = distance[:, :, 0] ** 2 + distance[:, :, 1] ** 2
    factor = (SAD / voxelSize) ** 2 / distance
    termaCanvas *= factor

    # segment the canvas into individual contexts
    assert densityCanvas.shape == termaCanvas.shape
    height, width = densityCanvas.shape
    assert width % fluenceDim == 0
    contextWidth = width // fluenceDim
    contextHeight = height
    densityContextList = []
    termaContextList = []
    for i in range(fluenceDim):
        localDensityContext = densityCanvas[:, i*contextWidth: (i+1)*contextWidth].copy()
        densityContextList.append(localDensityContext)
        localTermaContext = termaCanvas[:, i*contextWidth: (i+1)*contextWidth].copy()
        termaContextList.append(localTermaContext)
    
    def transform_from_context_to_rev(input, angle, offset):
        # here we define the axis order to be (width, hight)
        # input is in dimension (nPoints, 2)
        # offset is defined as the origin of the bev coordinate system in the context system
        axis_x = np.array((np.cos(angle), np.sin(angle)))
        axis_x = np.expand_dims(axis_x, axis=0)
        axis_y = np.array((-np.sin(angle), np.cos(angle)))
        axis_y = np.expand_dims(axis_y, axis=0)
        offset = np.expand_dims(offset, axis=0)
        coord_x_org = input[:, 0]
        coord_x_org = np.expand_dims(coord_x_org, axis=1)
        coord_y_org = input[:, 1]
        coord_y_org = np.expand_dims(coord_y_org, axis=1)
        coord_new = coord_x_org * axis_x + coord_y_org * axis_y
        coord_new = coord_new - offset
        return coord_new
    
    def transform_from_rev_to_context(input, angle, offset):
        offset = np.expand_dims(offset, axis=0)
        input = input + offset
        axis_x = np.array((np.cos(angle), np.sin(angle)))
        axis_x = np.expand_dims(axis_x, axis=0)
        axis_y = np.array((-np.sin(angle), np.cos(angle)))
        axis_y = np.expand_dims(axis_y, axis=0)
        coord_x = np.sum(input * axis_x, axis=1, keepdims=True)
        coord_y = np.sum(input * axis_y, axis=1, keepdims=True)
        output = np.concatenate((coord_x, coord_y), axis=1)
        return output

    densityCanvas = os.path.join(topicFolder, "densityCanvas.npy")
    densityCanvas = np.load(densityCanvas)
    
    nAngles = 4
    angleContributionList = []
    for i in range(-nAngles, nAngles):
        angle = np.pi / 2 * (i + 0.5) / nAngles
        # calculate the coordinates of the four vertices of the bev context in REV
        REV_min_x = None
        REV_max_x = None
        REV_min_y = None
        REV_max_y = None
        axis_x = np.array((np.cos(angle), np.sin(angle)))
        axis_y = np.array((-np.sin(angle), np.cos(angle)))
        for j in range(2):
            for k in range(2):
                coords_context = np.array((j * contextWidth, k * contextHeight))
                coords_new = coords_context[0] * axis_x + coords_context[1] * axis_y

                if REV_min_x is None:
                    REV_min_x = coords_new[0]
                else:
                    REV_min_x = min(REV_min_x, coords_new[0])

                if REV_max_x is None:
                    REV_max_x = coords_new[0]
                else:
                    REV_max_x = max(REV_max_x, coords_new[0])
                
                if REV_min_y is None:
                    REV_min_y = coords_new[1]
                else:
                    REV_min_y = min(REV_min_y, coords_new[1])
                
                if REV_max_y is None:
                    REV_max_y = coords_new[1]
                else:
                    REV_max_y = max(REV_max_y, coords_new[1])
        REVCanvasSize = np.array((int(np.ceil(REV_max_x - REV_min_x)), int(np.ceil(REV_max_y - REV_min_y))))
        REVCoords_x = np.arange(REVCanvasSize[0])
        REVCoords_x = np.expand_dims(REVCoords_x, axis=(0, 2))
        REVCoords_x = np.repeat(REVCoords_x, REVCanvasSize[1], axis=0)
        REVCoords_y = np.arange(REVCanvasSize[1])
        REVCoords_y = np.expand_dims(REVCoords_y, axis=(1, 2))
        REVCoords_y = np.repeat(REVCoords_y, REVCanvasSize[0], axis=1)
        REVCoords = np.concatenate((REVCoords_x, REVCoords_y), axis=2)
        REVCoords = np.reshape(REVCoords, (REVCoords.shape[0] * REVCoords.shape[1], 2))
        offset = np.array((REV_min_x, REV_min_y))
        REVCoordsInContext = transform_from_rev_to_context(REVCoords, angle, offset)


        ContextCoords_X = np.arange(contextWidth)
        ContextCoords_X = np.expand_dims(ContextCoords_X, axis=(0, 2))
        ContextCoords_X = np.repeat(ContextCoords_X, contextHeight, axis=0)
        ContextCoords_Y = np.arange(contextHeight)
        ContextCoords_Y = np.expand_dims(ContextCoords_Y, axis=(1, 2))
        ContextCoords_Y = np.repeat(ContextCoords_Y, contextWidth, axis=1)
        ContextCoords = np.concatenate((ContextCoords_X, ContextCoords_Y), axis=2)
        ContextCoords = np.reshape(ContextCoords, (contextHeight * contextWidth, 2))
        ContextCoordsInREV = transform_from_context_to_rev(ContextCoords, angle, offset)
        
        # interpolate from the context to the REV canvas
        angleDoseREVList = []
        for j in range(fluenceDim):
            contextDensity = RegularGridInterpolator(
                (np.arange(contextWidth), np.arange(contextHeight)),
                np.transpose(densityContextList[j], axes=(1, 0)),
                bounds_error=False, fill_value=0.0)
            REVDensity = contextDensity(REVCoordsInContext)
            REVDensity = np.reshape(REVDensity, (REVCanvasSize[1], REVCanvasSize[0]))
            
            contextTerma = RegularGridInterpolator(
                (np.arange(contextWidth), np.arange(contextHeight)),
                np.transpose(termaContextList[j], axes=(1, 0)),
                bounds_error=False, fill_value=0.0)
            REVTerma = contextTerma(REVCoordsInContext)
            REVTerma = np.reshape(REVTerma, (REVCanvasSize[1], REVCanvasSize[0]))
            
            # calculate dose in the context
            REVDose = np.zeros_like(REVTerma)
            for k in range(Array_A.size):
                A_value = Array_A[k]
                a_value = Array_a[k]
                X_Array = np.zeros(REVDose.shape[1], float)
                for l in range(REVDose.shape[0]):
                    ap_row = voxelSize * a_value * REVDensity[l, :]
                    exp_minus_ap = np.exp(-ap_row)
                    g_row = 1 - 1/2 * exp_minus_ap + 1/6 * exp_minus_ap ** 2 - 1/24 * exp_minus_ap ** 3
                    REVDose[l, :] += (A_value / a_value) * ((1-g_row) * REVTerma[l, :] + g_row * X_Array)
                    X_Array = exp_minus_ap * X_Array + (1 - exp_minus_ap) * REVTerma[l, :]
            
                B_value = Array_B[k]
                b_value = Array_b[k]
                X_Array = np.zeros(REVDose.shape[1], float)
                for l in range(REVDose.shape[0]):
                    bp_row = voxelSize * b_value * REVDensity[l, :]
                    exp_minus_bp = np.exp(-bp_row)
                    g_row = 1 - 1/2 * exp_minus_bp + 1/6 * exp_minus_bp ** 2 - 1/24 * exp_minus_bp ** 3
                    REVDose[l, :] += (B_value / b_value) * ((1-g_row) * REVTerma[l, :] + g_row * X_Array)
                    X_Array = exp_minus_bp * X_Array + (1 - exp_minus_bp) * REVTerma[l, :]
            
            # interpolate the dose array back to the context frame
            REVDoseInterpolator = RegularGridInterpolator(
                (np.arange(REVDose.shape[1]), np.arange(REVDose.shape[0])),
                np.transpose(REVDose, axes=(1, 0)),
                bounds_error=False, fill_value=0.0)
            ContextDose = REVDoseInterpolator(ContextCoordsInREV)
            ContextDose = np.reshape(ContextDose, (contextHeight, contextWidth))
            angleDoseREVList.append(ContextDose)
        angleDoseREV = np.concatenate(angleDoseREVList, axis=1)
        angleContributionList.append(angleDoseREV)
    
    totalDoseContribution = None
    for i in range(len(angleDoseREVList)):
        if totalDoseContribution is None:
            totalDoseContribution = angleContributionList[i]
        else:
            totalDoseContribution += angleContributionList[i]
    
    for i in range(4, 8):
        angleDose = angleContributionList[i]
        fig, ax = plt.subplots(figsize=(densityCanvas.shape[1]/100,
            densityCanvas.shape[0]/100), dpi=100)
        ax.imshow(densityCanvas, cmap="gray", vmin=0.5, vmax=1.5)
        ax.imshow(angleDose, cmap="jet", vmin=0.1*np.max(angleDose), vmax=np.max(angleDose), alpha=0.3)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        figureFile = os.path.join(topicFolder, "doseConvolveDirection{}.png".format(i-4))
        plt.savefig(figureFile)
        plt.close(fig)
        plt.clf()
    
    figureFile = os.path.join(topicFolder, "contextDoseTotal.png")
    fig, ax = plt.subplots(figsize=(densityCanvas.shape[1]/100,
        densityCanvas.shape[0]/100), dpi=100)
    ax.imshow(densityCanvas, cmap="gray", vmin=0.5, vmax=1.5)
    ax.imshow(totalDoseContribution, cmap="jet", vmin=0.1*np.max(totalDoseContribution),
        vmax=np.max(totalDoseContribution), alpha=0.3)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figureFile = os.path.join(topicFolder, "doseConvolveFull.png")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()



def putFiguresTogether():
    names = ["CartesianRay", "contextBundle", "densityTerma", "doseConvolveFull"] + \
        ["convolveREV{}".format(i) for i in range(4)] + \
        ["doseConvolveDirection{}".format(i) for i in range(4)]

    alphabet_list = list(string.ascii_lowercase)
    labels = ["({})".format(alphabet_list[i]) for i in range(len(names))]

    figures = []
    for name in names:
        file = os.path.join(topicFolder, "{}.png".format(name))
        figure = plt.imread(file)
        figures.append(figure)
    
    # crop the CartesianRay figure into uniform sizes
    shape0 = figures[0].shape
    shape1 = figures[1].shape
    height_expected = shape0[1] * (shape1[0] / shape1[1])
    height_expected = int(height_expected)
    cropDim = int((shape0[0] - height_expected) / 2)
    figure0 = figures[0].copy()
    figure0 = figure0[cropDim: cropDim+height_expected, :]
    figures[0] = figure0
    
    fig, axs = plt.subplots(3, 4, figsize=(16, 12), dpi=500)
    for i in range(3):
        for j in range(4):
            idx = i * 4 + j
            label = labels[idx]
            figure = figures[idx]
            ax = axs[i, j]
            ax.imshow(figure)
            ax.text(0.5, -0.05, label, ha="center", va="center",
                transform=ax.transAxes, fontsize=20)
            ax.axis("off")
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0.01, wspace=0.1, hspace=0.1)
    figureFile = os.path.join(topicFolder, "CCCSNephDiagram.png")
    plt.savefig(figureFile, dpi=500)
    plt.close(fig)
    plt.clf()


if __name__ == "__main__":
    densitySliceInit()
    # TermaCartesian()
    # contextBundle()
    # contextTerma()
    # overlayDensityTerma()
    # drawTermaInTheContext()
    # REVPreConvolution()
    # REVPostConvolution()
    putFiguresTogether()