import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import scipy.interpolate as interp
import scipy.integrate as integrate
from skimage import transform
import nrrd
import random

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
figuresFolder = "/data/qifan/AAPM2024/manufigures"
topicFolder = os.path.join(figuresFolder, "CCCSAhnesjo")
if not os.path.isdir(topicFolder):
    os.mkdir(topicFolder)

densitySlice = None

def densitySliceInit():
    global densitySlice
    CTFile = "/data/qifan/FastDoseWorkplace/TCIAAdd/002/CT.nrrd"
    CTArray, CTHeader = nrrd.read(CTFile)
    CTArray = np.transpose(CTArray, axes=(2, 1, 0))
    CTArray += 1024
    sliceIdx = 10
    densitySlice = CTArray[sliceIdx, :, :]
    densitySlice = densitySlice[230: 330, 50: 150]
    if True:
        densitySliceFile = os.path.join(topicFolder, "densitySlice.png")
        plt.imsave(densitySliceFile, densitySlice, cmap="gray", vmin=500, vmax=1500)


def drawFullConv1():
    """
    This function draws the dose deposition of the conventional full convolution,
    i.e., from the TERMA voxel to all voxels of interest
    """
    color = colors[1]
    canvas = os.path.join(topicFolder, "densitySlice.png")
    canvas = plt.imread(canvas)
    canvasResultPath = os.path.join(topicFolder, "fullConv(a).png")
    marginSize = 2
    contentSize = 16
    starting = 5
    stride = marginSize + contentSize
    for offset in range(starting, canvas.shape[0], stride):
        canvas[offset: offset+marginSize, :, :] = 1
        canvas[:, offset: offset+marginSize, :] = 1
    
    # calculate the endpoints
    coordsSet = []
    centerPoint = None
    middleValue = int((marginSize + contentSize) / 2)
    numCells = 5
    centerCell = 2
    for idx1 in range(numCells + 2):
        offset1 = starting + (idx1 - 1) * stride
        for idx2 in range(numCells + 2):
            offset2 = starting + (idx2 - 1) * stride
            point = np.array((offset1 + middleValue, offset2 + middleValue))
            if idx1 == 3 and idx2 == 3:
                centerPoint = point
            else:
                coordsSet.append(point)
    assert centerPoint is not None

    if False:
        # get a subset of the endpoints
        random.seed(1526)
        random.shuffle(coordsSet)
        nSelects = 12
        coordsSet = coordsSet[: nSelects]
    
    fig, ax = plt.subplots(figsize=(canvas.shape[0]/20, canvas.shape[1]/20), dpi=100)
    ax.imshow(canvas, cmap="gray", vmin=500, vmax=1500)

    doseLayer = np.zeros(canvas.shape[:2], dtype=float)
    doseLayer[starting + centerCell * stride + marginSize: starting + (centerCell + 1) * stride,
        starting + centerCell * stride + marginSize: starting + (centerCell + 1) * stride] = 1.0
    ax.imshow(doseLayer, cmap="jet", vmin=0, vmax=1, alpha=0.3 * doseLayer)

    # draw the arrows
    for endPoint in coordsSet:
        arrow_style = patches.FancyArrowPatch(centerPoint, endPoint,
            connectionstyle="arc3, rad=.2", arrowstyle="->", mutation_scale=40,
            color=color, linewidth=2, linestyle="dashed")
        ax.add_patch(arrow_style)

    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(canvasResultPath)
    plt.close(fig)

    if False:
        plt.imsave(canvasResultPath, canvas)


def drawFullConv2():
    """
    This function draws the dose deposition of the conventional full convolution,
    i.e., from the TERMA voxel to all voxels of interest
    """
    pixelSize = 0.108  # cm
    mu = 0.2
    canvas = os.path.join(topicFolder, "densitySlice.png")
    canvas = plt.imread(canvas)
    canvasResultPath = os.path.join(topicFolder, "fullConv(b).png")
    marginSize = 2
    contentSize = 16
    starting = 5
    stride = marginSize + contentSize
    for offset in range(starting, canvas.shape[0], stride):
        canvas[offset: offset+marginSize, :, :] = 1
        canvas[:, offset: offset+marginSize, :] = 1

    # get the original CT density slice
    CTFile = "/data/qifan/FastDoseWorkplace/TCIAAdd/002/CT.nrrd"
    CTArray, CTHeader = nrrd.read(CTFile)
    CTArray = np.transpose(CTArray, axes=(2, 1, 0))
    CTArray += 1024
    sliceIdx = 10
    densitySlice = CTArray[sliceIdx, :, :].copy()
    densitySlice = densitySlice[230: 330, 50: 150]
    densitySlice = densitySlice / 1024
    height, width = densitySlice.shape
    heightAxis = np.arange(height)
    widthAxis = np.arange(width)
    densitySlice = interp.RectBivariateSpline(heightAxis, widthAxis, densitySlice)

    centerCoord = np.array((50, 50))
    doseMap = np.zeros((height, width), dtype=float)
    nPoints = 100
    for i in range(height):
        path_x = np.linspace(centerCoord[0], i, nPoints)
        for j in range(width):
            path_y = np.linspace(centerCoord[1], j, nPoints)
            path_z = densitySlice(path_x, path_y, grid=False)
            dx = np.gradient(path_x)
            dy = np.gradient(path_y)
            ds = np.sqrt(dx**2 + dy**2)
            line_integral = integrate.simpson(path_z * ds, dx=1)
            doseMap[i, j] = line_integral
    doseMap = np.exp(- pixelSize * mu * doseMap)  # 100 x 100

    marginSize = 2
    contentSize = 16
    starting = 5
    numCells = 5
    stride = marginSize + contentSize
    doseMapSeg = np.zeros_like(doseMap)
    for i in range(numCells + 2):
        beginX = starting + (i - 1) * stride + marginSize
        beginX = max(0, beginX)
        endX = starting + i * stride
        endX = min(height, endX)
        for j in range(numCells + 2):
            beginY = starting + (j - 1) * stride + marginSize
            beginY = max(0, beginY)
            endY = starting + j * stride
            endY = min(width, endY)
            doseMapCrop = doseMap[beginX: endX, beginY: endY]
            doseAvg = np.sum(doseMapCrop) / doseMapCrop.size
            doseMapSeg[beginX: endX, beginY: endY] = doseAvg
    
    fig, ax = plt.subplots(figsize=(canvas.shape[0]/20, canvas.shape[1]/20), dpi=100)
    ax.imshow(canvas)
    ax.imshow(doseMapSeg, cmap="jet", vmin=0, vmax=np.max(doseMapSeg), alpha=0.3*(doseMapSeg>1e-4))
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(canvasResultPath)
    plt.close(fig)
    plt.clf()


def drawCCCS1():
    """
    This function draws the line integral part of the CCCS
    """
    pixelSize = 0.108  # cm
    mu = 0.2
    eps = 1e-4
    canvas = os.path.join(topicFolder, "densitySlice.png")
    canvas = plt.imread(canvas)
    marginSize = 2
    contentSize = 16
    starting = 5
    stride = marginSize + contentSize
    for offset in range(starting, canvas.shape[0], stride):
        canvas[offset: offset+marginSize, :, :] = 1
        canvas[:, offset: offset+marginSize, :] = 1

    drawCCCS1ResultFile = os.path.join(topicFolder, "collapsedCone(a).png")
    nCones = 12
    anglePerCone = 2 * np.pi / nCones

    fig, ax = plt.subplots(figsize=(canvas.shape[0]/20, canvas.shape[1]/20), dpi=100)
    ax.imshow(canvas)

    # firstly, generate cone segments
    halfSize = canvas.shape[0] / 2
    green = colors[1]
    for i in range(nCones):
        rayAngle = (i + 0.5) * anglePerCone
        rayDirection = np.array((-np.sin(rayAngle), np.cos(rayAngle)))
        scale1 = 1e+8  # initialize to infinite
        scale2 = 1e+8
        if np.abs(rayDirection[0]) > eps:
            scale1 = abs(halfSize / rayDirection[0])
        if np.abs(rayDirection[1]) > eps:
            scale2 = abs(halfSize / rayDirection[1])
        scale = scale1 if scale1 < scale2 else scale2
        target = np.clip(rayDirection * scale, - halfSize + eps, halfSize - 1 - eps)
        beginPoint = np.array((50.5, 50.5))
        endPoint = beginPoint + target
        ax.plot((beginPoint[0], endPoint[0]), (beginPoint[1], endPoint[1]), color=green, linestyle="--")
    
    CTFile = "/data/qifan/FastDoseWorkplace/TCIAAdd/002/CT.nrrd"
    CTArray, CTHeader = nrrd.read(CTFile)
    CTArray = np.transpose(CTArray, axes=(2, 1, 0))
    CTArray += 1024
    sliceIdx = 10
    densitySlice = CTArray[sliceIdx, :, :].copy()
    densitySlice = densitySlice[230: 330, 50: 150]
    densitySlice = densitySlice / 1024
    height, width = densitySlice.shape
    heightAxis = np.arange(height)
    widthAxis = np.arange(width)
    densitySlice = interp.RectBivariateSpline(heightAxis, widthAxis, densitySlice)

    # then draw the convolution lines
    nSamples = 100
    cmap = plt.get_cmap("jet")
    norm = plt.Normalize(vmin=0, vmax=1)
    for i in range(nCones):
        rayAngle = i * anglePerCone
        rayDirection = np.array((-np.cos(rayAngle), np.sin(rayAngle)))
        scale1 = 1e+8  # initialize to infinite
        scale2 = 1e+8
        if np.abs(rayDirection[0]) > eps:
            scale1 = abs(halfSize / rayDirection[0])
        if np.abs(rayDirection[1]) > eps:
            scale2 = abs(halfSize / rayDirection[1])
        scale = scale1 if scale1 < scale2 else scale2
        target = np.clip(rayDirection * scale, - halfSize + eps, halfSize - 1 - eps)
        beginPoint = np.array((50.5, 50.5))
        endPoint = beginPoint + target
        path_x = np.linspace(beginPoint[0], endPoint[0], nSamples)
        path_y = np.linspace(beginPoint[1], endPoint[1], nSamples)
        path_z = densitySlice(path_x, path_y, grid=False)
        radPathLength = np.cumsum(path_z) * pixelSize * mu
        path_dose = np.exp(-radPathLength)
        path_dose = cmap(norm(path_dose))
        for j in range(nSamples-1):
            ax.plot(path_x[j:j+2], path_y[j:j+2], color=path_dose[j, :], linewidth=4)

    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(drawCCCS1ResultFile)
    plt.close(fig)
    plt.clf()


def drawCCCS2():
    CTFile = "/data/qifan/FastDoseWorkplace/TCIAAdd/002/CT.nrrd"
    CTArray, CTHeader = nrrd.read(CTFile)
    CTArray = np.transpose(CTArray, axes=(2, 1, 0))
    CTArray += 1024
    sliceIdx = 10
    densitySlice = CTArray[sliceIdx, :, :].copy()
    densitySlice = densitySlice[230: 330, 50: 150]
    densitySlice = densitySlice / 1024
    height, width = densitySlice.shape
    heightAxis = np.arange(height)
    widthAxis = np.arange(width)
    densitySlice = interp.RectBivariateSpline(heightAxis, widthAxis, densitySlice)

    pixelSize = 0.108  # cm
    mu = 0.2
    eps = 1e-4

    CTFile = "/data/qifan/FastDoseWorkplace/TCIAAdd/002/CT.nrrd"
    CTArray, CTHeader = nrrd.read(CTFile)
    CTArray = np.transpose(CTArray, axes=(2, 1, 0))
    CTArray += 1024
    sliceIdx = 10
    canvas = CTArray[sliceIdx, :, :]
    canvas = canvas[230: 330, 50: 150]

    fig, ax = plt.subplots(figsize=(canvas.shape[0] / 20, canvas.shape[1] / 20), dpi=100)
    ax.imshow(canvas, cmap="gray", vmin=500, vmax=1500)

    marginSize = 2
    contentSize = 16
    stride = marginSize + contentSize
    starting = 5
    numCells = 5

    doseValues = drawCCCS2Dose()
    doseArray = np.zeros(canvas.shape, dtype=float)
    for i in range(numCells+2):
        idx_begin_x = starting + (i-1) * stride + marginSize
        idx_begin_x = max(0, idx_begin_x)
        idx_end_x = starting + i * stride
        idx_end_x = min(idx_end_x, 500)
        for j in range(numCells+2):
            idx_begin_y = starting + (j-1) * stride + marginSize
            idx_begin_y = max(0, idx_begin_y)
            idx_end_y = starting + j * stride
            idx_end_y = min(idx_end_y, 500)
            doseArray[idx_begin_x: idx_end_x, idx_begin_y: idx_end_y] = doseValues[i, j]
    ax.imshow(doseArray, cmap="jet", vmin=0, vmax=np.max(doseArray), alpha=0.3 * (doseArray>eps))

    nCones = 12
    halfSize = 50
    nSamples = 100
    cmap = plt.get_cmap("jet")
    norm = plt.Normalize(vmin=0, vmax=1)
    anglePerCone = 2 * np.pi / nCones

    for i in range(nCones):
        rayAngle = i * anglePerCone
        rayDirection = np.array((-np.cos(rayAngle), np.sin(rayAngle)))
        scale1 = 1e+8  # initialize to infinite
        scale2 = 1e+8
        if np.abs(rayDirection[0]) > eps:
            scale1 = abs(halfSize / rayDirection[0])
        if np.abs(rayDirection[1]) > eps:
            scale2 = abs(halfSize / rayDirection[1])
        scale = scale1 if scale1 < scale2 else scale2
        target = np.clip(rayDirection * scale, - halfSize + eps, halfSize - 1 - eps)
        beginPoint = np.array((50.5, 50.5))
        endPoint = beginPoint + target
        path_x = np.linspace(beginPoint[0], endPoint[0], nSamples)
        path_y = np.linspace(beginPoint[1], endPoint[1], nSamples)
        path_z = densitySlice(path_x, path_y, grid=False)
        radPathLength = np.cumsum(path_z) * pixelSize * mu
        path_dose = np.exp(-radPathLength)
        path_dose = cmap(norm(path_dose))
        for j in range(len(path_x) - 1):
            ax.plot(path_x[j: j+2], path_y[j: j+2], color=path_dose[j], linewidth=4)

    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.canvas.draw()
    image_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image_array = image_array.copy()

    marginSize *= 5
    contentSize *= 5
    stride *= 5
    starting *= 5
    for i in range(numCells + 1):
        idxBegin = starting + stride * i
        idxEnd = idxBegin + marginSize
        image_array[idxBegin: idxEnd, :, :] = 255
        image_array[:, idxBegin: idxEnd, :] = 255
    
    figureFile = os.path.join(topicFolder, "collapsedCone(b).png")
    plt.imsave(figureFile, image_array)


def drawCCCS2Dose():
    """
    In this function, we calculate the per-voxel dose
    """
    canvas = os.path.join(topicFolder, "densitySlice.png")
    canvas = plt.imread(canvas)
    drawCCCS2ResultFile = os.path.join(topicFolder, "collapsedCone(b).png")

    CTFile = "/data/qifan/FastDoseWorkplace/TCIAAdd/002/CT.nrrd"
    CTArray, CTHeader = nrrd.read(CTFile)
    CTArray = np.transpose(CTArray, axes=(2, 1, 0))
    CTArray += 1024
    sliceIdx = 10
    densitySlice = CTArray[sliceIdx, :, :].copy()
    densitySlice = densitySlice[230: 330, 50: 150]
    densitySlice = densitySlice / 1024
    height, width = densitySlice.shape
    heightAxis = np.arange(height)
    widthAxis = np.arange(width)
    densitySlice = interp.RectBivariateSpline(heightAxis, widthAxis, densitySlice)

    pixelSize = 0.108  # cm
    mu = 0.2
    eps = 1e-4
    marginSize = 2
    contentSize = 16
    stride = marginSize + contentSize
    starting = 5
    numCells = 5

    # Each cell contains line segments that pass through the voxel
    drawers = []
    for i in range(numCells + 2):
        row = []
        for j in range(numCells + 2):
            row.append([])
        drawers.append(row)

    nCones = 12
    halfSize = 50
    nSamples = 100
    anglePerCone = 2 * np.pi / nCones
    for i in range(nCones):
        rayAngle = i * anglePerCone
        rayDirection = np.array((-np.cos(rayAngle), np.sin(rayAngle)))
        scale1 = 1e+8  # initialize to infinite
        scale2 = 1e+8
        if np.abs(rayDirection[0]) > eps:
            scale1 = abs(halfSize / rayDirection[0])
        if np.abs(rayDirection[1]) > eps:
            scale2 = abs(halfSize / rayDirection[1])
        scale = scale1 if scale1 < scale2 else scale2
        target = np.clip(rayDirection * scale, - halfSize + eps, halfSize - 1 - eps)
        beginPoint = np.array((50.5, 50.5))
        endPoint = beginPoint + target
        path_x = np.linspace(beginPoint[0], endPoint[0], nSamples)
        path_y = np.linspace(beginPoint[1], endPoint[1], nSamples)
        path_z = densitySlice(path_x, path_y, grid=False)

        ds = np.sqrt((path_x[1] - path_x[0]) ** 2 + (path_y[1] - path_y[0]) ** 2)
        path_dose = np.cumsum(path_z) * ds * pixelSize * mu
        path_dose = np.exp(-path_dose)

        # segment the full ray into each voxel
        for j in range(numCells + 2):
            rowLower = starting + (j - 1) * stride + marginSize
            rowHigher = starting + j * stride
            flagRow = np.logical_and(path_x >= rowLower, path_x < rowHigher)
            if np.any(flagRow) == 0:
                continue
            for k in range(numCells + 2):
                colLower = starting + (k - 1) * stride + marginSize
                colHigher = starting + k * stride
                flagCol = np.logical_and(path_y >= colLower, path_y < colHigher)
                flagDrawer = np.logical_and(flagRow, flagCol)
                if np.any(flagDrawer) == 0:
                    continue
                path_x_drawer = path_x[flagDrawer].copy()
                path_y_drawer = path_y[flagDrawer].copy()
                path_dose_drawer = path_dose[flagDrawer].copy()
                path_x_drawer = np.expand_dims(path_x_drawer, axis=1)
                path_y_drawer = np.expand_dims(path_y_drawer, axis=1)
                path_dose_drawer = np.expand_dims(path_dose_drawer, axis=1)
                entry = np.concatenate([path_x_drawer, path_y_drawer, path_dose_drawer], axis=1)
                drawers[j][k].append(entry)
    
    # just return a dose map
    doseArray = (numCells+2, numCells+2)
    doseArray = np.zeros(doseArray, dtype=float)
    for i in range(numCells+2):
        for j in range(numCells+2):
            drawer = drawers[i][j]
            if len(drawer) == 0:
                continue
            nPoints = 0
            cumu = 0
            for entry in drawer:
                path_dose_drawer = entry[:, 2]
                nPoints += path_dose_drawer.size
                cumu += np.sum(path_dose_drawer)
            doseArray[i, j] = cumu / nPoints
    if False:
        plt.imsave(drawCCCS2ResultFile, doseArray, cmap="jet", vmin=0, vmax=np.max(doseArray))
    return doseArray

def figureGroup():
    """
    Stitch the four figures generated above into one
    """
    image1 = os.path.join(topicFolder, "fullConv(a).png")
    image1 = plt.imread(image1)
    image2 = os.path.join(topicFolder, "fullConv(b).png")
    image2 = plt.imread(image2)
    image3 = os.path.join(topicFolder, "collapsedCone(a).png")
    image3 = plt.imread(image3)
    image4 = os.path.join(topicFolder, "collapsedCone(b).png")
    image4 = plt.imread(image4)
    
    commonShape = (500, 500, 4)
    assert image1.shape == commonShape and image2.shape == commonShape \
        and image3.shape == commonShape and image4.shape == commonShape
    verticalGap = 80
    horizontalGap = 50
    horiGapShape = (commonShape[0], horizontalGap, commonShape[2])
    horiGap = np.ones(horiGapShape, dtype=image1.dtype)
    vertiGapShape = (verticalGap, 2 * commonShape[1] + horizontalGap, commonShape[2])
    vertiGap = np.ones(vertiGapShape, dtype=image1.dtype)
    firstRow = np.concatenate([image1, horiGap, image2], axis=1)
    secondRow = np.concatenate([image3, horiGap, image4], axis=1)
    result = np.concatenate([firstRow, vertiGap, secondRow, vertiGap], axis=0)

    fig, ax = plt.subplots(figsize=(result.shape[1] / 100, result.shape[0] / 100), dpi=100)
    ax.imshow(result)

    # write labels to the image
    titles = [["(a)", "(b)"], ["(c)", "(d)"]]
    offsetVertical = 30
    for i in range(2):
        vertiOffset = i * (commonShape[0] + verticalGap) + commonShape[0] + offsetVertical
        for j in range(2):
            horiOffset = j * (commonShape[1] + horizontalGap) + commonShape[1] / 2
            ax.text(horiOffset, vertiOffset, titles[i][j], fontsize=40,
                color="black", ha="center", va="center")
    
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    imagePath = os.path.join(topicFolder, "CCCSAhnesjo.png")
    plt.savefig(imagePath)
    plt.close(fig)
    plt.clf()
    # plt.imsave(imagePath, result)


if __name__ == "__main__":
    # densitySliceInit()
    # drawFullConv1()
    # drawFullConv2()
    # drawCCCS1()
    # drawCCCS2()
    figureGroup()