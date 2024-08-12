import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import random
from skimage.transform import resize

figuresFolder = "/data/qifan/AAPM2024/manufigures"
topicFolder = os.path.join(figuresFolder, "beamSparseMat")
if not os.path.isdir(topicFolder):
    os.mkdir(topicFolder)

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())

def figurePart1():
    numRowsTotal = 4.5
    numRowsValid = 3
    elementsBegin = 4
    elementsEllipse = 2
    elementsEnd = 1

    rowHeight = 1.2
    pointSpacing = 1.0
    pointRadius = 0.4
    transparency = 0.5
    padding = 0.25
    dotSpacing = 0.6
    dotRadius = 0.08

    totalHeight = numRowsTotal * rowHeight
    totalWidth = pointSpacing * (elementsBegin + elementsEllipse + elementsEnd)

    fig, ax = plt.subplots(figsize=(totalWidth+2*padding, totalHeight+2*padding), dpi=100)

    # draw the bounding boxes and elements within it.
    validRows = [0, 1, 3.5]
    for i, idx in enumerate(validRows):
        rectangleCoordsY = totalHeight / 2 - (idx + 0.5) * rowHeight - 0.5 * pointSpacing
        rectangleCoordsX = -totalWidth / 2
        color = colors[i]
        rounded_rect = patches.FancyBboxPatch(
            (rectangleCoordsX, rectangleCoordsY), totalWidth, pointSpacing,
            boxstyle="round,pad=0.0,rounding_size=0.5",
            edgecolor=color,
            facecolor=color,
            linewidth=2,
            alpha=transparency)
        ax.add_patch(rounded_rect)

        for j in list(range(elementsBegin)) + [elementsBegin + elementsEllipse + elementsEnd - 1]:
            pointCoordsX = (j + 0.5) * pointSpacing - totalWidth / 2
            pointCoordsY = totalHeight / 2 - (idx + 0.5) * rowHeight
            circle = patches.Circle((pointCoordsX, pointCoordsY),
                pointRadius, edgecolor=color, facecolor=color, alpha=0.3)
            ax.add_patch(circle)
        
        dotCenterX = elementsBegin + elementsEllipse / 2 - \
            (elementsBegin + elementsEllipse + elementsEnd) / 2
        for x in [-1, 0, 1]:
            dotCoordsX = dotCenterX * pointSpacing + x * dotSpacing
            dotCoordsY = totalHeight / 2 - (idx + 0.5) * rowHeight
            circle = patches.Circle((dotCoordsX, dotCoordsY), dotRadius,
                edgecolor="black", facecolor="black")
            ax.add_patch(circle)
    
    # draw the dots between the rectangles
    dotCentralY = (2 + 3.5) / 2
    dotCentralY = totalHeight / 2 - dotCentralY * rowHeight
    for y in [-1, 0, 1]:
        dotCoordsX = 0
        dotCoordsY = dotCentralY + y * dotSpacing
        circle = patches.Circle((dotCoordsX, dotCoordsY), dotRadius,
            edgecolor="black", facecolor="black")
        ax.add_patch(circle)

    ax.set_xlim(-totalWidth/2-padding, totalWidth/2+padding)
    ax.set_ylim(-totalHeight/2-padding, totalHeight/2+padding)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    figureFile = os.path.join(topicFolder, "part1.png")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()


def figurePart2():
    edgeSize = 8
    pointSpacing = 1.0
    pointRadius = 0.4
    transparency = 0.5
    SAD = 16
    beamAngle = np.pi / 6
    halfAngle = np.pi / 36
    color = colors[1]

    figSize = edgeSize * pointSpacing
    fig, ax = plt.subplots(figsize=(figSize, figSize), dpi=100)

    # draw rectangle
    rectangleCoordsX = - edgeSize * pointSpacing / 2
    rectangleCoordsY = rectangleCoordsX
    rounded_rect = patches.FancyBboxPatch(
        (rectangleCoordsX, rectangleCoordsY), figSize, figSize,
        boxstyle="round,pad=0.0,rounding_size=0.5",
        edgecolor=color,
        facecolor=color,
        linewidth=2,
        alpha=transparency)
    ax.add_patch(rounded_rect)

    # draw irradiated area
    x0, y0 = SAD * np.sin(beamAngle), SAD * np.cos(beamAngle)
    k1 = 1 / np.tan(beamAngle + halfAngle)
    k2 = 1 / np.tan(beamAngle - halfAngle)
    
    lowerBoundX = - edgeSize * pointSpacing / 2
    upperBoundX = edgeSize * pointSpacing / 2
    lowerBoundY = - edgeSize * pointSpacing / 2
    upperBoundY = edgeSize * pointSpacing / 2
    
    intersect0, intersect1 = calcIntersection(
        lowerBoundX, upperBoundX, lowerBoundY, upperBoundY, x0, y0, k1)
    plt.plot((intersect0[0], intersect1[0]), (intersect0[1], intersect1[1]),
        linestyle="--", color=colors[2], linewidth=3)
    
    intersect0, intersect1 = calcIntersection(
        lowerBoundX, upperBoundX, lowerBoundY, upperBoundY, x0, y0, k2)
    plt.plot((intersect0[0], intersect1[0]), (intersect0[1], intersect1[1]),
        linestyle="--", color=colors[2], linewidth=3)

    # draw circles
    norm1 = (-k1, 1) / np.sqrt(k1**2 + 1)
    norm2 = (-k2, 1) / np.sqrt(k2**2 + 1)
    for i in range(edgeSize):
        circleCoordsX = (- edgeSize / 2 + i + 0.5) * pointSpacing
        circleCoordsY_upper = k1 * (circleCoordsX - x0) + y0
        circleCoordsY_lower = k2 * (circleCoordsX - x0) + y0
        for j in range(edgeSize):
            circleCoordsY = (- edgeSize / 2 + j + 0.5) * pointSpacing
            distance = np.array((x0, y0)) - np.array((circleCoordsX, circleCoordsY))
            distance1 = np.abs(np.sum(norm1 * distance))
            distance2 = np.abs(np.sum(norm2 * distance))
            if circleCoordsY >= circleCoordsY_lower and circleCoordsY <= circleCoordsY_upper:
                transparency_local = 1
            elif distance1 < pointRadius or distance2 < pointRadius:
                transparency_local = 1
            else:
                transparency_local = 0.5
            circle = patches.Circle((circleCoordsX, circleCoordsY),
                pointRadius, edgecolor=color, facecolor=color,
                alpha=transparency_local)
            ax.add_patch(circle)

    ax.set_xlim(rectangleCoordsX, -rectangleCoordsX)
    ax.set_ylim(rectangleCoordsY, -rectangleCoordsY)
    ax.axis("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figureFile = os.path.join(topicFolder, "part2.png")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.close()


def calcIntersection(lowerX, upperX, lowerY, upperY, x0, y0, k):
    lowerBoundX = lowerX - x0
    upperBoundX = upperX - x0
    lowerBoundY = (lowerY - y0) / k
    upperBoundY = (upperY - y0) / k
    lowerXt, upperXt = min(lowerBoundX, upperBoundX), max(lowerBoundX, upperBoundX)
    lowerYt, upperYt = min(lowerBoundY, upperBoundY), max(lowerBoundY, upperBoundY)
    lower = max(lowerXt, lowerYt)
    upper = min(upperXt, upperYt)
    assert lower < upper
    intersect_lower = (x0 + lower, y0 + lower * k)
    intersect_upper = (x0 + upper, y0 + upper * k)
    return intersect_lower, intersect_upper


def figurePart3():
    solidProb = 0.4
    random.seed(10086)
    numRowsTotal = 4.5
    numRowsValid = 3
    elementsBegin = 4
    elementsEllipse = 2
    elementsEnd = 1

    rowHeight = 1.2
    pointSpacing = 1.0
    pointRadius = 0.4
    padding = 0.25
    dotSpacing = 0.6
    dotRadius = 0.08

    totalHeight = numRowsTotal * rowHeight
    totalWidth = pointSpacing * (elementsBegin + elementsEllipse + elementsEnd)

    fig, ax = plt.subplots(figsize=(totalWidth+2*padding, totalHeight+2*padding), dpi=100)

    # draw the bounding boxes and elements within it.
    validRows = [0, 1, 3.5]
    for i, idx in enumerate(validRows):
        rectangleCoordsY = totalHeight / 2 - (idx + 0.5) * rowHeight - 0.5 * pointSpacing
        rectangleCoordsX = -totalWidth / 2
        color = colors[i]
        rounded_rect = patches.FancyBboxPatch(
            (rectangleCoordsX, rectangleCoordsY), totalWidth, pointSpacing,
            boxstyle="round,pad=0.0,rounding_size=0.5",
            edgecolor=color,
            facecolor=color,
            linewidth=2,
            alpha=0.5)
        ax.add_patch(rounded_rect)

        for j in list(range(elementsBegin)) + [elementsBegin + elementsEllipse + elementsEnd - 1]:
            pointCoordsX = (j + 0.5) * pointSpacing - totalWidth / 2
            pointCoordsY = totalHeight / 2 - (idx + 0.5) * rowHeight

            if random.random() < solidProb:
                transparency = 1
            else:
                transparency = 0.3
            circle = patches.Circle((pointCoordsX, pointCoordsY),
                pointRadius, edgecolor=color, facecolor=color, alpha=transparency)
            ax.add_patch(circle)
        
        dotCenterX = elementsBegin + elementsEllipse / 2 - \
            (elementsBegin + elementsEllipse + elementsEnd) / 2
        for x in [-1, 0, 1]:
            dotCoordsX = dotCenterX * pointSpacing + x * dotSpacing
            dotCoordsY = totalHeight / 2 - (idx + 0.5) * rowHeight
            circle = patches.Circle((dotCoordsX, dotCoordsY), dotRadius,
                edgecolor="black", facecolor="black")
            ax.add_patch(circle)
    
    # draw the dots between the rectangles
    dotCentralY = (2 + 3.5) / 2
    dotCentralY = totalHeight / 2 - dotCentralY * rowHeight
    for y in [-1, 0, 1]:
        dotCoordsX = 0
        dotCoordsY = dotCentralY + y * dotSpacing
        circle = patches.Circle((dotCoordsX, dotCoordsY), dotRadius,
            edgecolor="black", facecolor="black")
        ax.add_patch(circle)

    ax.set_xlim(-totalWidth/2-padding, totalWidth/2+padding)
    ax.set_ylim(-totalHeight/2-padding, totalHeight/2+padding)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    figureFile = os.path.join(topicFolder, "part3.png")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()


def beamSparseMatGroup():
    part1 = os.path.join(topicFolder, "part1.png")
    part2 = os.path.join(topicFolder, "part2.png")
    part3 = os.path.join(topicFolder, "part3.png")

    part1 = plt.imread(part1)
    part2 = plt.imread(part2)
    part3 = plt.imread(part3)

    assert part1.shape == part3.shape
    height = part1.shape[0]
    shape2 = part2.shape
    newShape2 = (height, int(shape2[1] * height / shape2[0]))
    part2 = resize(part2, newShape2)

    group = np.concatenate((part1, part2, part3), axis=1)
    titleHeight = 50
    titleShape = (titleHeight, group.shape[1], 4)
    titleBand = np.ones(titleShape, dtype=np.float32)
    group = np.concatenate((group, titleBand), axis=0)
    fig, ax = plt.subplots(figsize=(group.shape[1]/100, group.shape[0]/100), dpi=100)
    ax.imshow(group)

    # add labels
    coords1 = (part1.shape[1] / 2, height + titleHeight / 3)
    ax.text(coords1[0], coords1[1], "(a)", fontsize=40,
        horizontalalignment="center", verticalalignment="center")
    coords2 = (part1.shape[1] + part2.shape[1] / 2, height + titleHeight / 3)
    ax.text(coords2[0], coords2[1], "(b)", fontsize=40,
        horizontalalignment="center", verticalalignment="center")
    coords3 = (part1.shape[1] + part2.shape[1] + part3.shape[1] / 2, height + titleHeight / 3)
    ax.text(coords3[0], coords3[1], "(c)", fontsize=40)

    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figurePath = os.path.join(topicFolder, "beamSparseMat.png")
    plt.savefig(figurePath)
    plt.close(fig)
    plt.clf()


if __name__ == "__main__":
    # figurePart1()
    # figurePart2()
    # figurePart3()
    beamSparseMatGroup()