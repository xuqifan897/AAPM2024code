import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import string

halfHeightGlobal = 120
halfWidthGlobal = 60
tailMarginGlobal = 20
headMarginGlobal = 100

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())

figureFolder = "/data/qifan/AAPM2024/manufigures"

def drawMedium(ax, colorValid, colorInvalid, offset, halfHeight,
    halfWidth, tailMargin, headMargin, gridRes, gridColor):
    """
    This function generates the cross-section of a cylindrical phantom
    """
    # region 1, tail margin
    tailMarginCorner = offset + np.array((-halfWidth, halfHeight-tailMargin))
    tailMarginRect = patches.Rectangle(tailMarginCorner, 2*halfWidth, tailMargin,
        linewidth=1, edgecolor=colorInvalid, facecolor=colorInvalid, zorder=0)
    # tailMarginRect = patches.Rectangle(tailMarginCorner, 2*halfWidth, tailMargin)
    ax.add_patch(tailMarginRect)

    # regin 2, valid range
    validAreaHeight = 2 * halfHeight - tailMargin - headMargin
    validAreaCorner = tailMarginCorner + np.array((0, -validAreaHeight))
    validAreaRect = patches.Rectangle(validAreaCorner, 2*halfWidth, validAreaHeight,
        linewidth=1, edgecolor=colorValid, facecolor=colorValid, zorder=0)
    ax.add_patch(validAreaRect)

    # region 3, head margin
    headMarginCorner = validAreaCorner + np.array((0, -headMargin))
    headMarginRect = patches.Rectangle(headMarginCorner, 2*halfWidth, headMargin,
        linewidth=1, edgecolor=colorInvalid, facecolor=colorInvalid, zorder=0)
    ax.add_patch(headMarginRect)

    # draw the coordinate lines
    # horizontal lines
    nHoriLines = int(2 * halfHeight / gridRes) - 1
    for i in range(nHoriLines):
        coordStart = np.array((-halfWidth, halfHeight - (i + 1) * gridRes)) + offset
        coordEnd = np.array((halfWidth, halfHeight - (i + 1) * gridRes)) + offset
        ax.plot((coordStart[0], coordEnd[0]), (coordStart[1], coordEnd[1]),
            linewidth=2, color=gridColor, linestyle="--", zorder=0)
    nVertiLines = int(2 * halfWidth / gridRes) - 1
    for i in range(nVertiLines):
        coordStart = np.array((-halfWidth + (i + 1) * gridRes, halfHeight)) + offset
        coordEnd = np.array((-halfWidth + (i + 1) * gridRes, -halfHeight)) + offset
        ax.plot((coordStart[0], coordEnd[0]), (coordStart[1], coordEnd[1]),
            linewidth=2, color=gridColor, linestyle="--", zorder=0)


def colorLighter(color, weight):
    color = color.astype(float)
    color = weight * color + (1 - weight) * 255
    color = color / 255
    return color


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return np.array(tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)))


class Point():
    def __init__(self):
        self.initialFlag = False
        self.angle = 0  # up to down
        self.coords = np.array((0, 0), float)
        self.children = []

        # for interaction
        self.PoissonLambda = 2  # the expected number of child particles
        self.PoissonLambdaDiscount = 0.8
        
        self.minimumChild = 3
        self.minimumChild_factor = 0.8

        self.mu = 60
        self.mu_factor = 0.8
        
        self.minimumLength = 40
        self.minimumLength_factor = 0.8

        self.angleSigma = np.pi / 8
        self.angleSigma_factor = 0.9

        self.leftBound = -60
        self.rightBound = 60
        self.topBound = 100
        self.bottomBound = -100

    def sampleChildren(self):
        # firstly, sample the number of child particles using Poisson distribution
        NumParticles = np.random.poisson(self.PoissonLambda)
        NumParticles = max(NumParticles, int(self.minimumChild))
        childMinimumLength = self.minimumLength * self.minimumLength_factor
        numOffsprings = 0
        for i in range(NumParticles):
            # sample direction
            childAngle = np.random.normal(self.angle, self.angleSigma)
            childMu = self.mu * self.mu_factor
            childLength = np.random.exponential(childMu)
            childLength = max(childLength, childMinimumLength)
            childDirection = np.array((np.sin(childAngle), -np.cos(childAngle)))
            childCoords = self.coords + childLength * childDirection
            valid = (self.leftBound < childCoords[0]) and (childCoords[0] < self.rightBound) \
                and (self.bottomBound < childCoords[1]) and (childCoords[1] < self.topBound)
            
            if valid:
                self.children.append(Point())
                self.children[-1].angle = childAngle
                self.children[-1].coords = childCoords
                self.children[-1].PoissonLambda = self.PoissonLambda * self.PoissonLambdaDiscount
                self.children[-1].minimumChild = self.minimumChild * self.minimumChild_factor
                self.children[-1].mu = self.mu * self.mu_factor
                self.children[-1].minimumLength = childMinimumLength
                self.children[-1].angleSigma = self.angleSigma_factor * self.angleSigma + \
                    (1 - self.angleSigma_factor) * np.pi
                numOffsprings += self.children[-1].sampleChildren()
        return numOffsprings + 1


def drawScatter(ax, initialPoint: Point, offset, color):
    if initialPoint.initialFlag:
        initialCoords = initialPoint.coords + offset
        circle = patches.Circle(tuple((initialCoords[0], initialCoords[1])), radius=3, color=color)
        ax.add_patch(circle)
    for child in initialPoint.children:
        arrow = patches.FancyArrowPatch(
            initialPoint.coords+offset, child.coords+offset,
            arrowstyle="->", mutation_scale=20, color=color, linewidth=3)
        ax.add_patch(arrow)
        # ax.plot((initialPoint.coords[0], child.coords[0]),
        #     (initialPoint.coords[1], child.coords[1]), linewidth=2, color=color)
        drawScatter(ax, child, offset, color)


def main0():
    colorWeight = 0.8
    colorValid = colorLighter(hex_to_rgb(colors[0]), colorWeight)
    colorInvalid = colorLighter(hex_to_rgb(colors[7]), colorWeight)
    colorGrid = colorLighter(hex_to_rgb(colors[1]), colorWeight)
    gridRes = 20
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

    offset1 = np.array((80, 0))

    drawMedium(ax, colorValid, colorInvalid, offset1, halfHeightGlobal, halfWidthGlobal,
        tailMarginGlobal, headMarginGlobal, gridRes, colorGrid)
    
    point1 = Point()
    point1.initialFlag = True
    point1.coords = np.array((0, 90))
    numPoint1 = 0
    while numPoint1 < 30:
        point1.children.clear()
        numPoint1 = point1.sampleChildren()
    point1Parent = Point()
    point1Parent.coords = np.array((0, halfHeightGlobal))
    point1Parent.children.append(point1)
    drawScatter(ax, point1Parent, offset1, colors[3])

    offset2 = np.array((-80, 0))
    drawMedium(ax, colorValid, colorInvalid, offset2, halfHeightGlobal, halfWidthGlobal,
        tailMarginGlobal, headMarginGlobal, gridRes, colorGrid)

    point2 = Point()
    point2.initialFlag = True
    point2.coords = np.array((0, 25))
    numPoint2 = 0
    while numPoint2 < 30:
        point2.children.clear()
        numPoint2 = point2.sampleChildren()
    point2Parent = Point()
    point2Parent.coords = np.array((0, halfHeightGlobal))
    point2Parent.children.append(point2)
    drawScatter(ax, point2Parent, offset2, colors[4])

    figureHalfWidth = 80 + halfWidthGlobal + 10
    ax.set_xlim(-figureHalfWidth, figureHalfWidth)
    ax.set_ylim(-halfHeightGlobal-10, halfHeightGlobal+10)
    ax.set_aspect("equal")
    plt.tight_layout()
    figurePath = os.path.join(figureFolder, "KernelGen.png")
    plt.savefig(figurePath)
    plt.close(fig)
    plt.clf()


def main1():
    colorWeight = 0.7
    colorValid = colorLighter(hex_to_rgb(colors[0]), colorWeight)
    colorInvalid = colorLighter(hex_to_rgb(colors[7]), colorWeight)
    colorGrid = colorLighter(hex_to_rgb(colors[1]), colorWeight)
    gridRes = 20
    figureSpacing = 10

    # np.random.seed(90095)
    np.random.seed(94158)

    numFigures = 6
    figureTotalWidth = numFigures * 2 * halfWidthGlobal + (numFigures - 1) * figureSpacing
    textSpacing = 20
    figureTotalHeight = 2 * halfHeightGlobal + textSpacing
    figSizeHeight = 10
    figSizeWidth = figSizeHeight * figureTotalWidth / figureTotalHeight
    fig, ax = plt.subplots(figsize=(figSizeWidth, figSizeHeight), dpi=100)

    offsetList = [np.array((0, 80)), np.array((0, 110)), np.array((0, -50)),
        np.array((0, 0)), np.array((0, 40))]
    assert len(offsetList) < numFigures
    pointList = []
    # colorsLocal = ["blue", colors[3], colors[4], colors[5], colors[6]]
    colorsLocal = ["blue", "red", "purple", "brown", colors[1]]
    for i in range(len(offsetList)):
        localOffset = np.array((-figureTotalWidth / 2 + i * (figureSpacing + 2 * halfWidthGlobal) 
            + halfWidthGlobal, 0))
        
        drawMedium(ax, colorValid, colorInvalid, localOffset, halfHeightGlobal, halfWidthGlobal,
            tailMarginGlobal, headMarginGlobal, gridRes, colorGrid)
        
        pointList.append(Point())
        pointList[-1].coords = offsetList[i]
        pointList[-1].initialFlag = True
        numPointLocal = 0
        while numPointLocal < 30:
            pointList[-1].children.clear()
            numPointLocal = pointList[-1].sampleChildren()
        
        localParentPoint = Point()
        localParentPoint.coords = np.array((0, halfHeightGlobal))
        localParentPoint.children.append(pointList[-1])
        drawScatter(ax, localParentPoint, localOffset, colorsLocal[i])
        print(i)
    
    validList = [pointList[0], pointList[3], pointList[4]]
    colorPlot = [colorsLocal[0], colorsLocal[3], colorsLocal[4]]
    aggregatePlot(ax, validList, colorPlot, colorValid, halfHeightGlobal, halfWidthGlobal,
        tailMarginGlobal, headMarginGlobal, figureSpacing, numFigures,
        gridRes, colorGrid)
    
    # put text
    texts = ["({})".format(a) for a in string.ascii_lowercase]
    for i in range(6):
        localOffset = np.array((-figureTotalWidth / 2 + i * (figureSpacing + 2 * halfWidthGlobal) 
            + halfWidthGlobal, -halfHeightGlobal - 5))
        plt.text(localOffset[0], localOffset[1], texts[i], fontsize=40, va="top", ha="center")
    
    ax.set_xlim(-figureTotalWidth / 2, figureTotalWidth / 2)
    ax.set_ylim(-figureTotalHeight / 2 - textSpacing, figureTotalHeight / 2)
    ax.axis("off")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figurePath = os.path.join(figureFolder, "KernelGen.png")
    plt.savefig(figurePath)
    figurePath = os.path.join(figureFolder, "KernelGen.eps")
    plt.savefig(figurePath)
    plt.close(fig)
    plt.clf()


def aggregatePlot(ax, validList: list[Point], colorPlot, colorValid,
    halfHeight, halfWidth, tailMargin, headMargin, figureSpacing, numFigures,
    gridRes, gridColor):
    # firstly, draw the canvas
    HeightValid = 2 * halfHeight - tailMargin - headMargin
    WidthValid = 2 * halfWidth
    
    figureTotalWidth = numFigures * 2 * halfWidth + (numFigures - 1) * figureSpacing
    aggregateOffset = (figureTotalWidth / 2 - WidthValid / 2, 0)

    drawMedium(ax, colorValid, colorValid, aggregateOffset, HeightValid/2,
        WidthValid/2, 0, 0, gridRes, gridColor)
    
    assert len(validList) == len(colorPlot)
    for i in range(len(validList)):
        # calculate the offset needed
        point = validList[i]
        point_copied = Point()
        bottomBound  = point.coords[1] - headMargin
        topBound = point.coords[1] + tailMargin
        pointClone(point, point_copied, -WidthValid/2, WidthValid/2, topBound, bottomBound)
        localOffset = np.array((0, HeightValid / 2 - tailMargin - point.coords[1]))
        localOffset = localOffset + aggregateOffset
        drawScatter(ax, point_copied, localOffset, colorPlot[i])


def pointClone(sample: Point, clone: Point, leftBound, rightBound, topBound, bottomBound):
    clone.initialFlag = sample.initialFlag
    clone.angle = sample.angle
    clone.coords = sample.coords
    clone.children = []

    clone.PoissonLambda = sample.PoissonLambda
    clone.PoissonLambdaDiscount = sample.PoissonLambdaDiscount
    clone.minimumChild = sample.minimumChild
    clone.minimumChild_factor = sample.minimumChild_factor
    clone.mu = sample.mu
    clone.mu_factor = sample.mu_factor
    clone.minimumLength = sample.minimumLength
    clone.minimumLength_factor = sample.minimumLength_factor
    clone.angleSigma = sample.angleSigma
    clone.angleSigma_factor = sample.angleSigma_factor
    clone.leftBound = leftBound
    clone.rightBound = rightBound
    clone.topBound = topBound
    clone.bottomBound = bottomBound

    for i in range(len(sample.children)):
        childCoords = sample.children[i].coords
        validFlag = (leftBound < childCoords[0]) and (childCoords[0] < rightBound) \
            and (bottomBound < childCoords[1]) and (childCoords[1] < topBound)
        if validFlag:
            clone.children.append(Point())
            pointClone(sample.children[i], clone.children[-1],
                leftBound, rightBound, topBound, bottomBound)


if __name__ == "__main__":
    main1()