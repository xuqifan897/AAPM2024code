import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from skimage import transform, measure
from PIL import Image, ImageDraw, ImageFont
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

resultFolder = "/data/qifan/projects/FastDoseWorkplace/TCIAAdd"
figureFolder = "/data/qifan/projects/AAPM2024/manufigures"
patients = ["002", "003", "009", "013", "070", "125", "132", "190"]

StructureList = []
exclude = ["SKIN", "PTVMerge", "rind", "PTVSeg0", "PTVSeg1", "PTVSeg2", "PTVSeg3", "PTVMerge"]
Converge = {"BrainStem": ["BRAIN_STEM", "Brainstem", "BRAIN_STEM_PRV"],
            "OralCavity": ["oralcavity", "oralCavity", "ORAL_CAVITY", "OralCavity"],
            "OPTIC_NERVE": ["OPTIC_NERVE", "OPTC_NERVE"]}
ConvergeReverse = {}
for name, collection in Converge.items():
    for child in collection:
        ConvergeReverse[child] = name

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
colors_skip = [11, 13, 14, 16, 18]
idx = 18
for i in colors_skip:
    colors[i] = colors[idx]
    idx += 1
colorMap = {}

def StructsInit():
    """
    This function is to generate a coherent structure list for all patients
    """
    global StructureList, colorMap
    for patient in patients:
        if ".txt" in patient:
            continue
        patientFolder = os.path.join(resultFolder, patient)
        InputMaskFolder = os.path.join(patientFolder, "PlanMask")
        structuresLocal = os.listdir(InputMaskFolder)
        structuresLocal = [a.split(".")[0].replace(" ", "") for a in structuresLocal]
        for a in structuresLocal:
            if a not in StructureList:
                StructureList.append(a)
    StructureList_copy = []
    for name in StructureList:
        if name in ConvergeReverse:
            name = ConvergeReverse[name]
        if name not in StructureList_copy and name not in exclude and "+" not in name:
            StructureList_copy.append(name)
    StructureList = StructureList_copy.copy()
    for i in range(len(StructureList)):
        colorMap[StructureList[i]] = colors[i]


def DVH_plot():
    patients = [a for a in os.listdir(resultFolder) if ".txt" not in a]
    patients.sort()
    rowSize = 4
    colSize = 4
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(colSize, rowSize, width_ratios=[0.2, 5, 5, 5],
        height_ratios=[4, 4, 4, 0.2])
    
    # Create the common ylabel
    ylabel_block = fig.add_subplot(gs[:-1, 0])
    ylabel_block.text(0.9, 0.5, "Fractional Volume (%)", ha="center", va="center",
        rotation="vertical", fontsize=20)
    ylabel_block.axis("off")

    # Create the common xlabel
    xlabel_block = fig.add_subplot(gs[-1, 1:])
    xlabel_block.text(0.5, 0.9, "Dose (Gy)", ha="center", va="center", fontsize=20)
    xlabel_block.axis("off")

    # Create the DVH plots
    localRowSize = rowSize - 1
    localColSize = colSize - 1
    for i in range(len(patients)):
        patientFolder = os.path.join(resultFolder, patients[i])
        FastDoseFolder = os.path.join(patientFolder, "FastDose")
        dimensionFile = os.path.join(FastDoseFolder, "prep_output", "dimension.txt")
        with open(dimensionFile, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension = np.flip(dimension)

        MaskFolder = os.path.join(patientFolder, "PlanMask")
        StructuresLocal = [b for a in os.listdir(MaskFolder) if
            (b:=a.split(".")[0]).replace(" ", "") not in exclude and "+" not in a]
        maskDict = {}
        for struct in StructuresLocal:
            maskFile = os.path.join(MaskFolder, "{}.bin".format(struct))
            maskArray = np.fromfile(maskFile, dtype=np.uint8)
            maskArray = np.reshape(maskArray, dimension)
            if struct in ConvergeReverse:
                struct = ConvergeReverse[struct]
            name = struct.replace(" ", "")
            if name in ConvergeReverse:
                name = ConvergeReverse[name]
            maskDict[struct] = maskArray
        
        DoseExpFile = os.path.join(FastDoseFolder, "plan1", "dose.bin")
        DoseExp = np.fromfile(DoseExpFile, dtype=np.float32)
        DoseExp = np.reshape(DoseExp, dimension)

        DoseRefFile = os.path.join(patientFolder, "dose.bin")
        DoseRef = np.fromfile(DoseRefFile, dtype=np.float32)
        DoseRef = np.reshape(DoseRef, dimension)

        # normalize
        percentile_value = 5
        if patients[i] in ["003", "125"]:
            percentile_value = 10
        PrimaryPTVName = "PTV70"
        assert PrimaryPTVName in maskDict
        PrimaryMask = maskDict[PrimaryPTVName].astype(bool)
        thresh = np.percentile(DoseExp[PrimaryMask], percentile_value)
        DoseExp *= 70 / thresh
        thresh = np.percentile(DoseRef[PrimaryMask], percentile_value)
        DoseRef *= 70 / thresh

        rowIdx = i % localRowSize + 1
        colIdx = i // localRowSize
        assert colIdx < colSize - 1, "Figure index ({}, {}) error".format(rowIdx, colIdx)
        block = fig.add_subplot(gs[colIdx, rowIdx])

        for name, mask in maskDict.items():
            color = colorMap[name]
            mask = mask.astype(bool)
            StructDoseExp = DoseExp[mask]
            StructDoseExp = np.sort(StructDoseExp)
            StructDoseExp = np.insert(StructDoseExp, 0, 0.0)

            StructDoseRef = DoseRef[mask]
            StructDoseRef = np.sort(StructDoseRef)
            StructDoseRef = np.insert(StructDoseRef, 0, 0.0)

            y_axis = np.linspace(100, 0, np.sum(mask)+1)
            block.plot(StructDoseExp, y_axis, color=color, linewidth=2.0)
            block.plot(StructDoseRef, y_axis, color=color, linewidth=2.0, linestyle="--")
            print(name)

        block.set_xlim(0, 95)
        block.tick_params(axis="x", labelsize=16)
        block.tick_params(axis="y", labelsize=16)
        block.set_title("Patient {}".format(patients[i]), fontsize=20)
        print()

    # prepare legend
    legendBlock = fig.add_subplot(gs[colSize-2, rowSize-1])
    legendBlock.axis("off")
    handles = []
    labels = []
    for name, color in colorMap.items():
        handleEntry = plt.Line2D([0], [0], color=color, lw=2)
        handles.append(handleEntry)
        labels.append(name)
    legendBlock.legend(handles, labels, loc="center", ncol=2, fontsize=16)
    plt.tight_layout()

    figureFile = os.path.join(figureFolder, "FastDoseTCIAAdd.png")
    plt.savefig(figureFile)
    figureFile = os.path.join(figureFolder, "FastDoseTCIAAdd.eps")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()


def DrawDoseWash():
    ImageHeight = 80
    AxialWidth = 80
    CoronalWidth = 80
    SagittalWidth = 60
    sliceFolder = os.path.join(figureFolder, "DoseWashSample")
    if not os.path.isdir(sliceFolder):
        os.mkdir(sliceFolder)
    halfCropSize = 50

    SagittalIndices = {
        "002": 108,
        "003": 108,
        "009": 111,
        "013": 106,
        "070": 98,
        "125": 81,
        "132": 108,
        "190": 107
    }

    for i, patient in enumerate(patients):
        patientFolder = os.path.join(resultFolder, patient)
        dimensionFile = os.path.join(patientFolder,
            "FastDose", "prep_output", "dimension.txt")
        with open(dimensionFile, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension = np.flip(dimension)

        densityFile = os.path.join(patientFolder, "density_raw.bin")
        densityArray = np.fromfile(densityFile, dtype=np.uint16)
        densityArray = np.reshape(densityArray, dimension)

        MaskFolder = os.path.join(patientFolder, "PlanMask")
        structuresLocal = os.listdir(MaskFolder)
        structuresLocal = [b for a in structuresLocal if
            (b:=a.split(".")[0]).replace(" ", "") not in exclude and "+" not in a]
        BodyName = "SKIN"
        PTVMerge = "PTVMerge"
        structuresLocal.extend([BodyName, PTVMerge])
        structuresLocalNorm = {}
        for a in structuresLocal:
            name = a.replace(" ", "")
            if name not in ConvergeReverse:
                structuresLocalNorm[a] = name
            else:
                structuresLocalNorm[a] = ConvergeReverse[name]
        maskDict = {}
        for a, b in structuresLocalNorm.items():
            file = os.path.join(MaskFolder, a + ".bin")
            mask = np.fromfile(file, dtype=np.uint8)
            mask = np.reshape(mask, dimension)
            maskDict[b] = mask
        BodyMask = maskDict[BodyName].astype(bool)
        PTVMergeMask = maskDict[PTVMerge].astype(bool)
        del maskDict[BodyName]
        del maskDict[PTVMerge]

        # load FastDose result
        DoseExpFile = os.path.join(patientFolder, "FastDose", "plan1", "dose.bin")
        DoseExp = np.fromfile(DoseExpFile, dtype=np.float32)
        DoseExp = np.reshape(DoseExp, dimension)
        DoseExp[np.logical_not(BodyMask)] = 0

        # load clinical result
        DoseRefFile = os.path.join(patientFolder, "dose.bin")
        DoseRef = np.fromfile(DoseRefFile, dtype=np.float32)
        DoseRef = np.reshape(DoseRef, dimension)
        DoseRef[np.logical_not(BodyMask)] = 0

        # normalize
        percentile_value = 5
        if patient in ["003", "125"]:
            percentile_value = 10
        PrimaryStructureName = "PTV70"
        assert PrimaryStructureName in maskDict, "Structure {} not " \
            "found".format(PrimaryStructureName)
        PrimaryMask = maskDict[PrimaryStructureName]
        PrimaryMask = PrimaryMask.astype(bool)
        PrimaryExp = DoseExp[PrimaryMask]
        thresh = np.percentile(PrimaryExp, percentile_value)
        DoseExp *= 70 / thresh
        PrimaryRef = DoseRef[PrimaryMask]
        thresh = np.percentile(PrimaryRef, percentile_value)
        DoseRef *= 70 / thresh

        z, y, x = FindCentroid3D(PTVMergeMask)
        # if patient in SagittalIndices:
        #     x = SagittalIndices[patient]
        BodySlice = BodyMask[z, :, :]
        DensityAxial = densityArray[z, :, :]
        DoseExpAxial = DoseExp[z, :, :]
        DoseRefAxial = DoseRef[z, :, :]
        MasksSlice = [(a, b[z, :, :]) for a, b in maskDict.items()]
        file = os.path.join(sliceFolder, "patient{}ExpAxial.png".format(patient))
        DrawSlice(DensityAxial, DoseExpAxial, MasksSlice, BodySlice, ImageHeight, AxialWidth, file)
        file = os.path.join(sliceFolder, "patient{}RefAxial.png".format(patient))
        DrawSlice(DensityAxial, DoseRefAxial, MasksSlice, BodySlice, ImageHeight, AxialWidth, file)

        BodySlice = BodyMask[:, y, :]
        DensityCoronal = densityArray[:, y, :]
        DoseExpCoronal = DoseExp[:, y, :]
        DoseRefCoronal = DoseRef[:, y, :]
        MasksSlice = [(a, b[:, y, :]) for a, b in maskDict.items()]
        file = os.path.join(sliceFolder, "patient{}ExpCoronal.png".format(patient))
        DrawSlice(DensityCoronal, DoseExpCoronal, MasksSlice, BodySlice, ImageHeight, CoronalWidth, file)
        file = os.path.join(sliceFolder, "patient{}RefCoronal.png".format(patient))
        DrawSlice(DensityCoronal, DoseRefCoronal, MasksSlice, BodySlice, ImageHeight, CoronalWidth, file)

        BodySlice = BodyMask[:, :, x]
        DensitySagittal = densityArray[:, :, x]
        DoseExpSagittal = DoseExp[:, :, x]
        DoseRefSagittal = DoseRef[:, :, x]
        MasksSlice = [(a, b[:, :, x]) for a, b in maskDict.items()]
        file = os.path.join(sliceFolder, "patient{}ExpSagittal.png".format(patient))
        DrawSlice(DensitySagittal, DoseExpSagittal, MasksSlice, BodySlice, ImageHeight, SagittalWidth, file)
        file = os.path.join(sliceFolder, "patient{}RefSagittal.png".format(patient))
        DrawSlice(DensitySagittal, DoseRefSagittal, MasksSlice, BodySlice, ImageHeight, SagittalWidth, file)

        print("Patient {}".format(patient))


def FindCentroid3D(array):
    zWeight = np.sum(array, axis=(1, 2))
    zAxis = np.arange(zWeight.size)
    zCoord = np.sum(zWeight * zAxis) / np.sum(zWeight)
    zCoord = int(zCoord)

    yWeight = np.sum(array, axis=(0, 2))
    yAxis = np.arange(yWeight.size)
    yCoord = np.sum(yWeight * yAxis) / np.sum(yWeight)
    yCoord = int(yCoord)

    xWeight = np.sum(array, axis=(0, 1))
    xAxis = np.arange(xWeight.size)
    xCoord = np.sum(xWeight * xAxis) / np.sum(xWeight)
    xCoord = int(xCoord)

    return (zCoord, yCoord, xCoord)


def FindCentroid2D(array):
    yWeight = np.sum(array, axis=1)
    yAxis = np.arange(yWeight.size)
    yCoord = np.sum(yWeight * yAxis) / np.sum(yWeight)
    yCoord = int(yCoord)

    xWeight = np.sum(array, axis=0)
    xAxis = np.arange(xWeight.size)
    xCoord = np.sum(xWeight * xAxis) / np.sum(xWeight)
    xCoord = int(xCoord)

    return (yCoord, xCoord)


def DrawSlice(DensitySlice, DoseSlice, MasksSlice,
    BodySlice, height, width, file):
    DoseThresh = 3.0

    MaskCentroid = FindCentroid2D(BodySlice)
    Dim0LowerBound = MaskCentroid[0] - height
    Dim0LowerBound = max(0, Dim0LowerBound)
    Dim0HigherBound = MaskCentroid[0] + height
    Dim0HigherBound = min(Dim0HigherBound, BodySlice.shape[0])

    Dim1LowerBound = MaskCentroid[1] - width
    Dim1LowerBound = max(0, Dim1LowerBound)
    Dim1HigherBound = MaskCentroid[1] + width
    Dim1HigherBound = min(Dim1HigherBound, BodySlice.shape[1])

    DoseThresh = 3.0

    MaskCentroid = FindCentroid2D(BodySlice)
    Dim0LowerBound = MaskCentroid[0] - height
    Dim0LowerBound = max(0, Dim0LowerBound)
    Dim0HigherBound = MaskCentroid[0] + height
    Dim0HigherBound = min(Dim0HigherBound, BodySlice.shape[0])

    Dim1LowerBound = MaskCentroid[1] - width
    Dim1LowerBound = max(0, Dim1LowerBound)
    Dim1HigherBound = MaskCentroid[1] + width
    Dim1HigherBound = min(Dim1HigherBound, BodySlice.shape[1])

    RealHeight = Dim0HigherBound - Dim0LowerBound
    RealWidth = Dim1HigherBound - Dim1LowerBound

    DensitySlice = DensitySlice[Dim0LowerBound: Dim0HigherBound,
            Dim1LowerBound: Dim1HigherBound]
    DoseSlice = DoseSlice[Dim0LowerBound: Dim0HigherBound,
            Dim1LowerBound: Dim1HigherBound]
    fig, ax = plt.subplots(figsize=(RealWidth/10, RealHeight/10))
    ax.imshow(DensitySlice, cmap="gray", vmin=0, vmax=2000)

    for name, mask in MasksSlice:
        color = colorMap[name]
        maskSlice = mask[Dim0LowerBound: Dim0HigherBound,
            Dim1LowerBound: Dim1HigherBound]
        contours = measure.find_contours(maskSlice)
        initial = True
        for contour in contours:
            if initial:
                plt.plot(contour[:, 1], contour[:, 0], color=color, linewidth=4, label=name)
                initial = False
            else:
                plt.plot(contour[:, 1], contour[:, 0], color=color, linewidth=4)
    ax.imshow(DoseSlice, cmap="jet", vmin=0, vmax=95, alpha=(DoseSlice>DoseThresh)*0.3)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(file)
    plt.close(fig)
    plt.clf()


def CoalesceFigures():
    """
    This function groups multiple figures into one figure
    """
    sliceFolder = os.path.join(figureFolder, "DoseWashSample")
    directionList = [("Axial", 1600), ("Sagittal", 1600), ("Coronal", 1600)]
    height = 1600
    widthTotal = 0
    for direction, width in directionList:
        widthTotal += width
    
    for patient in patients:
        collectionExp = []
        for direction, width in directionList:
            figureFile = os.path.join(sliceFolder, "patient{}Exp{}.png".format(patient, direction))
            image = plt.imread(figureFile)
            if direction in ["Sagittal", "Coronal"]:
                image = np.flip(image, axis=0)
            image = TrimPadding(image, height, width)
            collectionExp.append(image)
        imageExp = np.concatenate(collectionExp, axis=1)
        imageExp = imageExp[:, :, :3].copy()
        imageExp = (imageExp * 255).astype(np.uint8)
        imageExp = Image.fromarray(imageExp)
        draw = ImageDraw.Draw(imageExp)
        text = "Patient{} Ours".format(patient)
        position = (10, 10)
        color = (255, 255, 255)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 90)
        draw.text(position, text, fill=color, font=font)
        imageExp = np.array(imageExp)

        collectionRef = []
        for direction, width in directionList:
            figureFile = os.path.join(sliceFolder, "patient{}Ref{}.png".format(patient, direction))
            image = plt.imread(figureFile)
            if direction in ["Sagittal", "Coronal"]:
                image = np.flip(image, axis=0)
            image = TrimPadding(image, height, width)
            collectionRef.append(image)
        imageRef = np.concatenate(collectionRef, axis=1)
        imageRef = imageRef[:, :, :3].copy()
        imageRef = (imageRef * 255).astype(np.uint8)
        imageRef = Image.fromarray(imageRef)
        draw = ImageDraw.Draw(imageRef)
        text = "Patient{} Baseline".format(patient)
        draw.text(position, text, fill=color, font=font)
        imageRef = np.array(imageRef)

        patientImage = np.concatenate((imageExp, imageRef), axis=0)
        figureFile = os.path.join(sliceFolder, "StackPatient{}.png".format(patient))
        plt.imsave(figureFile, patientImage)
        print(figureFile)


def TrimPadding(image, height, width):
    HeightOrg, WidthOrg, channels = image.shape
    if HeightOrg > height:
        IdxBegin = int((HeightOrg - height) / 2)
        image = image[IdxBegin:IdxBegin+height, :, :]
    elif HeightOrg < height:
        backgroundShape = (height, WidthOrg, channels)
        background = np.zeros(backgroundShape, dtype=np.float32)
        if channels == 4:
            background[:, :, 3] = 1.0
        IdxBegin = int((height - HeightOrg) / 2)
        background[IdxBegin: IdxBegin+HeightOrg, :, :] = image
        image = background
    
    if WidthOrg > width:
        IdxBegin = int((WidthOrg - width) / 2)
        image = image[:, IdxBegin: IdxBegin+width]
    elif WidthOrg < width:
        backgroundShape = (height, width, channels)
        background = np.zeros(backgroundShape, dtype=np.float32)
        if channels == 4:
            background[:, :, 3] = 1.0
        IdxBegin = int((width - WidthOrg) / 2)
        background[:, IdxBegin: IdxBegin+WidthOrg, :] = image
    return image


def colorBarGen():
    """
    This function generates the colorbar
    """
    image = np.random.rand(100, 100) * 75
    fig, ax = plt.subplots()

    # Set the face color of the figure and axis to black
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    cax = ax.imshow(image, cmap="jet", vmin=0, vmax=75)
    cbar = fig.colorbar(cax, ax=ax, orientation="vertical")

    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    cbar.ax.tick_params(axis="y", colors="white", labelsize=16)
    cbar.set_label("Dose (Gy)", color="white", fontsize=16)

    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(canvas.get_width_height()[::-1] + (3,))
    if False:
        figureFile = os.path.join(ManuFiguresFolder, "DoseWashSample", "colorbar.png")
        plt.imsave(figureFile, image)
        print(figureFile)
    
    ImageSliced = image[:, -160:-40, :]
    if False:
        figureFile = os.path.join(figureFolder, "DoseWashSample", "colorbar.png")
        plt.imsave(figureFile, ImageSliced)
        print(figureFile)

    return ImageSliced


def concatVertical():
    PatientsSelect = ["003", "132"]
    collection = []
    for patient in PatientsSelect:
        figureFile = os.path.join(figureFolder, "DoseWashSample", "StackPatient{}.png".format(patient))
        figure = plt.imread(figureFile)
        collection.append(figure)
    image = np.concatenate(collection, axis=0)
    image = image[:, :, :3]
    image = (255 * image).astype(np.uint8)

    ColorBar = colorBarGen()
    ColorBar = ColorBar / 255
    factor = 5
    shapeNew = (ColorBar.shape[0] * factor, ColorBar.shape[1] * factor, 3)
    ColorBar = transform.resize(ColorBar, shapeNew)
    ColorBar = (ColorBar * 255).astype(np.uint8)
    ColorBarHeight = ColorBar.shape[0]
    Margin = int((image.shape[0] - ColorBarHeight)/2)
    Canvas = np.zeros((image.shape[0], ColorBar.shape[1], 3), dtype=np.uint8)
    Canvas[Margin:Margin + ColorBarHeight, :, :] = ColorBar

    image = np.concatenate((image, Canvas), axis=1)
    globalImagePath = os.path.join(figureFolder, "DoseWashComp.png")
    plt.imsave(globalImagePath, image)
    print(globalImagePath)
    # globalImagePath = os.path.join(figureFolder, "DoseWashComp.eps")
    # plt.imsave(globalImagePath, image)
    # print(globalImagePath)
    # plt.clf()


if __name__ == "__main__":
    # StructsInit()
    # DVH_plot()
    # DrawDoseWash()
    # CoalesceFigures()
    concatVertical()