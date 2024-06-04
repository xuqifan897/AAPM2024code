import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import transform, measure
from PIL import Image, ImageDraw, ImageFont
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

resultFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect"
figureFolder = "/data/qifan/projects/AAPM2024/manufigures"
ManuFiguresFolder = "/data/qifan/projects/AAPM2024/manufigures"

StructureList = []
exclude = ["SKIN", "PTVMerge", "rind"]
Converge = {"BrainStem": ["BRAIN_STEM", "Brainstem", "BRAIN_STEM_PRV"],
            "SpinalCord": ["SPINAL_CORD", "SPINL_CRD_PRV"],
            "oralCavity": ["oralCavity", "ORAL_CAVITY", "OralCavity"]}
ConvergeReverse = {}
for name, collection in Converge.items():
    for child in collection:
        ConvergeReverse[child] = name

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
colorMap = {}

# z, y, x
SlicingIndices = [
    (85, 109, 115),
    (98, 107, 121),
    (90, 107, 121),
    (134, 84, 95),
    (83, 70, 105),
    (73, 105, 95),
    (81, 96, 108),
    (111, 83, 96)
]

def StructsInit():
    """
    This function is to generate a coherent structure list for all patients
    """
    global StructureList, colorMap
    patients = os.listdir(resultFolder)
    patients.sort()
    for patient in patients:
        patientFolder = os.path.join(resultFolder, patient)
        InputMaskFolder = os.path.join(patientFolder, "InputMask")
        structuresLocal = os.listdir(InputMaskFolder)
        structuresLocal = [a.split(".")[0] for a in structuresLocal]
        for a in structuresLocal:
            if a not in StructureList:
                StructureList.append(a)
    StructureList_copy = []
    for name in StructureList:
        if name in ConvergeReverse:
            name = ConvergeReverse[name]
        if name not in StructureList_copy and name not in exclude:
            StructureList_copy.append(name)

    # move "PTV63" forward
    subject = "PTV63"
    assert subject in StructureList_copy, "{} not found".format(subject)
    StructureList_copy.remove(subject)
    StructureList_copy.insert(3, subject)

    StructureList.clear()
    StructureList = StructureList_copy.copy()
    for i in range(len(StructureList)):
        colorMap[StructureList[i]] = colors[i]


def DVH_plot():
    patients = os.listdir(resultFolder)
    patients.sort()
    rowSize = 3
    colSize = int(np.ceil(len(patients) / rowSize))
    fig, axes = plt.subplots(colSize, rowSize, figsize=(15, 12))
    for i, patient in enumerate(patients):
        patientFolder = os.path.join(resultFolder, patient)
        dimensionFile = os.path.join(patientFolder,
            "FastDose", "prep_output", "dimension.txt")
        with open(dimensionFile, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension = np.flip(dimension)

        MaskFolder = os.path.join(patientFolder, "InputMask")
        structuresLocal = os.listdir(MaskFolder)
        structuresLocal = [a.split(".")[0] for a in structuresLocal]
        structuresLocal = [a for a in structuresLocal if a not in exclude]
        structuresLocalNorm = {}
        for a in structuresLocal:
            if a not in ConvergeReverse:
                structuresLocalNorm[a] = a
            else:
                structuresLocalNorm[a] = ConvergeReverse[a]
        maskDict = {}
        for a, b in structuresLocalNorm.items():
            file = os.path.join(MaskFolder, a + ".bin")
            mask = np.fromfile(file, dtype=np.uint8)
            mask = np.reshape(mask, dimension)
            maskDict[b] = mask

        # load FastDose result
        DoseExpFile = os.path.join(patientFolder, "FastDose", "plan1", "dose.bin")
        DoseExp = np.fromfile(DoseExpFile, dtype=np.float32)
        DoseExp = np.reshape(DoseExp, dimension)

        # load clinical result
        DoseRefFile = os.path.join(patientFolder, "doseExp1.npy")
        DoseRef = np.load(DoseRefFile)
        DoseRef = np.transpose(DoseRef, axes=(2, 0, 1))
        DoseRef = np.flip(DoseRef, axis=0)
        DoseRef = transform.resize(DoseRef, dimension)

        # normalize
        PrimaryStructureName = "PTV70"
        assert PrimaryStructureName in maskDict, "Structure {} not " \
            "found".format(PrimaryStructureName)
        PrimaryMask = maskDict[PrimaryStructureName]
        PrimaryMask = PrimaryMask.astype(bool)
        PrimaryExp = DoseExp[PrimaryMask]
        thresh = np.percentile(PrimaryExp, 5)
        DoseExp *= 70 / thresh
        PrimaryRef = DoseRef[PrimaryMask]
        thresh = np.percentile(PrimaryRef, 5)
        DoseRef *= 70 / thresh

        rowIdx = i % rowSize
        colIdx = i // rowSize
        assert colIdx < colSize, "Figure index ({}, {}) error".format(rowIdx, colIdx)
        ax = axes[rowIdx, colIdx]
        print(patient)
        for name, mask in maskDict.items():
            mask = mask.astype(bool)
            StructDoseExp = DoseExp[mask]
            StructDoseExp = np.sort(StructDoseExp)
            StructDoseExp = np.insert(StructDoseExp, 0, 0)

            StructDoseRef = DoseRef[mask]
            StructDoseRef = np.sort(StructDoseRef)
            StructDoseRef = np.insert(StructDoseRef, 0, 0)

            y_axis = np.linspace(100, 0, np.sum(mask)+1)
            ax.plot(StructDoseExp, y_axis, color=colorMap[name], linewidth=2.0)
            ax.plot(StructDoseRef, y_axis, color=colorMap[name], linewidth=2.0, linestyle="--")
            print(name)
        # ax.set_xlabel("Dose (Gy)", fontsize=16)
        # ax.set_ylabel("Fractional Volume (%)", fontsize=16)
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_title("Patient {}".format(patient), fontsize=16)
        print()
    fig.delaxes(axes[2, 2])

    # prepare legend
    legend_ax = fig.add_subplot(3, 3, 9)
    legend_ax.axis("off")
    handles = []
    labels = []
    for name, color in colorMap.items():
        handleEntry = plt.Line2D([0], [0], color=color, lw=2)
        handles.append(handleEntry)
        labels.append(name)
    legend_ax.legend(handles, labels, loc="center", ncol=2, fontsize=15)

    # prepare global xlabel and ylabel
    fig.subplots_adjust(left=0.2, right=1.0, top=1.0, bottom=0.5)
    fig.text(0.5, 0.01, "Dose (Gy)", ha='center', va='center', fontsize=16)
    fig.text(0.01, 0.5, "Fractional Volume (%)", ha='center', va='center',
        rotation='vertical', fontsize=16)

    plt.tight_layout()
    figureFile = os.path.join(ManuFiguresFolder, "FastDoseVSClinical.png")
    plt.savefig(figureFile)
    figureFile = os.path.join(ManuFiguresFolder, "FastDoseVSClinical.eps")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()
    print(figureFile, "\n")


def DrawDoseWash():
    sliceFolder = os.path.join(ManuFiguresFolder, "DoseWashSample")
    if not os.path.isdir(sliceFolder):
        os.mkdir(sliceFolder)
    patients = os.listdir(resultFolder)
    patients.sort()
    halfCropSize = 50
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

        MaskFolder = os.path.join(patientFolder, "InputMask")
        structuresLocal = os.listdir(MaskFolder)
        structuresLocal = [a.split(".")[0] for a in structuresLocal]
        BodyName = "SKIN"
        assert BodyName in structuresLocal, "No bounding box found"
        structuresLocal = [a for a in structuresLocal if a not in exclude]
        structuresLocal.append(BodyName)
        structuresLocalNorm = {}
        for a in structuresLocal:
            if a not in ConvergeReverse:
                structuresLocalNorm[a] = a
            else:
                structuresLocalNorm[a] = ConvergeReverse[a]
        maskDict = {}
        for a, b in structuresLocalNorm.items():
            file = os.path.join(MaskFolder, a + ".bin")
            mask = np.fromfile(file, dtype=np.uint8)
            mask = np.reshape(mask, dimension)
            maskDict[b] = mask
        BodyMask = maskDict[BodyName].astype(bool)
        del maskDict[BodyName]

        # load FastDose result
        DoseExpFile = os.path.join(patientFolder, "FastDose", "plan1", "dose.bin")
        DoseExp = np.fromfile(DoseExpFile, dtype=np.float32)
        DoseExp = np.reshape(DoseExp, dimension)
        DoseExp[np.logical_not(BodyMask)] = 0

        # load clinical result
        DoseRefFile = os.path.join(patientFolder, "doseExp1.npy")
        DoseRef = np.load(DoseRefFile)
        DoseRef = np.transpose(DoseRef, axes=(2, 0, 1))
        DoseRef = np.flip(DoseRef, axis=0)
        DoseRef = transform.resize(DoseRef, dimension)
        DoseRef[np.logical_not(BodyMask)] = 0

        # normalize
        PrimaryStructureName = "PTV70"
        assert PrimaryStructureName in maskDict, "Structure {} not " \
            "found".format(PrimaryStructureName)
        PrimaryMask = maskDict[PrimaryStructureName]
        PrimaryMask = PrimaryMask.astype(bool)
        PrimaryExp = DoseExp[PrimaryMask]
        thresh = np.percentile(PrimaryExp, 5)
        DoseExp *= 70 / thresh
        PrimaryRef = DoseRef[PrimaryMask]
        thresh = np.percentile(PrimaryRef, 5)
        DoseRef *= 70 / thresh

        CenterIdxZ = SlicingIndices[i][0]
        CenterIdxY =  SlicingIndices[i][1]
        CenterIdxX =  SlicingIndices[i][2]

        BodyMaskSlice = BodyMask[CenterIdxZ, :, :]
        BodyMaskSliceCentroid = CalcCentroid(BodyMaskSlice)
        print(BodyMaskSliceCentroid)
        
        # draw axial slice
        densitySlice = densityArray[CenterIdxZ, :, :]
        densitySlice = densitySlice[BodyMaskSliceCentroid[0] - halfCropSize: BodyMaskSliceCentroid[0] + halfCropSize,
            BodyMaskSliceCentroid[1] - halfCropSize: BodyMaskSliceCentroid[1] + halfCropSize]
        plt.imshow(densitySlice, cmap="gray", vmin=500, vmax=1500)
        for name, maskArray in maskDict.items():
            color = colorMap[name]
            maskSlice = maskArray[CenterIdxZ, :, :]
            maskSlice = maskSlice[BodyMaskSliceCentroid[0] - halfCropSize: BodyMaskSliceCentroid[0] + halfCropSize,
                BodyMaskSliceCentroid[1] - halfCropSize: BodyMaskSliceCentroid[1] + halfCropSize]
            if np.sum(maskSlice) == 0:
                continue
            contours = measure.find_contours(maskSlice)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1)

        DoseSliceExp = DoseExp[CenterIdxZ, :, :]
        DoseSliceExp = DoseSliceExp[BodyMaskSliceCentroid[0] - halfCropSize: BodyMaskSliceCentroid[0] + halfCropSize,
                BodyMaskSliceCentroid[1] - halfCropSize: BodyMaskSliceCentroid[1] + halfCropSize]
        plt.imshow(DoseSliceExp, cmap="jet", vmin=0, vmax=75, alpha=0.3)
        figureFile = os.path.join(sliceFolder, "Patient{}Axial.png".format(patient))
        plt.savefig(figureFile)
        plt.clf()
        print(figureFile)


def CalcCentroid(binarySlice):
    height, width = binarySlice.shape
    Coords0 = np.arange(height)
    Coords0 = np.expand_dims(Coords0, axis=1)
    Centroid0 = np.sum(Coords0 * binarySlice) / np.sum(binarySlice)

    Coords1 = np.arange(width)
    Coords1 = np.expand_dims(Coords1, axis=0)
    Centroid1 = np.sum(Coords1 * binarySlice) / np.sum(binarySlice)
    
    return (int(Centroid0), int(Centroid1))


def ViewCoronalSagittal():
    """
    This function prints out the coronal and sagittal views
    of all the 8 head-neck patients
    """
    patients = os.listdir(resultFolder)
    for patient in patients:
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

        MaskFolder = os.path.join(patientFolder, "InputMask")
        structuresLocal = os.listdir(MaskFolder)
        structuresLocal = [a.split(".")[0] for a in structuresLocal]
        BodyName = "SKIN"
        assert BodyName in structuresLocal, "No bounding box found"
        structuresLocal = [a for a in structuresLocal if a not in exclude]
        structuresLocal.append(BodyName)
        structuresLocalNorm = {}
        for a in structuresLocal:
            if a not in ConvergeReverse:
                structuresLocalNorm[a] = a
            else:
                structuresLocalNorm[a] = ConvergeReverse[a]
        maskDict = {}
        for a, b in structuresLocalNorm.items():
            file = os.path.join(MaskFolder, a + ".bin")
            mask = np.fromfile(file, dtype=np.uint8)
            mask = np.reshape(mask, dimension)
            maskDict[b] = mask
        BodyMask = maskDict[BodyName].astype(bool)
        del maskDict[BodyName]
        
        # load FastDose result
        DoseExpFile = os.path.join(patientFolder, "FastDose", "plan1", "dose.bin")
        DoseExp = np.fromfile(DoseExpFile, dtype=np.float32)
        DoseExp = np.reshape(DoseExp, dimension)
        DoseExp[np.logical_not(BodyMask)] = 0

        # normalize
        PrimaryStructureName = "PTV70"
        assert PrimaryStructureName in maskDict, "Structure {} not " \
            "found".format(PrimaryStructureName)
        PrimaryMask = maskDict[PrimaryStructureName]
        PrimaryMask = PrimaryMask.astype(bool)
        PrimaryExp = DoseExp[PrimaryMask]
        thresh = np.percentile(PrimaryExp, 5)
        DoseExp *= 70 / thresh

        if False:
            # print coronal slices
            coronalViewFolder = os.path.join(patientFolder, "FastDose", "plan1", "CoronalView")
            if not os.path.isdir(coronalViewFolder):
                os.mkdir(coronalViewFolder)
            coronalDim = densityArray.shape[1]
            for j in range(coronalDim):
                densitySlice = densityArray[:, j, :]
                doseSlice = DoseExp[:, j, :]
                plt.imshow(densitySlice, cmap="gray", vmin=500, vmax=1500)
                for name, maskArray in maskDict.items():
                    color = colorMap[name]
                    maskSlice = maskArray[:, j, :]
                    if np.sum(maskSlice) == 0:
                        continue
                    contours = measure.find_contours(maskSlice)
                    for contour in contours:
                        plt.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1)
                plt.imshow(doseSlice, cmap="jet", vmin=0, vmax=75, alpha=0.3)
                figureFile = os.path.join(coronalViewFolder, "{:03d}.png".format(j))
                plt.savefig(figureFile)
                plt.clf()
                print(figureFile)
        
        if True:
            # print sagittal slices
            sagittalViewFolder = os.path.join(patientFolder, "FastDose", "plan1", "SagittalView")
            if not os.path.isdir(sagittalViewFolder):
                os.mkdir(sagittalViewFolder)
            sagittalDim = densityArray.shape[2]
            for j in range(sagittalDim):
                densitySlice = densityArray[:, :, j]
                doseSlice = DoseExp[:, :, j]
                plt.imshow(densitySlice, cmap="gray", vmin=500, vmax=1500)
                for name, maskArray in maskDict.items():
                    color = colorMap[name]
                    maskSlice = maskArray[:, :, j]
                    if np.sum(maskSlice) == 0:
                        continue
                    contours = measure.find_contours(maskSlice)
                    for contour in contours:
                        plt.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1)
                plt.imshow(doseSlice, cmap="jet", vmin=0, vmax=75, alpha=0.3)
                figureFile = os.path.join(sagittalViewFolder, "{:03d}.png".format(j))
                plt.savefig(figureFile)
                plt.clf()
                print(figureFile)


def DoseWashComb():
    """
    This function draws the combination of different views of the dose wash
    """
    # group = "Exp"
    group = "Ref"
    sliceFolder = os.path.join(ManuFiguresFolder, "DoseWashSample")
    if not os.path.isdir(sliceFolder):
        os.mkdir(sliceFolder)
    patients = os.listdir(resultFolder)
    patients.sort()
    # # z, y, x
    # CropSizeList = [(200, 100), (200, 200), (200, 128)]
    # figureWidth = CropSizeList[0][1] + CropSizeList[1][1] + CropSizeList[2][1]
    DoseWashSampleFolder = os.path.join(ManuFiguresFolder, "DoseWashSample")
    if not os.path.isdir(DoseWashSampleFolder):
        os.mkdir(DoseWashSampleFolder)
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

        MaskFolder = os.path.join(patientFolder, "InputMask")
        structuresLocal = os.listdir(MaskFolder)
        structuresLocal = [a.split(".")[0] for a in structuresLocal]
        structuresLocal = [a for a in structuresLocal if a not in exclude]
        structuresLocalNorm = {}
        for a in structuresLocal:
            if a not in ConvergeReverse:
                structuresLocalNorm[a] = a
            else:
                structuresLocalNorm[a] = ConvergeReverse[a]
        maskDict = {}
        for a, b in structuresLocalNorm.items():
            file = os.path.join(MaskFolder, a + ".bin")
            mask = np.fromfile(file, dtype=np.uint8)
            mask = np.reshape(mask, dimension)
            maskDict[b] = mask
        
        # load FastDose result
        if group == "Exp":
            DoseFile = os.path.join(patientFolder, "FastDose", "plan1", "dose.bin")
            Dose = np.fromfile(DoseFile, dtype=np.float32)
            Dose = np.reshape(Dose, dimension)
        elif group == "Ref":
            DoseFile = os.path.join(patientFolder, "doseExp1.npy")
            Dose = np.load(DoseFile)
            Dose = np.transpose(Dose, axes=(2, 0, 1))
            Dose = np.flip(Dose, axis=0)
            Dose = transform.resize(Dose, dimension)

        # normalize
        PrimaryStructureName = "PTV70"
        assert PrimaryStructureName in maskDict, "Structure {} not " \
            "found".format(PrimaryStructureName)
        PrimaryMask = maskDict[PrimaryStructureName]
        PrimaryMask = PrimaryMask.astype(bool)
        PrimaryExp = Dose[PrimaryMask]
        thresh = np.percentile(PrimaryExp, 5)
        Dose *= 70 / thresh
        Dose[densityArray < 500] = 0

        sliceIdxZ, SliceIdxY, SliceIdxX = SlicingIndices[i]

        # Print axial slice
        AxialDensitySlice = densityArray[sliceIdxZ, :, :]
        AxialDoseSlice = Dose[sliceIdxZ, :, :].copy()
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.imshow(AxialDensitySlice, cmap="gray", vmin=500, vmax=1500)
        for name, maskArray in maskDict.items():
            if name == "brain":
                continue  # skip the self-created structures
            color = colorMap[name]
            maskSlice = maskArray[sliceIdxZ, :, :]
            if np.sum(maskSlice) == 0:
                continue
            contours = measure.find_contours(maskSlice)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1)
        alpha = 0.3 * (AxialDoseSlice > 5)
        ax.imshow(AxialDoseSlice, cmap="jet", vmin=0, vmax=75, alpha=alpha)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        figureHeight = 5 * AxialDensitySlice.shape[0] / 200
        figureWidth = 5 * AxialDensitySlice.shape[1] / 200
        fig.set_figwidth(figureHeight)
        fig.set_figheight(figureWidth)
        figureFile = os.path.join(sliceFolder, "Patient{}Axial{}.png".format(patient, group))
        plt.savefig(figureFile)
        plt.close(fig)
        plt.clf()
        print(figureFile)

        # Print coronal slice
        CoronalDensitySlice = densityArray[:, SliceIdxY, :]
        CoronalDoseSlice = Dose[:, SliceIdxY, :].copy()
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.imshow(CoronalDensitySlice, cmap="gray", vmin=500, vmax=1500)
        for name, maskArray in maskDict.items():
            if name == "brain":
                continue
            color = colorMap[name]
            maskSlice = maskArray[:, SliceIdxY, :]
            if np.sum(maskSlice) == 0:
                continue
            contours = measure.find_contours(maskSlice)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color=color)
        alpha = 0.3 * (CoronalDoseSlice > 5)
        ax.imshow(CoronalDoseSlice, cmap="jet", vmin=0, vmax=75, alpha=alpha)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        figureHeight = 5 * CoronalDensitySlice.shape[0] / 200
        figureWidth = 5 * CoronalDensitySlice.shape[1] / 200
        fig.set_figheight(figureHeight)
        fig.set_figwidth(figureWidth)
        figureFile = os.path.join(sliceFolder, "Patient{}Coronal{}.png".format(patient, group))
        plt.savefig(figureFile)
        plt.close(fig)
        plt.clf()
        print(figureFile)

        # Print sagittal slices
        SagittalDensitySlice = densityArray[:, :, SliceIdxX]
        SagittalDoseSlice = Dose[:, :, SliceIdxX].copy()
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.imshow(SagittalDensitySlice, cmap="gray", vmin=500, vmax=1500)
        for name, maskArray in maskDict.items():
            if name == "brain":
                continue
            color = colorMap[name]
            maskSlice = maskArray[:, :, SliceIdxX]
            if np.sum(maskSlice) == 0:
                continue
            contours = measure.find_contours(maskSlice)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], color=color)
        alpha = 0.3 * (SagittalDoseSlice > 5)
        ax.imshow(SagittalDoseSlice, cmap="jet", vmin=0, vmax=75, alpha=alpha)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        figureHeight = 5 * SagittalDensitySlice.shape[0] / 200
        figureWidth = 5 * SagittalDensitySlice.shape[1] / 200
        fig.set_figheight(figureHeight)
        fig.set_figwidth(figureWidth)
        figureFile = os.path.join(sliceFolder, "Patient{}Sagittal{}.png".format(patient, group))
        plt.savefig(figureFile)
        plt.close(fig)
        plt.clf()
        print(figureFile)


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
        figureFile = os.path.join(ManuFiguresFolder, "DoseWashSample", "colorbar.png")
        plt.imsave(figureFile, ImageSliced)
        print(figureFile)

    return ImageSliced



def CoalesceFigures():
    """
    This function groups multiple figures into one figure
    """
    # patients = [3, 9, 13, 27, 57, 132, 251, 479]
    patients = [3, 13]
    patients = ["{:03d}".format(a) for a in patients]
    CorpusFolder = os.path.join(ManuFiguresFolder, "DoseWashSample")
    directionList = [("Axial", 300), ("Sagittal", 350), ("Coronal", 450)]
    widthTotal = 0
    for direction, width in directionList:
        widthTotal += width

    figureOutputFolder = os.path.join(ManuFiguresFolder, "DoseWashSample")
    if not os.path.isdir(figureOutputFolder):
        os.mkdir(figureOutputFolder)
    height = 400
    stacking = "vertical"
    # stacking = "horizontal"
    PatientCollection = []
    for patient in patients:
        collectionExp = []
        for direction, width in directionList:
            figureFile = os.path.join(CorpusFolder, "Patient{}{}Exp.png".format(patient, direction))
            image = plt.imread(figureFile)
            if direction in ["Sagittal", "Coronal"]:
                image = np.flip(image, axis=0)
            # print(patient, direction, image.shape)
            image = TrimPadding(image, height, width)
            collectionExp.append(image)
        imageExp = np.concatenate(collectionExp, axis=1)
        # convert to 3 channels to comply with PIL convention
        imageExp = imageExp[:, :, :3].copy()
        imageExp = (imageExp * 255).astype(np.uint8)
        imageExp = Image.fromarray(imageExp)
        draw = ImageDraw.Draw(imageExp)
        text = "Patient{} Ours".format(patient)
        position = (10, 10)
        color = (255, 255, 255)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        draw.text(position, text, fill=color, font=font)
        imageExp = np.array(imageExp)
        
        collectionRef = []
        for direction, width in directionList:
            figureFile = os.path.join(CorpusFolder, "Patient{}{}Ref.png".format(patient, direction))
            image = plt.imread(figureFile)
            if direction in ["Sagittal", "Coronal"]:
                image = np.flip(image, axis=0)
            image = TrimPadding(image, height, width)
            collectionRef.append(image)
        imageRef = np.concatenate(collectionRef, axis=1)
        # convert to 3 channels to comply with PIL convention
        imageRef = imageRef[:, :, :3].copy()
        imageRef = (imageRef * 255).astype(np.uint8)
        imageRef = Image.fromarray(imageRef)
        draw = ImageDraw.Draw(imageRef)
        text = "Patient{} Baseline".format(patient)
        draw.text(position, text, fill=color, font=font)
        imageRef = np.array(imageRef)

        if stacking == "horizontal":
            patientImage = np.concatenate((imageExp, imageRef), axis=1)
        elif stacking == "vertical":
            patientImage = np.concatenate((imageExp, imageRef), axis=0)
        
        if True:
            PatientCollection.append(patientImage)
        else:
            figureFile = os.path.join(CorpusFolder, "StackPatient{}.png".format(patient))
            plt.imsave(figureFile, patientImage)
            print(figureFile)
    globalImage = np.concatenate(PatientCollection, axis=0)

    ColorBar = colorBarGen()
    ColorBar = ColorBar.astype(float) / 255
    # resize ColorBar to appropriate size
    ColorBarHeight = ColorBar.shape[0]
    ColorBarWidth = ColorBar.shape[1]
    ColorBarWidthNew = 160
    ColorBarHeightNew = int(ColorBarHeight * ColorBarWidthNew / ColorBarWidth)
    ColorBar = transform.resize(ColorBar, (ColorBarHeightNew, ColorBarWidthNew))
    ColorBar = (ColorBar * 255).astype(np.uint8)
    # padding
    background = np.zeros((globalImage.shape[0], ColorBar.shape[1], 3), dtype=np.uint8)
    IdxBegin = int((globalImage.shape[0] - ColorBar.shape[0]) / 2)
    background[IdxBegin: IdxBegin + ColorBar.shape[0], :, :] = ColorBar
    ColorBar = background
    if False:
        ColorBarFile = os.path.join(ManuFiguresFolder, "DoseWashSample", "colorbar.png")
        plt.imsave(ColorBarFile, ColorBar)
        print(ColorBarFile)
        return
    globalImage = np.concatenate((globalImage, ColorBar), axis=1)

    globalImagePath = os.path.join(ManuFiguresFolder, "DoseWashComp.png")
    plt.imsave(globalImagePath, globalImage)
    print(globalImagePath)
    globalImagePath = os.path.join(ManuFiguresFolder, "DoseWashComp.eps")
    plt.imsave(globalImagePath, globalImage)
    print(globalImagePath)
    plt.clf()


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
        IdxBegin = int((WidthOrg - width) / 2)
        background[:, IdxBegin: IdxBegin+WidthOrg, :] = image
    return image


if __name__ == "__main__":
    StructsInit()
    # DVH_plot()
    # DrawDoseWash()
    # ViewCoronalSagittal()
    # DoseWashComb()
    # colorBarGen()
    CoalesceFigures()