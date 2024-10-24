import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import nrrd
from collections import OrderedDict
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage import measure, transform
from io import BytesIO
import h5py

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
sourceFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB"
numPatients = 5
isoRes = 2.5
figureFolder = "/data/qifan/projects/AAPM2024/manufigures/PancreasSIBReplan"
if not os.path.isdir(figureFolder):
    os.mkdir(figureFolder)

def DVH_comp():
    nRows = 4
    nCols = 3
    fig = plt.figure(figsize=(12, 15))
    gs = gridspec.GridSpec(nRows, nCols, height_ratios=[4, 4, 4, 0.2],
        width_ratios=[0.2, 4, 4])
    
    # create the common y label
    ylabel_block = fig.add_subplot(gs[:-1, 0])
    ylabel_block.text(0.9, 0.5, "Fractional Volume (%)", ha="center", va="center",
        rotation="vertical", fontsize=20)
    ylabel_block.axis("off")

    # create the common x label
    xlabel_block = fig.add_subplot(gs[-1, 1:])
    xlabel_block.text(0.5, 0.5, "Dose (Gy)", ha="center", va="center", fontsize=20)
    xlabel_block.axis("off")

    relevantStructures = ["PTV", "Stomach_duo_planCT", "Bowel_sm_planCT", "kidney_left", "kidney_right", "liver"]
    colorMap = {}
    for i, struct in enumerate(relevantStructures):
        colorMap[struct] = colors[i]
    patientsPerRow = 2
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)

        clinicalDose = os.path.join(patientFolder, "doseNorm.bin")
        clinicalDose = np.fromfile(clinicalDose, dtype=np.float32)
        physicalDose = os.path.join(patientFolder, "dosePhysical.bin")
        physicalDose = np.fromfile(physicalDose, dtype=np.float32)
        FastDoseDose = os.path.join(patientFolder, "FastDose", "plan2", "dose.bin")
        FastDoseDose = np.fromfile(FastDoseDose, dtype=np.float32)
        QihuiRyanDose = os.path.join(patientFolder, "QihuiRyan", "doseQihuiRyan_else5.bin")
        if i == 1:
            QihuiRyanDose = os.path.join(patientFolder, "QihuiRyan", "doseQihuiRyan_else5_PTV50.bin")
        QihuiRyanDose = np.fromfile(QihuiRyanDose, dtype=np.float32)

        # scale to physical dose
        scaleFactor = np.max(physicalDose) / np.max(clinicalDose)
        clinicalDose *= scaleFactor
        assert np.max(np.abs(clinicalDose - physicalDose)) < 1e-4
        FastDoseDose *= scaleFactor
        QihuiRyanDose *= scaleFactor

        masks = {}
        for name in relevantStructures:
            filename = name
            if name == "PTV":
                filename = "ROI"
            filename = os.path.join(patientFolder, "InputMask", filename+".bin")
            maskArray = np.fromfile(filename, dtype=np.uint8).astype(bool)
            masks[name] = maskArray
        
        rowIdx = i // patientsPerRow
        colIdx = 1 + i % patientsPerRow
        block = fig.add_subplot(gs[rowIdx, colIdx])
        for name, maskArray in masks.items():
            color = colorMap[name]
            clinicalStructDose = np.sort(clinicalDose[maskArray])
            clinicalStructDose = np.insert(clinicalStructDose, 0, 0.0)
            FastDoseStructDose = np.sort(FastDoseDose[maskArray])
            FastDoseStructDose = np.insert(FastDoseStructDose, 0, 0.0)
            QihuiRyanStructDose = np.sort(QihuiRyanDose[maskArray])
            QihuiRyanStructDose = np.insert(QihuiRyanStructDose, 0, 0.0)
            assert (nPoints:=clinicalStructDose.size) == FastDoseStructDose.size \
                and nPoints == QihuiRyanStructDose.size
            yAxis = (1 - np.arange(nPoints) / (nPoints - 1)) * 100
            block.plot(FastDoseStructDose, yAxis, color=color, linewidth=3)
            block.plot(QihuiRyanStructDose, yAxis, color=color, linewidth=3, linestyle="--")
            block.plot(clinicalStructDose, yAxis, color=color, linewidth=1)
    
        block.tick_params(axis="x", labelsize=16)
        block.tick_params(axis="y", labelsize=16)
        block.set_title("Patient {:03d}".format(i+1), fontsize=20)
        print(patientName)
    
    legendBlock = fig.add_subplot(gs[nRows-2, nCols-1])
    legendBlock.axis("off")
    handles = []
    labels = []
    for name, color in colorMap.items():
        handleEntry = plt.Line2D([0], [0], color=color, lw=2)
        handles.append(handleEntry)
        labels.append(name)
    legendBlock.legend(handles, labels, loc="center", ncols=1, fontsize=16)
    plt.tight_layout()

    figureFile = os.path.join(figureFolder, "PancreasSIBDVH.png")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()

def R50Show():
    structs_exclude_DVH = ["RingStructure", "BODY", "ELSE"]
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)
        dimension = os.path.join(patientFolder, "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension_flip = np.flip(dimension)

        density = os.path.join(patientFolder, "density_raw.bin")
        density = np.fromfile(density, dtype=np.uint16)
        density = np.reshape(density, dimension_flip)
        density = np.flip(density, axis=0)

        # prescriptionLevel = 5  # D95
        prescriptionLevel = 10  # D90

        doseRef = os.path.join(patientFolder, "doseNorm.bin")
        doseRef = np.fromfile(doseRef, dtype=np.float32)
        doseRef = np.reshape(doseRef, dimension_flip)
        doseRef = np.flip(doseRef, axis=0)

        doseUHPP = os.path.join(patientFolder, "FastDose", "plan2", "dose.bin")
        doseUHPP = np.fromfile(doseUHPP, dtype=np.float32)
        doseUHPP = np.reshape(doseUHPP, dimension_flip)
        doseUHPP = np.flip(doseUHPP, axis=0)

        doseSOTA = os.path.join(patientFolder, "QihuiRyan", "doseQihuiRyan_else5.bin")
        if i == 1:
            doseSOTA = os.path.join(patientFolder, "QihuiRyan", "doseQihuiRyan_else5_PTV50.bin")
        doseSOTA = np.fromfile(doseSOTA, dtype=np.float32)
        doseSOTA = np.reshape(doseSOTA, dimension_flip)
        doseSOTA = np.flip(doseSOTA, axis=0)

        dosePhysical = os.path.join(patientFolder, "dosePhysical.bin")
        dosePhysical = np.fromfile(dosePhysical, dtype=np.float32)
        dosePhysical = np.reshape(dosePhysical, dimension_flip)
        dosePhysical = np.flip(dosePhysical, axis=0)

        # normalize
        factor = np.max(dosePhysical) / np.max(doseRef)
        doseRef *= factor
        assert np.max(np.abs(dosePhysical - doseRef)) < 1e-4
        doseUHPP *= factor
        doseSOTA *= factor

        FastDoseROIFile = os.path.join(patientFolder, "FastDose", "prep_output_else", "roi_list.h5")
        FastDoseROIDict = getStructures(FastDoseROIFile)
        FastDoseROIDict = FastDoseDictRename(FastDoseROIDict)
        FastDoseROIDict = {a: np.flip(b, axis=0) for a, b in FastDoseROIDict.items()}
        StructureList = os.path.join(patientFolder, "FastDose", "StructureInfo_else.csv")
        with open(StructureList, "r") as f:
            StructureList = f.readlines()
        StructureList = StructureList[1: -1]  # remove the title and the auxiliary structure
        StructureList = [b for a in StructureList if (b:=a.split(",")[0]) not in structs_exclude_DVH]
        StructureList = [a if a != "ROI" else "PTV" for a in StructureList]
        for a in StructureList:
            assert a in FastDoseROIDict

        # normalize
        PTVMask = FastDoseROIDict["PTV"] > 0
        BODYMask_not = np.logical_not(FastDoseROIDict["BODY"] > 0)

        doseList = [doseRef, doseUHPP, doseSOTA]
        for j in range(len(doseList)):
            doseList[j][BODYMask_not] = 0.0

        # draw doseWash
        doseShowMax = max(np.max(doseRef), np.max(doseUHPP), np.max(doseSOTA))
        PTVMask = FastDoseROIDict["PTV"]
        z, y, x = calcCentroid(PTVMask).astype(int)
        fig= plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1, 0.05], height_ratios=[1, 1, 1])
        subPlotList = []
        titleList = ["Clinical", "UHPP", "SOTA"]
        for j in range(3):
            currentDose = doseList[j]
            currentDosePTV = currentDose[PTVMask]
            prescriptionDose = np.percentile(currentDosePTV, prescriptionLevel)
            R50Level = prescriptionDose * 0.5
            doseCoronal = currentDose[:, y, :]
            densityCoronal = density[:, y, :]
            block = fig.add_subplot(gs[j, 0])
            subPlotList.append(block)
            block.imshow(densityCoronal, cmap="gray", vmin=0, vmax=1600)
            c1 = block.imshow(doseCoronal, cmap="jet", vmin=0, vmax=doseShowMax, alpha=0.3*(doseCoronal>R50Level))
            for k, struct in enumerate(StructureList):
                structMask = FastDoseROIDict[struct]
                structMaskSlice = structMask[:, y, :]
                contours = measure.find_contours(structMaskSlice)
                for contour in contours:
                    block.plot(contour[:, 1], contour[:, 0], color=colors[k], linewidth=1)
            block.text(0, 0, titleList[j], ha="left", va="top", color="white")
            
            doseSagittal = currentDose[:, :, x]
            densitySagittal = density[:, :, x]
            block = fig.add_subplot(gs[j, 1])
            subPlotList.append(block)
            block.imshow(densitySagittal, cmap="gray", vmin=0, vmax=1600)
            block.imshow(doseSagittal, cmap="jet", vmin=0, vmax=doseShowMax, alpha=0.3*(doseSagittal>R50Level))
            for k, struct in enumerate(StructureList):
                structMask = FastDoseROIDict[struct]
                structMaskSlice = structMask[:, :, x]
                contours = measure.find_contours(structMaskSlice)
                for contour in contours:
                    block.plot(contour[:, 1], contour[:, 0], color=colors[k], linewidth=1)
            
            doseAxial = currentDose[z, :, :]
            densityAxial = density[z, :, :]
            block = fig.add_subplot(gs[j, 2])
            subPlotList.append(block)
            block.imshow(densityAxial, cmap="gray", vmin=0, vmax=1600)
            block.imshow(doseAxial, cmap="jet", vmin=0, vmax=doseShowMax, alpha=0.3*(doseAxial>R50Level))
            for k, struct in enumerate(StructureList):
                structMask = FastDoseROIDict[struct]
                structMaskSlice = structMask[z, :, :]
                contours = measure.find_contours(structMaskSlice)
                for contour in contours:
                    block.plot(contour[:, 1], contour[:, 0], color=colors[k], linewidth=1)
        colorBarBlock = fig.add_subplot(gs[1, -1])
        fig.colorbar(c1, cax=colorBarBlock, orientation="vertical")
        fig.tight_layout()
        doseWashFile = os.path.join(figureFolder, "doseColorWash{}.png".format(patientName))
        plt.savefig(doseWashFile)
        plt.close(fig)
        plt.clf()
        print(doseWashFile)


def calcCentroid(mask):
    mask = mask > 0
    nVoxels = np.sum(mask)
    shape = mask.shape

    xWeight = np.arange(shape[0])
    xWeight = np.expand_dims(xWeight, axis=(1, 2))
    xCoord = np.sum(mask * xWeight) / nVoxels

    yWeight = np.arange(shape[1])
    yWeight = np.expand_dims(yWeight, axis=(0, 2))
    yCoord = np.sum(mask * yWeight) / nVoxels

    zWeight = np.arange(shape[2])
    zWeight = np.expand_dims(zWeight, axis=(0, 1))
    zCoord = np.sum(mask * zWeight) / nVoxels

    result = np.array((xCoord, yCoord, zCoord))
    return result


def FastDoseDictRename(FastDoseROIDict):
    subsMap = {"SKIN": "BODY", "ROI": "PTV"}
    result = {}
    for key, value in FastDoseROIDict.items():
        if key in subsMap:
            key = subsMap[key]
        result[key] = value
    return result


def getStructures(maskFile:str):
    dataset = h5py.File(maskFile, "r")
    structures_names = list(dataset.keys())
    result = {}
    for struct_name in structures_names:
        struct = dataset[struct_name]
        structProps = struct["ArrayProps"]
        structMask = struct["mask"]

        structSize = structProps.attrs["size"]
        structCropSize = structProps.attrs["crop_size"]
        structCropStart = structProps.attrs["crop_start"]

        structSize = np.flip(structSize, axis=0)
        structCropSize = np.flip(structCropSize, axis=0)
        structCropStart = np.flip(structCropStart, axis=0)

        structMask = np.array(structMask)
        structMask = np.reshape(structMask, structCropSize)
        struct_mask = np.zeros(structSize, dtype=bool)
        struct_mask[structCropStart[0]: structCropStart[0] + structCropSize[0],
            structCropStart[1]: structCropStart[1] + structCropSize[1],
            structCropStart[2]: structCropStart[2] + structCropSize[2]] = structMask
        result[struct_name] = struct_mask
    return result


def R50Calc():
    factor = 0.5
    prescriptionLevel = 10
    
    content = [
        "PTV dose: {}%".format(100 - prescriptionLevel),
        "R{} comparison".format(int(factor * 100)),
        "| Patient | Clinical | UHPP | SOTA |", "| - | - | - | - |"]
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)
        dimension = os.path.join(patientFolder, "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension_flip = np.flip(dimension)
        
        doseRef = os.path.join(patientFolder, "doseNorm.bin")
        doseRef = np.fromfile(doseRef, dtype=np.float32)
        doseRef = np.reshape(doseRef, dimension_flip)

        dosePhysical = os.path.join(patientFolder, "dosePhysical.bin")
        dosePhysical = np.fromfile(dosePhysical, dtype=np.float32)
        dosePhysical = np.reshape(dosePhysical, dimension_flip)

        doseUHPP = os.path.join(patientFolder, "FastDose", "plan2", "dose.bin")
        doseUHPP = np.fromfile(doseUHPP, dtype=np.float32)
        doseUHPP = np.reshape(doseUHPP, dimension_flip)

        doseSOTA = os.path.join(patientFolder, "QihuiRyan", "doseQihuiRyan_else5.bin")
        if i == 1:
            doseSOTA = os.path.join(patientFolder, "QihuiRyan", "doseQihuiRyan_else5_PTV50.bin")
        doseSOTA = np.fromfile(doseSOTA, dtype=np.float32)
        doseSOTA = np.reshape(doseSOTA, dimension_flip)

        # normalize
        scaleFactor = np.max(dosePhysical) / np.max(doseRef)
        doseRef *= scaleFactor
        # assert np.max(np.abs(dosePhysical - doseRef)) < 1e-4
        doseUHPP *= scaleFactor
        doseSOTA *= scaleFactor

        bodyMask = os.path.join(patientFolder, "InputMask", "SKIN.bin")
        bodyMask = np.fromfile(bodyMask, dtype=np.uint8)
        bodyMask = np.reshape(bodyMask, dimension_flip) > 0
        notBody = np.logical_not(bodyMask)

        PTVMask = os.path.join(patientFolder, "InputMask", "ROI.bin")
        PTVMask = np.fromfile(PTVMask, dtype=np.uint8)
        PTVMask = np.reshape(PTVMask, dimension_flip) > 0

        density = os.path.join(patientFolder, "density_raw.bin")
        density = np.fromfile(density, dtype=np.uint16)
        density = np.reshape(density, dimension_flip)

        doseList = [doseRef, doseUHPP, doseSOTA]
        for doseArray in doseList:
            doseArray[notBody] = 0.0
        ptvVoxels = np.sum(PTVMask)
        RscoreList = []
        R50ThreshList = []
        for doseArray in doseList:
            doseArrayPTV = doseArray[PTVMask]
            # doseArrayPTV = doseRef[PTVMask]  # use the same threshold
            prescriptionDose = np.percentile(doseArrayPTV, prescriptionLevel)
            R50Thresh = prescriptionDose * factor
            Rscore = np.sum(doseArray > R50Thresh) / ptvVoxels
            RscoreList.append(Rscore)
            R50ThreshList.append(R50Thresh)
        print(R50ThreshList)
        ratio = RscoreList[2] / RscoreList[1]
        currentLine = "| {:03d} | {:.3f} | {:.3f} | {:.3f} |".format(i+1, *RscoreList)
        content.append(currentLine)
    content = "\n".join(content)
    print(content)


if __name__ == "__main__":
    # DVH_comp()
    # R50Show()
    R50Calc()