import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import json
import h5py
from io import BytesIO

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
sourceFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB"
numPatients = 5

def R50Calc():
    factor = 0.5
    prescriptionLevel = 10
    
    content = [
        "PTV dose: {}%".format(100 - prescriptionLevel),
        "R{} comparison".format(int(factor * 100)),
        "| Patient | Clinical | UHPP | SOTA |", "| - | - | - | - | - |"]
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
        currentLine = "| {:03d} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(i+1, *RscoreList, ratio)
        content.append(currentLine)

        if False:
            patientImageFolder = os.path.join(figureFolder, patientName)
            if not os.path.isdir(patientImageFolder):
                os.mkdir(patientImageFolder)
            for j in range(dimension_flip[0]):
                densitySlice = density[j, :, :]
                PTVSlice = PTVMask[j, :, :]
                fig = plt.figure(figsize=(12, 4))
                gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
                PTVcontours = measure.find_contours(PTVSlice)
                for k in range(3):
                    current_block = fig.add_subplot(gs[0, k])
                    current_block.imshow(densitySlice, vmin=0, vmax=1200, cmap="gray")
                    doseSlice = doseList[k][j, :, :]
                    current_block.imshow(doseSlice, vmin=0, vmax=50, alpha=0.3, cmap="jet")
                    doseR50 = doseSlice > R50Thresh
                    doseR50Contours = measure.find_contours(doseR50)
                    for contour in PTVcontours:
                        current_block.plot(contour[:, 1], contour[:, 0], color=colors[0], linewidth=1, linestyle="--")
                    for contour in doseR50Contours:
                        current_block.plot(contour[:, 1], contour[:, 0], color=colors[1], linewidth=1, linestyle="--")
                fig.tight_layout()
                figureFile = os.path.join(patientImageFolder, "{:03d}.png".format(j+1))
                plt.savefig(figureFile)
                plt.close(fig)
                plt.clf()
                print(figureFile)
    content = "\n".join(content)
    print(content)


def DVH_doseWash():
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

        # calculate the prescription dose
        doseRefPTV = doseRef[PTVMask]
        prescriptionDose = np.percentile(doseRefPTV, prescriptionLevel)
        R50Level = prescriptionDose * 0.5
        doseShowMax = np.max(doseRefPTV)

        for j, struct in enumerate(StructureList):
            color = colors[j]
            struct_mask = FastDoseROIDict[struct] > 0
            structDoseRef = doseRef[struct_mask]
            structDoseRef = np.sort(structDoseRef)
            structDoseRef = np.insert(structDoseRef, 0, 0)
            structDoseUHPP = doseUHPP[struct_mask]
            structDoseUHPP = np.sort(structDoseUHPP)
            structDoseUHPP = np.insert(structDoseUHPP, 0, 0)
            structDoseSOTA = doseSOTA[struct_mask]
            structDoseSOTA = np.sort(structDoseSOTA)
            structDoseSOTA = np.insert(structDoseSOTA, 0, 0)
            n_points = structDoseUHPP.size
            y_axis = 100 * (1 - np.arange(n_points)/n_points)
            plt.plot(structDoseRef, y_axis, color=color, linestyle="-", linewidth=0.5)
            plt.plot(structDoseUHPP, y_axis, color=color, linestyle="-", linewidth=2)
            plt.plot(structDoseSOTA, y_axis, color=color, linestyle="--", linewidth=2)
        plt.tight_layout()
        figureFile = os.path.join(patientFolder, "FastDose", "plan2", "DVH_comp.png")
        plt.savefig(figureFile)
        plt.clf()
        print(figureFile)

        # draw doseWash
        PTVMask = FastDoseROIDict["PTV"]
        z, y, x = calcCentroid(PTVMask).astype(int)
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1])
        for j in range(3):
            currentDose = doseList[j]
            doseCoronal = currentDose[:, y, :]
            densityCoronal = density[:, y, :]
            block = fig.add_subplot(gs[j, 0])
            block.imshow(densityCoronal, cmap="gray", vmin=0, vmax=1600)
            block.imshow(doseCoronal, cmap="jet", vmin=0, vmax=doseShowMax, alpha=0.3*(doseCoronal>R50Level))
            for k, struct in enumerate(StructureList):
                structMask = FastDoseROIDict[struct]
                structMaskSlice = structMask[:, y, :]
                contours = measure.find_contours(structMaskSlice)
                for contour in contours:
                    block.plot(contour[:, 1], contour[:, 0], color=colors[k], linewidth=1)
            
            doseSagittal = currentDose[:, :, x]
            densitySagittal = density[:, :, x]
            block = fig.add_subplot(gs[j, 1])
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
            block.imshow(densityAxial, cmap="gray", vmin=0, vmax=1600)
            block.imshow(doseAxial, cmap="jet", vmin=0, vmax=50, alpha=0.3*(doseAxial>R50Level))
            for k, struct in enumerate(StructureList):
                structMask = FastDoseROIDict[struct]
                structMaskSlice = structMask[z, :, :]
                contours = measure.find_contours(structMaskSlice)
                for contour in contours:
                    block.plot(contour[:, 1], contour[:, 0], color=colors[k], linewidth=1)
        doseWashFile = os.path.join(patientFolder, "FastDose", "plan2", "doseWash.png")
        fig.tight_layout()
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

def calcCentroid2d(mask):
    mask = mask > 0
    nPixels = np.sum(mask)
    shape = mask.shape

    xWeight = np.arange(shape[0])
    xWeight = np.expand_dims(xWeight, axis=1)
    xCoord = np.sum(xWeight * mask) / nPixels

    yWeight = np.arange(shape[1])
    yWeight = np.expand_dims(yWeight, axis=0)
    yCoord = np.sum(yWeight * mask) / nPixels

    result = np.array((xCoord, yCoord))
    return result


def beamExamine():
    """
    This function examines the relationship between
    where the beams pass and the OAR regions
    """
    targetFolder = os.path.join(sourceFolder, "beamExamine")
    if not os.path.isdir(targetFolder):
        os.mkdir(targetFolder)
    prescriptionLevel = 10
    factor = 0.5
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)

        dimension = os.path.join(patientFolder, "FastDose", "prep_output", 'dimension.txt')
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = eval(dimension.replace(" ", ", "))
        dimension_flip = np.flip(dimension)

        density = os.path.join(patientFolder, "density_raw.bin")
        density = np.fromfile(density, dtype=np.uint16)
        density = np.reshape(density, dimension_flip)

        FastDoseROIFile = os.path.join(patientFolder, "FastDose", "prep_output_else", "roi_list.h5")
        FastDoseROIDict = getStructures(FastDoseROIFile)
        FastDoseROIDict = FastDoseDictRename(FastDoseROIDict)

        targetPatientFolder = os.path.join(targetFolder, patientName)
        assert os.path.isdir(targetPatientFolder)
        QihuiRyanMaskFolder = os.path.join(targetPatientFolder, "masksQihuiRyan")
        QihuiRyanMaskDict = getMaskDict(QihuiRyanMaskFolder, dimension_flip)
        
        # common, dict1_unique, dict2_unique = keysComp(FastDoseROIDict, QihuiRyanMaskDict)
        # assert len(dict1_unique) == 0 and len(dict2_unique) == 0

        PTVMask = FastDoseROIDict["PTV"].astype(bool)
        BODYMask = FastDoseROIDict["BODY"].astype(bool)
        notBodyMask = np.logical_not(BODYMask)

        refDose = os.path.join(patientFolder, "doseNorm.bin")
        refDose = readDose(refDose, dimension_flip)
        refDose[notBodyMask] = 0.0
        
        QihuiRyanDose = os.path.join(patientFolder, "QihuiRyan", "doseQihuiRyan_else.bin")
        QihuiRyanDose = readDose(QihuiRyanDose, dimension_flip)
        QihuiRyanDose[notBodyMask] = 0.0

        FastDoseDose = os.path.join(patientFolder, "FastDose", "plan2", "dose.bin")
        FastDoseDose = readDose(FastDoseDose, dimension_flip)
        FastDoseDose[notBodyMask] = 0.0

        figureFolder = os.path.join(targetPatientFolder, "doseShow")
        if not os.path.isdir(figureFolder):
            os.mkdir(figureFolder)
        doseShowThresh = np.percentile(refDose[PTVMask], prescriptionLevel) * factor
        doseShowMax = np.max(refDose)
        
        for j in range(dimension_flip[0]):
            fig = plt.figure(figsize=(8, 4))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
            block1 = fig.add_subplot(gs[0, 0])
            block2 = fig.add_subplot(gs[0, 1])

            densitySlice = density[j, :, :]
            block1.imshow(densitySlice, cmap="gray", vmin=0, vmax=1500)
            block2.imshow(densitySlice, cmap="gray", vmin=0, vmax=1500)

            FastDoseSlice = FastDoseDose[j, :, :]
            block1.imshow(FastDoseSlice, cmap="jet", vmin=0, vmax=doseShowMax,
                alpha=0.3 * (FastDoseSlice > doseShowThresh))
            QihuiRyanSlice = QihuiRyanDose[j, :, :]
            block2.imshow(QihuiRyanSlice, cmap="jet", vmin=0, vmax=doseShowMax,
                alpha=0.3 * (QihuiRyanSlice > doseShowThresh))

            commonKeys = list(FastDoseROIDict.keys())
            for k, key in enumerate(commonKeys):
                maskSlice = FastDoseROIDict[key][j, :, :]
                color = colors[k]
                contours = measure.find_contours(maskSlice)
                for contour in contours:
                    block1.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1)
            for k, key in enumerate(commonKeys):
                maskSlice = FastDoseROIDict[key][j, :, :]
                color = colors[k]
                contours = measure.find_contours(maskSlice)
                for contour in contours:
                    block2.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1)
            figureFile = os.path.join(figureFolder, "{:03d}.png".format(j+1))
            plt.tight_layout()
            plt.savefig(figureFile)
            plt.close(fig)
            plt.clf()
            print(figureFile)


def readDose(file, dimension):
    result = np.fromfile(file, dtype=np.float32)
    result = np.reshape(result, dimension)
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

def keysComp(dict1, dict2):
    dict1_keys = set(dict1.keys())
    dict2_keys = set(dict2.keys())
    inter = dict1_keys.intersection(dict2_keys)
    dict1_unique = dict1_keys - inter
    dict2_unique = dict2_keys - inter
    return inter, dict1_unique, dict2_unique


def getMaskDict(folder, dimension):
    files = os.listdir(folder)
    result = {}
    for file in files:
        name = file.split(".")[0]
        path = os.path.join(folder, file)
        array = np.fromfile(path, dtype=np.uint8)
        array = np.reshape(array, dimension)
        result[name] = array > 0
    return result


def generateAdditionalMask():
    """
    After I took a closer look at the dose and the anatomy,
    I discovered that the abnormal dose spread is due to that
    there wasn't enough anatomies in the inferior direction.
    """
    SKINname = "SKIN"
    resultName = "ELSE"
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i + 1)
        dimension = os.path.join(sourceFolder, patientName, "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = eval(dimension.replace(" ", ", "))
        dimension_flip = np.flip(dimension)

        maskFolder = os.path.join(sourceFolder, patientName, "InputMask")
        structures = [a.split(".")[0] for a in os.listdir(maskFolder)]
        assert SKINname in structures
        maskDict = getMaskDict(maskFolder, dimension_flip)
        skinMask = maskDict[SKINname]
        # ROIs = {name: array for name, array in maskDict.items() if
        #     name != SKINname and name != resultName}
        ROIs = [array for name, array in maskDict.items() if
            name != SKINname and name != resultName]
        ROI_union = np.logical_or.reduce(ROIs)
        resultArray = np.logical_and(skinMask, np.logical_not(ROI_union)).astype(np.uint8)
        targetFile = os.path.join(maskFolder, "{}.bin".format(resultName))
        resultArray.tofile(targetFile)
        print(targetFile)


def json_prepare():
    PTV_name = "ROI"
    skin_name = "SKIN"
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)
        maskFolder = os.path.join(patientFolder, "InputMask")
        structures = [a.split(".")[0] for a in os.listdir(maskFolder)]
        assert PTV_name in structures and skin_name in structures
        normalStructures = [a for a in structures if a != PTV_name and a != skin_name]
        normalStructures.insert(0, skin_name)
        result = {"prescription": 20,
            "ptv": PTV_name,
            "oar": normalStructures}
        result = json.dumps(result, indent=4)
        resultFile = os.path.join(patientFolder, "FastDose", "structures_else.json")
        with open(resultFile, "w") as f:
            f.write(result)
        print(resultFile)


def StructureInfoPrep():
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i + 1)
        FastDoseFolder = os.path.join(sourceFolder, patientName, "FastDose")
        jsonFile = os.path.join(FastDoseFolder, "structures_else.json")
        with open(jsonFile, "r") as f:
            structures = f.read()
        structures = json.loads(structures)
        oarList = structures["oar"]
        PTV_name = structures["ptv"]
        SKIN_name = "SKIN"
        ELSE_name = "ELSE"
        assert SKIN_name in oarList and ELSE_name in oarList
        oarList = [a for a in oarList if a not in [SKIN_name, ELSE_name]]
        
        content = ["Name,maxWeights,maxDose,minDoseTargetWeights,"
            "minDoseTarget,OARWeights,IdealDose",
            "{},100,60,100,60,NaN,60".format(PTV_name)]
        for oar in oarList:
            content.append("{},0,18,NaN,NaN,5,0".format(oar))
        content.append("{},0,18,NaN,NaN,2,0".format(ELSE_name))
        content = "\n".join(content)
        file = os.path.join(FastDoseFolder, "StructureInfo_else.csv")
        with open(file, "w") as f:
            f.write(content)
        print(file)


def dosePhysicalStudy():
    """
    Here, we extracted the physical dose from the original dataset,
    from which we can infer the prescription dose, according to the SIB paper.
    """
    content = ["| Patient | Clinical | UHPP | SOTA |", "| - | - | - | - | - |"]
    prescriptionList = [45, 37.5, 45, 45, 45]
    structs_exclude_DVH = ["RingStructure", "BODY", "ELSE"]
    for i in range(numPatients):
        prescriptionDose = prescriptionList[i]
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)
        dimension = os.path.join(patientFolder, "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension_flip = np.flip(dimension)

        doseNorm = os.path.join(patientFolder, "doseNorm.bin")
        doseNorm = getArray(doseNorm, np.float32, dimension_flip)

        dosePhysical = os.path.join(patientFolder, "dosePhysical.bin")
        dosePhysical = getArray(dosePhysical, np.float32, dimension_flip)

        doseUHPP = os.path.join(patientFolder, "FastDose", "plan2", "dose.bin")
        doseUHPP = getArray(doseUHPP, np.float32, dimension_flip)

        doseSOTA = os.path.join(patientFolder, "QihuiRyan", "doseQihuiRyan_else5.bin")
        doseSOTA = getArray(doseSOTA, np.float32, dimension_flip)

        bodyMask = os.path.join(patientFolder, "InputMask", "SKIN.bin")
        bodyMask = getArray(bodyMask, np.uint8, dimension_flip).astype(bool)
        notBodyMask = np.logical_not(bodyMask)

        PTVMask = os.path.join(patientFolder, "InputMask", "ROI.bin")
        PTVMask = getArray(PTVMask, np.uint8, dimension_flip).astype(bool)

        density = os.path.join(patientFolder, "density_raw.bin")
        density = getArray(density, np.uint16, dimension_flip)

        # normalize dose to physical dose
        rescaleFactor = np.max(dosePhysical) / np.max(doseNorm)
        doseNorm_new = doseNorm * rescaleFactor
        # sanity check
        diff = doseNorm_new - dosePhysical
        assert np.max(np.abs(diff)) < 1e-4
        doseSOTA *= rescaleFactor
        doseUHPP *= rescaleFactor
        
        # dose mask
        dosePhysical[notBodyMask] = 0
        doseSOTA[notBodyMask] = 0
        doseUHPP[notBodyMask] = 0

        PTVVoxels = np.sum(PTVMask)
        R50Thresh = prescriptionDose * 0.5
        R50_physical = np.sum(dosePhysical > R50Thresh) / PTVVoxels
        R50_SOTA = np.sum(doseSOTA > R50Thresh) / PTVVoxels
        R50_UHPP = np.sum(doseUHPP > R50Thresh) / PTVVoxels
        currentLine = "| {} | {:.3f} | {:.3f} | {:.3f} |".format(patientName, R50_physical, R50_UHPP, R50_SOTA)
        content.append(currentLine)

        # draw DVH
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
        if True:
            for j, struct in enumerate(StructureList):
                color = colors[j]
                struct_mask = FastDoseROIDict[struct]
                
                structDosePhysical = dosePhysical[struct_mask]
                structDosePhysical = np.sort(structDosePhysical)
                structDosePhysical = np.insert(structDosePhysical, 0, 0)

                structDoseUHPP = doseUHPP[struct_mask]
                structDoseUHPP = np.sort(structDoseUHPP)
                structDoseUHPP = np.insert(structDoseUHPP, 0, 0)

                structDoseSOTA = doseSOTA[struct_mask]
                structDoseSOTA = np.sort(structDoseSOTA)
                structDoseSOTA = np.insert(structDoseSOTA, 0, 0)

                n_points = structDoseUHPP.size
                y_axis = 100 * (1 - np.arange(n_points)/n_points)
                plt.plot(structDosePhysical, y_axis, color=color, linestyle="-", linewidth=0.5)
                plt.plot(structDoseUHPP, y_axis, color=color, linestyle="-", linewidth=2)
                plt.plot(structDoseSOTA, y_axis, color=color, linestyle="--", linewidth=2)
            plt.tight_layout()
            figureFile = os.path.join(patientFolder, "FastDose", "plan2", "DVH_comp.png")
            plt.savefig(figureFile)
            plt.clf()
            print(figureFile)
        
        if True:
            # draw doseWash
            doseShowMax = np.max(dosePhysical)
            doseList = [dosePhysical, doseUHPP, doseSOTA]
            PTVMask = FastDoseROIDict["PTV"]
            z, y, x = calcCentroid(PTVMask).astype(int)
            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1])
            for j in range(3):
                currentDose = doseList[j]
                doseCoronal = currentDose[:, y, :]
                densityCoronal = density[:, y, :]
                block = fig.add_subplot(gs[j, 0])
                block.imshow(densityCoronal, cmap="gray", vmin=0, vmax=1600)
                block.imshow(doseCoronal, cmap="jet", vmin=0, vmax=doseShowMax, alpha=0.3*(doseCoronal>R50Thresh))
                for k, struct in enumerate(StructureList):
                    structMask = FastDoseROIDict[struct]
                    structMaskSlice = structMask[:, y, :]
                    contours = measure.find_contours(structMaskSlice)
                    for contour in contours:
                        block.plot(contour[:, 1], contour[:, 0], color=colors[k], linewidth=1)
                
                doseSagittal = currentDose[:, :, x]
                densitySagittal = density[:, :, x]
                block = fig.add_subplot(gs[j, 1])
                block.imshow(densitySagittal, cmap="gray", vmin=0, vmax=1600)
                block.imshow(doseSagittal, cmap="jet", vmin=0, vmax=doseShowMax, alpha=0.3*(doseSagittal>R50Thresh))
                for k, struct in enumerate(StructureList):
                    structMask = FastDoseROIDict[struct]
                    structMaskSlice = structMask[:, :, x]
                    contours = measure.find_contours(structMaskSlice)
                    for contour in contours:
                        block.plot(contour[:, 1], contour[:, 0], color=colors[k], linewidth=1)
                
                doseAxial = currentDose[z, :, :]
                densityAxial = density[z, :, :]
                block = fig.add_subplot(gs[j, 2])
                block.imshow(densityAxial, cmap="gray", vmin=0, vmax=1600)
                block.imshow(doseAxial, cmap="jet", vmin=0, vmax=50, alpha=0.3*(doseAxial>R50Thresh))
                for k, struct in enumerate(StructureList):
                    structMask = FastDoseROIDict[struct]
                    structMaskSlice = structMask[z, :, :]
                    contours = measure.find_contours(structMaskSlice)
                    for contour in contours:
                        block.plot(contour[:, 1], contour[:, 0], color=colors[k], linewidth=1)
            doseWashFile = os.path.join(patientFolder, "FastDose", "plan2", "doseWash.png")
            fig.tight_layout()
            plt.savefig(doseWashFile)
            plt.close(fig)
            plt.clf()
            print(doseWashFile)


    content = "\n".join(content)
    print(content)



def getArray(file, dtype, dimension):
    result = np.fromfile(file, dtype=dtype)
    result = np.reshape(result, dimension)
    result = np.flip(result, axis=0)
    return result


if __name__ == "__main__":
    # R50Calc()
    DVH_doseWash()
    # beamExamine()
    # generateAdditionalMask()
    # json_prepare()
    # StructureInfoPrep()
    # dosePhysicalStudy()