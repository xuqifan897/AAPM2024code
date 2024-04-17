import os
import numpy as np
import pydicom
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
import matplotlib.colors  as mcolors
from skimage import measure, transform
import json


def drawDVH_ref():
    patientFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003"
    doseFile = os.path.join(patientFolder, "doseExp1.npy")
    densityFile = os.path.join(patientFolder, "density_raw.bin")
    maskFolder = os.path.join(patientFolder, "InputMask")
    maskShape = (160, 227, 227)

    densityArray = np.fromfile(densityFile, dtype=np.uint16)
    densityArray = np.reshape(densityArray, maskShape)
    densityArray = np.flip(densityArray, axis=0)

    masks = []
    files = os.listdir(maskFolder)
    for file in files:
        name = file.split(".")[0]
        path = os.path.join(maskFolder, file)
        maskArray = np.fromfile(path, dtype=np.uint8)
        maskArray = np.reshape(maskArray, maskShape)
        maskArray = np.flip(maskArray, axis=0)
        masks.append((name, maskArray))
        print(name)
    
    doseArray = np.load(doseFile)
    doseArray = np.transpose(doseArray, axes=(2, 0, 1))
    doseArray = transform.resize(doseArray, maskShape)

    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
    if False:
        imageFolder = os.path.join(patientFolder, "doseExp1View")
        if not os.path.isdir(imageFolder):
            os.mkdir(imageFolder)
        maxDose = np.max(doseArray)
        for i in range(maskShape[0]):
            densitySlice = densityArray[i, :, :]
            doseSlice = doseArray[i, :, :]
            plt.imshow(densitySlice, cmap="gray")
            plt.imshow(doseSlice, cmap="jet", vmin=0, vmax=maxDose, alpha=0.3)
            for j, entry in enumerate(masks):
                name, array = entry
                color = colors[j]
                maskSlice = array[i, :, :]
                contours = measure.find_contours(maskSlice)
                initial = True
                for contour in contours:
                    if initial:
                        plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                        initial = False
                    else:
                        plt.plot(contour[:, 1], contour[:, 0], color=color)
            plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
            imageFile = os.path.join(imageFolder, "{:03d}.png").format(i)
            plt.savefig(imageFile)
            plt.clf()
            print(imageFile)
    
    PrimaryPTV = "PTV70"
    PTVMask = None
    for name, mask in masks:
        if name == PrimaryPTV:
            PTVMask = mask
            break
    PTVMask = PTVMask > 0
    PTVDose = doseArray[PTVMask]
    thresh = np.percentile(PTVDose, 5)
    doseArray = doseArray / thresh * 70
    
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, entry in enumerate(masks):
        name, mask = entry
        color = colors[i]
        mask = mask > 0
        structDose = doseArray[mask]
        structDose = np.sort(structDose)
        structDose = np.insert(structDose, 0, 0)
        nVoxels = structDose.size
        percentage = np.linspace(100, 0, nVoxels)
        ax.plot(structDose, percentage, label=name, color=color)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Percentile")
    ax.set_title("Reference dose for patient 003")
    file = os.path.join(patientFolder, "DVH_ref.png")
    plt.savefig(file)
    plt.clf()


def PTVMerge():
    maskFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/InputMask"
    PTV1File = os.path.join(maskFolder, "PTV70.bin")
    PTV2File = os.path.join(maskFolder, "PTV56.bin")
    PTV3File = os.path.join(maskFolder, "leftptv56.bin")
    PTV1 = np.fromfile(PTV1File, dtype=np.uint8)
    PTV2 = np.fromfile(PTV2File, dtype=np.uint8)
    PTV3 = np.fromfile(PTV3File, dtype=np.uint8)
    PTV_merge = np.logical_or(PTV1, np.logical_or(PTV2, PTV3))
    PTV_merge_file = os.path.join(maskFolder, "PTVMerge.bin")
    PTV_merge.tofile(PTV_merge_file)


def structuresFileGen():
    """
    This function generates the structures file
    """
    maskFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/InputMask"
    structures = [a.split(".")[0] for a in os.listdir(maskFolder)]
    BODY = "SKIN"
    PTV = "PTVMerge"
    structures.remove(BODY)
    structures.remove(PTV)
    structures.insert(0, BODY)
    
    content = {
        "prescription": 20,
        "ptv": PTV,
        "oar": structures
    }
    content = json.dumps(content, indent=4)
    file = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/structures.json"
    with open(file, "w") as f:
        f.write(content)


def checkInfo():
    exampleFile = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/data/1-012.dcm"
    dataset = pydicom.dcmread(exampleFile)
    print(dataset.RescaleIntercept)
    print(dataset.RescaleSlope)


def StructInfoGen():
    """
    This function generates the StructureInfo.csv file for a new plan
    """
    if True:
        # Specify the variables below
        expFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/FastDose"
        dimensionFile = os.path.join(expFolder, "prep_output", "dimension.txt")

    with open(dimensionFile, "r") as f:
        lines = f.readlines()
    organs = lines[2]
    organs = organs.split(" ")
    organs.remove("")

    PTVs = [("PTV70", 70), ("PTV56", 56), ("leftptv56", 56)]
    irrelevant = ["SKIN", "PTVMerge", "RingStructure"]
    PTV_names = [a[0] for a in PTVs]
    exclude = irrelevant + PTV_names
    OARs = [a for a in organs if a not in exclude]
    print(OARs)

    # prepare the StructureInfo content
    content = "Name,maxWeights,maxDose,minDoseTargetWeights,minDoseTarget,OARWeights,IdealDose"
    for name, dose in PTVs:
        newline = "{},100,{},100,{},NaN,{}".format(name, dose, dose, dose)
        content = content + "\n" + newline
    for name in OARs:
        newline = newline = "{},0,18,NaN,NaN,5,0".format(name)
        content = content + "\n" + newline
    print(content)
    
    outputFile = os.path.join(expFolder, "StructureInfo.csv")
    with open(outputFile, 'w') as f:
        f.write(content)


def viewDensity():
    """
    Just view the preprocessed density
    """
    prepFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/FastDose/prep_output"
    densityFile = os.path.join(prepFolder, "density.raw")
    densityShape = (160, 227, 227)
    densityArray = np.fromfile(densityFile, dtype=np.float32)
    densityArray = np.reshape(densityArray, densityShape)
    densityViewFolder = os.path.join(prepFolder, "densityView")
    if not os.path.isdir(densityViewFolder):
        os.mkdir(densityViewFolder)
    for i in range(densityShape[0]):
        slice = densityArray[i, :, :]
        image = os.path.join(densityViewFolder, "{:03d}.png".format(i))
        plt.imsave(image, slice, cmap="gray")
        print(image)


def doseMatMerge():
    """
    This function merges different matrices into one
    """
    NonZeroElements_collection = []
    numRowsPerMat_collection = []
    offsetsBuffer_collection = []
    columnsBuffer_collection = []
    valuesBuffer_collection = []
    fluenceMap_collection = []

    expFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/FastDose"
    numMatrices = 4
    for i in range(1, numMatrices+1):
        doseMatFolder = os.path.join(expFolder, "doseMat{}/doseMatFolder".format(i))
        NonZeroElementsFile = os.path.join(doseMatFolder, "NonZeroElements.bin")
        numRowsPerMatFile = os.path.join(doseMatFolder, "numRowsPerMat.bin")
        offsetsBufferFile = os.path.join(doseMatFolder, "offsetsBuffer.bin")
        columnsBufferFile = os.path.join(doseMatFolder, "columnsBuffer.bin")
        valuesBufferFile = os.path.join(doseMatFolder, "valuesBuffer.bin")
        fluenceMapFile = os.path.join(doseMatFolder, "fluenceMap.bin")

        NonZeroElements = np.fromfile(NonZeroElementsFile, dtype=np.uint64)
        numRowsPerMat = np.fromfile(numRowsPerMatFile, dtype=np.uint64)
        offsetsBuffer = np.fromfile(offsetsBufferFile, dtype=np.uint64)
        columnsBuffer = np.fromfile(columnsBufferFile, dtype=np.uint64)
        valuesBuffer = np.fromfile(valuesBufferFile, dtype=np.float32)
        fluenceMap = np.fromfile(fluenceMapFile, dtype=np.uint8)

        NonZeroElements_collection.append(NonZeroElements)
        numRowsPerMat_collection.append(numRowsPerMat)
        offsetsBuffer_collection.append(offsetsBuffer)
        columnsBuffer_collection.append(columnsBuffer)
        valuesBuffer_collection.append(valuesBuffer)
        fluenceMap_collection.append(fluenceMap)
        print(doseMatFolder)
    
    NonZeroElements = np.concatenate(NonZeroElements_collection)
    numRowsPerMat = np.concatenate(numRowsPerMat_collection, axis=0)
    offsetsBuffer = np.concatenate(offsetsBuffer_collection, axis=0)
    columnsBuffer = np.concatenate(columnsBuffer_collection, axis=0)
    valuesBuffer = np.concatenate(valuesBuffer_collection, axis=0)
    fluenceMap = np.concatenate(fluenceMap_collection, axis=0)
    print("concatenation")

    targetFolder = os.path.join(expFolder, "doseMatMerge/doseMatFolder")
    if not os.path.isdir(targetFolder):
        os.makedirs(targetFolder)
    NonZeroElements.tofile(os.path.join(targetFolder, "NonZeroElements.bin"))
    numRowsPerMat.tofile(os.path.join(targetFolder, "numRowsPerMat.bin"))
    offsetsBuffer.tofile(os.path.join(targetFolder, "offsetsBuffer.bin"))
    columnsBuffer.tofile(os.path.join(targetFolder, "columnsBuffer.bin"))
    valuesBuffer.tofile(os.path.join(targetFolder, "valuesBuffer.bin"))
    fluenceMap.tofile(os.path.join(targetFolder, "fluenceMap.bin"))
    print(targetFolder)


def patch():
    """
    Added fleunceMap to the code above
    """
    fluenceMap_collection = []
    expFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/FastDose"
    numMatrices = 4
    for i in range(1, numMatrices+1):
        doseMatFolder = os.path.join(expFolder, "doseMat{}/doseMatFolder".format(i))
        fluenceMapFile = os.path.join(doseMatFolder, "fluenceMap.bin")
        fluenceMap = np.fromfile(fluenceMapFile, dtype=np.uint8)
        fluenceMap_collection.append(fluenceMap)
        print(doseMatFolder)
    
    fluenceMap = np.concatenate(fluenceMap_collection, axis=0)
    print("concatenation")

    targetFolder = os.path.join(expFolder, "doseMatMerge/doseMatFolder")
    fluenceMap.tofile(os.path.join(targetFolder, "fluenceMap.bin"))
    print(targetFolder)


def drawDose_opt():
    """
    This function draws the dose map generated by the optimization algorithm
    """
    patientFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003"
    expFolder = os.path.join(patientFolder, "FastDose")
    planFolder = os.path.join(expFolder, "plan1")
    doseFile = os.path.join(planFolder, "dose.bin")
    doseShape = (160, 227, 227)
    doseArray = np.fromfile(doseFile, dtype=np.float32)
    doseArray = np.reshape(doseArray, doseShape)
    doseArray = np.flip(doseArray, axis=0)
    
    densityFile = os.path.join(patientFolder, "density_raw.bin")
    densityArray = np.fromfile(densityFile, dtype=np.uint16)
    densityArray = np.reshape(densityArray, doseShape)

    maskFolder = os.path.join(patientFolder, "InputMask")
    files = os.listdir(maskFolder)
    masks = []
    exclude = ["PTVMerge"]
    for file in files:
        name = file.split(".")[0]
        if name in exclude:
            continue
        path = os.path.join(maskFolder, file)
        maskArray = np.fromfile(path, dtype=np.uint8)
        maskArray = np.reshape(maskArray, doseShape)
        masks.append((name, maskArray))
    
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
    if True:
        nSlices = densityArray.shape[0]
        doseViewFolder = os.path.join(planFolder, "doseView")
        if not os.path.isdir(doseViewFolder):
            os.mkdir(doseViewFolder)
        maxDose = 80
        for i in range(nSlices):
            densitySlice = densityArray[i, :, :]
            doseSlice = doseArray[i, :, :]
            fig, ax = plt.subplots(figsize=(8, 5))
            plt.imshow(densitySlice, cmap="gray")
            plt.imshow(doseSlice, cmap="jet", vmin=0, vmax=maxDose, alpha=0.3)
            for j, entry in enumerate(masks):
                color = colors[j]
                name, maskArray = entry
                maskSlice = maskArray[i, :, :]
                contours = measure.find_contours(maskSlice)
                initial = True
                for contour in contours:
                    if initial:
                        plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                        initial = False
                    else:
                        plt.plot(contour[:, 1], contour[:, 0], color=color)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            file = os.path.join(doseViewFolder, "{:03d}.png".format(i))
            plt.savefig(file)
            plt.clf()
            print(file)


def drawDVH_opt():
    """
    Draw DVH for the optimized plan
    """
    patientFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003"
    planFolder = os.path.join(patientFolder, "FastDose", "plan1")
    maskFolder = os.path.join(patientFolder, "InputMask")
    doseFile = os.path.join(planFolder, "dose.bin")
    refDoseFile = os.path.join(patientFolder, "doseExp1.npy")
    doseShape = (160, 227, 227)

    doseArray = np.fromfile(doseFile, dtype=np.float32)
    doseArray = np.reshape(doseArray, doseShape)
    masks = []
    files = os.listdir(maskFolder)
    exclude = ["SKIN", "PTVMerge"]
    PrimaryPTV = "PTV70"
    primaryMask = None
    for file in files:
        name = file.split(".")[0]
        if name in exclude:
            continue
        path = os.path.join(maskFolder, file)
        maskArray = np.fromfile(path, dtype=np.uint8)
        maskArray = np.reshape(maskArray, doseShape)
        masks.append((name, maskArray))
        if name == PrimaryPTV:
            primaryMask = maskArray
    
    primaryMask = primaryMask > 0
    primaryDose = doseArray[primaryMask]
    thresh = np.percentile(primaryDose, 5)
    doseArray = doseArray / thresh * 70

    doseRef = np.load(refDoseFile)
    doseRef = np.transpose(doseRef, axes=(2, 0, 1))
    doseRef = np.flip(doseRef, axis=0)
    doseRef = transform.resize(doseRef, doseShape)
    primaryDoseRef = doseRef[primaryMask]
    threshRef = np.percentile(primaryDoseRef, 5)
    doseRef = doseRef / threshRef * 70
    
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, entry in enumerate(masks):
        color = colors[i]
        name, mask = entry
        mask = mask > 0
        struct_dose = doseArray[mask]
        struct_dose = np.sort(struct_dose)
        struct_dose = np.insert(struct_dose, 0, 0.0)
        numPoints = struct_dose.size
        percentile = np.linspace(100, 0, numPoints)
        ax.plot(struct_dose, percentile, color=color, label=name)

        struct_ref_dose = doseRef[mask]
        struct_ref_dose = np.sort(struct_ref_dose)
        struct_ref_dose = np.insert(struct_ref_dose, 0, 0)
        ax.plot(struct_ref_dose, percentile, color=color, linestyle="--")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    DVH_opt_file = os.path.join(patientFolder, "DVH_opt.png")
    plt.savefig(DVH_opt_file)
    plt.clf()


if __name__ == "__main__":
    # drawDVH_ref()
    # PTVMerge()
    # structuresFileGen()
    # checkInfo()
    # StructInfoGen()
    # viewDensity()
    # doseMatMerge()
    # patch()
    # drawDose_opt()
    drawDVH_opt()