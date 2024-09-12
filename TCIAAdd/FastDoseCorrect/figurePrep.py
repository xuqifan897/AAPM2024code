import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import h5py
from skimage import measure
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import nrrd
from collections import OrderedDict

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
rootFolder = "/data/qifan/projects/FastDoseWorkplace/TCIAAdd"
sourceFolder = os.path.join(rootFolder, "plansAngleCorrect")
patientList = ["002", "003", "009", "013", "070", "125", "132", "190"]
isoRes = 2.5
figureFolder = "/data/qifan/projects/AAPM2024/manufigures/TCIACompare"
if not os.path.isdir(figureFolder):
    os.mkdir(figureFolder)

exclude = ["SKIN", "PTVMerge", "rind", "PTVSeg0", "PTVSeg1", "PTVSeg2", "PTVSeg3", "PTVMerge",
           "RingStructure", "RingStructModify", "RingStructUpper", "RingStructLower",
           "RingStructMiddle"]
Converge = {"BrainStem": ["BRAIN_STEM", "Brainstem", "BRAIN_STEM_PRV"],
            "OralCavity": ["oralcavity", "oralCavity", "ORAL_CAVITY", "OralCavity"],
            "OPTIC_NERVE": ["OPTIC_NERVE", "OPTC_NERVE"]}
ConvergeReverse = {}
for name, collection in Converge.items():
    for child in collection:
        ConvergeReverse[child] = name

DVHStructs = {"PTV70": ["PTV70"], "PTV56": ["PTV56"], "SPINAL_CORD": ["SPINAL_CORD"],
    "BrainStem": ["BrainStem"], "PAROTID": ["PAROTID_LT", "PAROTID_RT"],
    "MANDIBLE": ["MANDIBLE"], "OPTICS": ["EYE_RT", "EYE_LT", "OPTIC_NERVE", "CHIASM"],
    "LARYNX": ["LARYNX"], "OralCavity": ["OralCavity"]}
DVHstructsColorMap = {}

beamViewStructs = []
beamViewColorMap = {}

StructsMetadata = {
        "002": {"exclude": ["TransPTV56", "CTV56", "TransPTV70", "GTV", "CTV56", "avoid"],
            "PTV": ["PTV70", "PTV56"],
            "BODY": "SKIN"},
        "003": {"exclude": ["GTV", "ptv54combo", "transvol70"],
            "PTV": ["CTV56", "PTV56", "PTV70", "leftptv56"],
            "BODY": "SKIN"},
        "009": {"exclude": ["ptv_70+", "GTV", "CTV70", "ltpar+", "rtpar+"],
            "PTV": ["CTV56", "PTV56", "PTV70"],
            "BODY": "SKIN"},
        "013": {"exclude": ["CTV70", "GTV"],
             "PTV": ["CTV56", "PTV56", "PTV70"],
             "BODY": "SKIN"},
        "070": {"exclude": ["CTV56", "CTV70", "GTV"],
             "PTV": ["PTV56", "PTV70"],
             "BODY": "SKIN"},
        "125": {"exclude": ["CTV56", "CTV70", "GTV"],
              "PTV": ["PTV56", "PTV70"],
              "BODY": "SKIN"},
        "132": {"exclude": ["CTV70", "GTV"],
              "PTV": ["CTV56", "PTV56", "PTV70"],
              "BODY": "SKIN"},
        "190": {"exclude": ["CTV70", "GTV"],
              "PTV": ["CTV56", "PTV56", "PTV70"],
              "BODY": "SKIN"}
    }


def StructsInit():
    """
    This function is to generate a coherent structure list for all patients
    """
    global beamViewStructs, beamViewColorMap
    for patient in patientList:
        if ".txt" in patient:
            continue
        patientFolder = os.path.join(rootFolder, patient)
        InputMaskFolder = os.path.join(patientFolder, "PlanMask")
        structuresLocal = os.listdir(InputMaskFolder)
        structuresLocal = [a.split(".")[0].replace(" ", "") for a in structuresLocal]
        for a in structuresLocal:
            if a not in beamViewStructs:
                beamViewStructs.append(a)
    beamViewStructs_copy = []
    for name in beamViewStructs:
        if name in ConvergeReverse:
            name = ConvergeReverse[name]
        if name not in beamViewStructs_copy and name not in exclude and "+" not in name:
            beamViewStructs_copy.append(name)
    beamViewStructs = beamViewStructs_copy.copy()

    # bring PTV70 and PTV56 forward
    beamViewStructs.remove("PTV70")
    beamViewStructs.remove("PTV56")
    beamViewStructs.insert(0, "PTV56")
    beamViewStructs.insert(0, "PTV70")

    # add the four sets of beams for the four isocenters
    additional = ["SKIN"] + ["PTVSeg{}".format(i) for i in range(4)]
    allStructs = beamViewStructs + additional
    for i in range(len(allStructs)):
        beamViewColorMap[allStructs[i]] = colors[i]

    for i, name in enumerate(DVHStructs.keys()):
        DVHstructsColorMap[name] = colors[i]


def DVH_comp():
    nRows = 4
    nCols = 4
    fig = plt.figure(figsize=(16, 13))
    gs = gridspec.GridSpec(nRows, nCols, height_ratios=[4, 4, 4, 0.2],
        width_ratios=[0.2, 4, 4, 4])
    
    # create the common y label
    ylabel_block = fig.add_subplot(gs[:-1, 0])
    ylabel_block.text(0.9, 0.5, "Fractional Volume (%)", ha="center", va="center",
        rotation="vertical", fontsize=20)
    ylabel_block.axis("off")

    # create the common x label
    xlabel_block = fig.add_subplot(gs[-1, 1:])
    xlabel_block.text(0.5, 0.5, "Dose (Gy)", ha="center", va="center", fontsize=20)
    xlabel_block.axis("off")

    patientsPerRow = 3
    normLevel = 10
    prescription = 70
    numPatients = len(patientList)
    for i in range(numPatients):
        patientName = patientList[i]
        doseClinic = os.path.join(rootFolder, patientName, "dose.bin")
        doseClinic = np.fromfile(doseClinic, dtype=np.float32)
        dataSize = doseClinic.size

        doseFastDose = os.path.join(sourceFolder, patientName, "FastDose", "plan1", "dose.bin")
        doseFastDose = np.fromfile(doseFastDose, dtype=np.float32)
        assert(doseFastDose.size == dataSize)

        doseQihuiRyan = os.path.join(sourceFolder, patientName, "QihuiRyan", "doseRef.bin")
        doseQihuiRyan = np.fromfile(doseQihuiRyan, dtype=np.float32)
        assert(doseQihuiRyan.size == dataSize)
        
        maskDict = {}
        maskFolder = os.path.join(rootFolder, patientName, "PlanMask")
        structs = [a.split(".")[0] for a in os.listdir(maskFolder)]
        for struct in structs:
            if struct in exclude:
                continue
            structFile = os.path.join(maskFolder, struct + ".bin")
            structMask = np.fromfile(structFile, dtype=np.uint8)
            assert structMask.size == dataSize
            if struct in ConvergeReverse:
                struct = ConvergeReverse[struct]
            maskDict[struct] = structMask
        
        maskDictConverge = {}
        for key, include in DVHStructs.items():
            localMask = None
            for struct in include:
                if struct not in maskDict:
                    continue
                structMask = maskDict[struct]
                if localMask is None:
                    localMask = structMask
                else:
                    localMask = np.logical_or(localMask, structMask)
            if localMask is not None:
                maskDictConverge[key] = localMask.astype(bool)
        
        # normalize dose
        ptv70 = "PTV70"
        ptv70 = maskDictConverge[ptv70]
        thresh = np.percentile(doseClinic[ptv70], normLevel)
        doseClinic *= prescription / thresh

        thresh = np.percentile(doseFastDose[ptv70], normLevel)
        doseFastDose *= prescription / thresh

        thresh = np.percentile(doseQihuiRyan[ptv70], normLevel)
        doseQihuiRyan *= prescription / thresh

        rowIdx = i // patientsPerRow
        colIdx = i % patientsPerRow + 1
        block = fig.add_subplot(gs[rowIdx, colIdx])

        for struct, mask in maskDictConverge.items():
            color = DVHstructsColorMap[struct]

            doseClinicMask = doseClinic[mask]
            doseClinicMask = np.sort(doseClinicMask)
            doseClinicMask = np.insert(doseClinicMask, 0, 0.0)

            doseFastDoseMask = doseFastDose[mask]
            doseFastDoseMask = np.sort(doseFastDoseMask)
            doseFastDoseMask = np.insert(doseFastDoseMask, 0, 0.0)

            doseQihuiRyanMask = doseQihuiRyan[mask]
            doseQihuiRyanMask = np.sort(doseQihuiRyanMask)
            doseQihuiRyanMask = np.insert(doseQihuiRyanMask, 0, 0.0)

            yAxis = np.sum(mask) + 1
            yAxis = (1 - np.arange(yAxis) / (yAxis-1)) * 100
            block.plot(doseFastDoseMask, yAxis, color=color, linewidth=3)
            block.plot(doseQihuiRyanMask, yAxis, color=color, linewidth=3, linestyle="--")
            block.plot(doseClinicMask, yAxis, color=color, linewidth=1)
        block.tick_params(axis="x", labelsize=16)
        block.tick_params(axis="y", labelsize=16)
        block.set_title("Patient {}".format(patientName), fontsize=20)
        print(patientName)
    
    legendBlock = fig.add_subplot(gs[2, 3])
    legendBlock.axis("off")
    handles = []
    labels = []
    for name, color in DVHstructsColorMap.items():
        handleEntry = plt.Line2D([0], [0], color=color, lw=2)
        handles.append(handleEntry)
        labels.append(name)
    legendBlock.legend(handles, labels, loc="center", ncols=1, fontsize=16)
    plt.tight_layout()
    figureFile = os.path.join(figureFolder, "TCIACompareDVH.png")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()


def drawDoseWash():
    doseWashFolder = os.path.join(figureFolder, "doseWash")
    if not os.path.isdir(doseWashFolder):
        os.mkdir(doseWashFolder)
    
    ImageHeight = 80
    ImageWidth = 80
    CoronalWidth = 80
    SagittalWidth = 80
    numPatients = len(patientList)
    normLevel = 10
    prescription = 70
    for i in range(numPatients):
        patientName = patientList[i]
        dimension = os.path.join(rootFolder, patientName, "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = eval(dimension.replace(" ", ", "))
        dimension_flip = np.flip(dimension)

        density = os.path.join(rootFolder, patientName, "density_raw.bin")
        density = np.fromfile(density, dtype=np.uint16)
        density = np.reshape(density, dimension_flip)

        doseClinic = os.path.join(rootFolder, patientName, "dose.bin")
        doseClinic = np.fromfile(doseClinic, dtype=np.float32)
        doseClinic = np.reshape(doseClinic, dimension_flip)  # (z, y, x)

        doseFastDose = os.path.join(sourceFolder, patientName, "FastDose", "plan1", "dose.bin")
        doseFastDose = np.fromfile(doseFastDose, dtype=np.float32)
        doseFastDose = np.reshape(doseFastDose, dimension_flip)  # (z, y, x)

        doseQihuiRyan = os.path.join(sourceFolder, patientName, "QihuiRyan", "doseRef.bin")
        doseQihuiRyan = np.fromfile(doseQihuiRyan, dtype=np.float32)
        doseQihuiRyan = np.reshape(doseQihuiRyan, dimension_flip)  # (z, y, x)

        maskFolder = os.path.join(rootFolder, patientName, "PlanMask")
        maskDict = {}
        structs = [a.split(".")[0] for a in os.listdir(maskFolder)]
        body = "SKIN"
        for struct in structs:
            if struct in exclude and struct != body:
                continue
            maskFile = os.path.join(maskFolder, struct+".bin")
            maskArray = np.fromfile(maskFile, dtype=np.uint8)
            maskArray = np.reshape(maskArray, dimension_flip).astype(bool)
            if struct in ConvergeReverse:
                struct = ConvergeReverse[struct]
            maskDict[struct] = maskArray
        
        # normalization
        ptv70 = "PTV70"
        ptv70 = maskDict[ptv70]
        body = maskDict[body]

        doseClinic[np.logical_not(body)] = 0
        doseClinicThresh = np.percentile(doseClinic[ptv70], normLevel)
        doseClinic *= prescription / doseClinicThresh
        doseFastDose[np.logical_not(body)] = 0
        doseFastDoseThresh = np.percentile(doseFastDose[ptv70], normLevel)
        doseFastDose *= prescription / doseFastDoseThresh
        doseQihuiRyan[np.logical_not(body)] = 0
        doseQihuiRyanThresh = np.percentile(doseQihuiRyan[ptv70], normLevel)
        doseQihuiRyan *= prescription / doseQihuiRyanThresh

        centroid = calcCentroid(ptv70)
        z, y, x = centroid.astype(int)

        doseList = [("clinical", doseClinic), ("FastDose", doseFastDose),
            ("QihuiRyan", doseQihuiRyan)]
        imageList = []
        doseShowMax = max(np.max(doseClinic), np.max(doseFastDose), np.max(doseQihuiRyan))
        print(patientName)
        for name, doseArray in doseList:
            threshMask = doseArray > 0.2 * prescription

            densityAxial = density[z, :, :]
            doseAxial = doseArray[z, :, :]
            masksAxial = [(struct, array[z, :, :]) for struct, array in maskDict.items()]
            bodyAxial = body[z, :, :]
            threshMaskAxial = threshMask[z, :, :]
            axialImage = drawSlice(densityAxial, doseAxial, masksAxial, bodyAxial,
                ImageHeight, ImageWidth, beamViewColorMap, doseShowMax, threshMaskAxial)
            
            densityCoronal = np.flip(density[:, y, :], axis=0)
            doseCoronal = np.flip(doseArray[:, y, :], axis=0)
            masksCoronal = [(struct, np.flip(array[:, y, :], axis=0)) for struct, array in maskDict.items()]
            bodyCoronal = np.flip(body[:, y, :], axis=0)
            threshMaskCoronal = np.flip(threshMask[:, y, :], axis=0)
            coronalImage = drawSlice(densityCoronal, doseCoronal, masksCoronal, bodyCoronal,
                ImageHeight, CoronalWidth, beamViewColorMap, doseShowMax, threshMaskCoronal)
            
            densitySagittal = np.flip(density[:, :, x], axis=0)
            doseSagittal = np.flip(doseArray[:, :, x], axis=0)
            masksSagittal = [(struct, np.flip(array[:, :, x], axis=0)) for struct, array in maskDict.items()]
            bodySagittal = np.flip(body[:, :, x], axis=0)
            threshMaskSagittal = np.flip(threshMask[:, :, x], axis=0)
            sagittalImage = drawSlice(densitySagittal, doseSagittal, masksSagittal, bodySagittal,
                ImageHeight, SagittalWidth, beamViewColorMap, doseShowMax, threshMaskSagittal)
            
            ImageRow = np.concatenate((axialImage, coronalImage, sagittalImage), axis=1)
            imageList.append(ImageRow)
            print(name)
        patientImage = np.concatenate(imageList, axis=0)

        # generate colorbar
        colorBarLocal = colorBarGen(doseShowMax, patientImage.shape[0])
        patientImage = np.concatenate((patientImage, colorBarLocal), axis=1)
        patientImageFile = os.path.join(doseWashFolder, patientName+".png")
        plt.imsave(patientImageFile, patientImage)
        print(patientImageFile, "\n")


def colorBarGen(doseShowMax, targetHeight):
    """
    This function generates the colorbar
    """
    image = np.random.rand(100, 100) * 75
    fig, ax = plt.subplots()

    # Set the face color of the figure and axis to black
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    cax = ax.imshow(image, cmap="jet", vmin=0, vmax=doseShowMax)
    cbar = fig.colorbar(cax, ax=ax, orientation="vertical")

    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    cbar.ax.tick_params(axis="y", colors="white", labelsize=16)
    cbar.set_label("Dose (Gy)", color="white", fontsize=16)

    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(canvas.get_width_height()[::-1] + (3,))
    ImageSliced = image[:, -160:-40, :]

    colorBarEnlarged = (targetHeight, ImageSliced.shape[1], 4)
    colorBarEnlarged = np.zeros(colorBarEnlarged, dtype=ImageSliced.dtype)
    offset = int((targetHeight - ImageSliced.shape[0]) / 2)
    colorBarEnlarged[offset: offset+ImageSliced.shape[0], :, :3] = ImageSliced
    # colorBarEnlarged.dtype == np.uint8
    colorBarEnlarged = (colorBarEnlarged / 255).astype(np.float32)
    colorBarEnlarged[:, :, -1] = 1.0
    return colorBarEnlarged


def drawSlice(densitySlice, doseSlice, maskSlice, bodySlice,
    height, width, colorMap, doseShowMax, threshMask):
    doseThresh = 10
    maskCentroid = calcCentroid2d(bodySlice)
    densityCrop = crop_and_fill(densitySlice, maskCentroid, height, width)
    doseCrop = crop_and_fill(doseSlice, maskCentroid, height, width)
    threshMaskCrop = crop_and_fill(threshMask, maskCentroid, height, width)
    maskSliceCrop = []
    for name, mask_slice in maskSlice:
        mask_slice_crop = crop_and_fill(mask_slice, maskCentroid, height, width)
        maskSliceCrop.append((name, mask_slice_crop))
    fig, ax = plt.subplots(figsize=(width/50, height/50), dpi=200)
    ax.imshow(densityCrop, cmap="gray", vmin=0, vmax=2000)
    for name, mask in maskSliceCrop:
        color = colorMap[name]
        contours = measure.find_contours(mask)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=0.5)
    ax.imshow(doseCrop, cmap="jet", vmin=0, vmax=doseShowMax, alpha=threshMaskCrop*0.3)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    plt.clf()

    buf.seek(0)
    image = plt.imread(buf)
    buf.close()
    return image


def crop_and_fill(array, center, height, width):
    # height and width are in half
    # crop an array, and fill the out-of-range values with 0
    topLeftAngleArray = np.array((center[0] - height, center[1] - width)).astype(int)
    bottomRightAngleArray = np.array((center[0] + height, center[1] + width)).astype(int)

    topLeftBound = topLeftAngleArray.copy()
    topLeftBound[topLeftBound < 0] = 0
    bottomRightBound = bottomRightAngleArray.copy()
    if bottomRightBound[0] >= array.shape[0] - 1:
        bottomRightBound[0] = array.shape[0] - 1
    if bottomRightBound[1] >= array.shape[1] - 1:
        bottomRightBound[1] = array.shape[1] - 1
    
    startIdx = topLeftBound - topLeftAngleArray
    endIdx = bottomRightBound - topLeftAngleArray
    canvas = np.zeros((2*height, 2*width), dtype=array.dtype)
    canvas[startIdx[0]: endIdx[0], startIdx[1]: endIdx[1]] = \
        array[topLeftBound[0]: bottomRightBound[0], topLeftBound[1]: bottomRightBound[1]]
    return canvas


def calcCentroid(mask):
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


def nrrdVerification():
    nrrdFastDose = os.path.join(sourceFolder, "nrrdFastDose")
    if not os.path.isdir(nrrdFastDose):
        os.mkdir(nrrdFastDose)
    nrrdQihuiRyan = os.path.join(sourceFolder, "nrrdQihuiRyan")
    if not os.path.isdir(nrrdQihuiRyan):
        os.mkdir(nrrdQihuiRyan)
    nSegs = 4
    
    for patientName in patientList:
        patientMetaData = StructsMetadata[patientName]
        excludeList = patientMetaData["exclude"]
        ptvList = patientMetaData["PTV"]
        bodyName = patientMetaData["BODY"]

        maskFolder = os.path.join(rootFolder, patientName, "PlanMask")
        masks = [a.split(".")[0] for a in os.listdir(maskFolder)]
        masks = [a for a in masks if a not in excludeList]
        assert bodyName in masks
        masks_clean = [a for a in masks if "PTV" not in a and a != bodyName]
        masks_clean.sort()
        masks_clean.insert(0, "PTV56")
        masks_clean.insert(0, "PTV70")
        masks_clean.append(bodyName)

        dimension = os.path.join(rootFolder, patientName, "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension_flip = np.flip(dimension)

        maskDict = {}
        for name in masks_clean:
            if name in exclude and name != "SKIN":
                continue
            maskFile = os.path.join(maskFolder, name + ".bin")
            mask = np.fromfile(maskFile, dtype=np.uint8)
            mask = np.reshape(mask, dimension_flip)  # (z, y, x)
            name_key = name
            if name in ConvergeReverse:
                name_key = ConvergeReverse[name]
            maskDict[name_key] = mask
        print("{} done loading masks".format(patientName))

        # generate the beamMask
        PTVSegMaskList = []
        for segIdx in range(nSegs):
            PTVSegMask = os.path.join(maskFolder, "PTVSeg{}.bin".format(segIdx))
            PTVSegMask = np.fromfile(PTVSegMask, dtype=np.uint8)
            PTVSegMask = np.reshape(PTVSegMask, dimension_flip)  # (z, y, x)
            PTVSegMaskList.append(PTVSegMask)
        
        # generate the beam list
        isocenterMapping = {}
        cumuIdx = 0
        for segIdx in range(nSegs):
            beamListFile = os.path.join(sourceFolder, patientName, "beamlist{}.txt".format(segIdx))
            with open(beamListFile, "r") as f:
                beamListLocal = f.readlines()
            beamListLocal = [eval(a.replace(" ", ", ")) for a in beamListLocal]
            beamListLocal = [np.array(a) * np.pi / 180 for a in beamListLocal]
            for entry in beamListLocal:
                isocenterMapping[cumuIdx] = (entry, segIdx)
                cumuIdx += 1
        
        if False:
            counts = [0] * nSegs
            for key, value in isocenterMapping.items():
                entry, segIdx = value
                counts[segIdx] += 1
        
        if False:
            # generate the beam masks for FastDose
            fastDoseResults = []
            for segIdx in range(nSegs):
                fastDoseResults.append([])
            selectedBeams = os.path.join(sourceFolder, patientName, "FastDose", "plan1", "metadata.txt")
            with open(selectedBeams, "r") as f:
                selectedBeams = f.readlines()
            selectedBeams = selectedBeams[3]
            selectedBeams = eval(selectedBeams.replace("  ", ", "))
            for idx in selectedBeams:
                angle, isocenterIdx = isocenterMapping[idx]
                fastDoseResults[isocenterIdx].append(angle)
    
            beamMaskFastDoseDict = {}
            for segIdx in range(nSegs):
                PTVSegMask = PTVSegMaskList[segIdx]
                PTVSegBeamList = fastDoseResults[segIdx]
                PTVBeamsMask = genBeamsMask(PTVSegMask, PTVSegBeamList)
                beamMaskFastDoseDict["PTVSeg{}".format(segIdx)] = PTVBeamsMask
                print("{} done generating beam mask {}".format(patientName, segIdx))
            FastDoseMask = maskDict.copy()
            for key, value in beamMaskFastDoseDict.items():
                FastDoseMask[key] = value
            result_mask, result_header = nrrdGen(FastDoseMask)
            file = os.path.join(nrrdFastDose, patientName + ".nrrd")
            nrrd.write(file, result_mask, result_header)
            print(file)
        if True:
            beamMaskQihuiRyanDict = {}
            selectedBeams = os.path.join(sourceFolder, patientName, "QihuiRyan",
                "selected_angles_S1_P1.csv")
            with open(selectedBeams, "r") as f:
                selectedBeams = f.readlines()
            selectedBeams = selectedBeams[1:]  # remove the header
            selectedBeams = [a for a in selectedBeams if a != "\n"]
            selectedBeams = [eval(a) for a in selectedBeams]
            # convert the matlab 1-based indexing to python 0-based indexing
            for ii in range(len(selectedBeams)):
                beamIdx, gantry, couch = selectedBeams[ii]
                beamIdx -= 1
                angleRad = np.array((gantry, couch, 0)) * np.pi / 180  # convert degree to rad
                angleRadRef, segIdx = isocenterMapping[beamIdx]
                angleRad[angleRad > np.pi] -= 2 * np.pi
                assert np.linalg.norm(angleRad - angleRadRef) < 1e-4
                selectedBeams[ii] = (angleRad, segIdx)
            QihuiRyanResult = []
            for i in range(nSegs):
                QihuiRyanResult.append([])
            for angleRad, segIdx in selectedBeams:
                QihuiRyanResult[segIdx].append(angleRad)
            beamMaskFastDoseDict = {}
            for segIdx in range(nSegs):
                PTVSegMask = PTVSegMaskList[segIdx]
                PTVSegBeamList = QihuiRyanResult[segIdx]
                PTVBeamMask = genBeamsMask(PTVSegMask, PTVSegBeamList)
                beamMaskFastDoseDict["PTVSeg{}".format(segIdx)] = PTVBeamMask
                print("{} done generating beam mask {}".format(patientName, segIdx))
            QihuiRyanMask = maskDict.copy()
            for key, value in beamMaskFastDoseDict.items():
                QihuiRyanMask[key] = value
            result_mask, result_header = nrrdGen(QihuiRyanMask)
            file = os.path.join(nrrdQihuiRyan, patientName + ".nrrd")
            nrrd.write(file, result_mask, result_header)
            print(file)


def nrrdGen(maskDict):
    def hex_to_rgb(hex_color):
        """Converts a color from hexadecimal format to RGB."""
        hex_color = hex_color.lstrip('#')
        result = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        result = np.array(result) / 255
        result = "{} {} {}".format(*result)
        return result

    nStructs = len(maskDict)
    dimensionOrg = maskDict["SKIN"].shape
    dimensionFlip = (dimensionOrg[2], dimensionOrg[1], dimensionOrg[0])
    fullDimension = (nStructs,) + dimensionFlip
    fullDimension = np.array(fullDimension, dtype=np.int64)

    space_directions = np.array([
        [np.nan, np.nan, np.nan],
        [isoRes, 0, 0],
        [0, isoRes, 0],
        [0, 0, isoRes]
    ])
    space_origin = np.array((0, 0, 0), dtype=np.float64)

    header_beginning = [
        ("type", "uint8"),
        ("dimension", 4),
        ("space", "left-posterior-superior"),
        ("sizes", fullDimension),
        ("space directions", space_directions),
        ("kinds", ["list", "domain", "domain", "domain"]),
        ("encoding", "gzip"),
        ("space origin", space_origin)
    ]

    header_ending = [
        ("Segmentation_ContainedRepresentationNames", "Binary labelmap|Closed surface|"),
        ("Segmentation_ConversionParameters",""),
        ("Segmentation_MasterRepresentation","Binary labelmap"),
        ("Segmentation_ReferenceImageExtentOffset", "0 0 0")
    ]
    extent_str = "0 {} 0 {} 0 {}".format(*dimensionFlip)

    header_middle = []
    seg_array = np.zeros(fullDimension, dtype=np.uint8)
    for i, entry in enumerate(maskDict.items()):
        name, localMask = entry
        seg_array[i, :, :, :] = np.transpose(localMask)
        key_header = "Segment{}_".format(i)
        
        color = hex_to_rgb(beamViewColorMap[name])
        header_middle.append((key_header + "Color", color))
        header_middle.append((key_header + "ColorAutoGenerated", "1"))
        header_middle.append((key_header + "Extent", extent_str))
        header_middle.append((key_header + "ID", name))
        header_middle.append((key_header + "LabelValue", "1"))
        header_middle.append((key_header + "Layer", i))
        header_middle.append((key_header + "Name", name))
        header_middle.append((key_header + "NameAutoGenerated", "1"))
        header_middle.append((key_header + "Tags",
            "DicomRtImport.RoiNumber:{}|TerminologyEntry:Segmentation category and type - 3D Slicer General Anatomy ".format(i+1)))

    header_middle.sort(key = lambda a: a[0])
    header_result = header_beginning + header_middle + header_ending
    header_result = OrderedDict(header_result)

    return seg_array, header_result


def genBeamsMask(PTVSegMask, beamAngles):
    # PTVSegMask order: (z, y, x)
    def rotateAroundAxisAtOriginRHS(p, axis, angle):
        # p: the vector to rotate
        # axis: the rotation axis
        # angle: in rad. The angle to rotate
        sint = np.sin(angle)
        cost = np.cos(angle)
        one_minus_cost = 1 - cost
        p_dot_axis = np.dot(p, axis)
        first_term_coeff = one_minus_cost * p_dot_axis
        result = first_term_coeff * axis + \
            cost * p + \
            sint * np.cross(axis, p)
        return result

    def inverseRotateBeamAtOriginRHS(vec, theta, phi, coll):
        # convert BEV coords to PVCS coords
        tmp = rotateAroundAxisAtOriginRHS(vec, np.array((0, 1, 0)), -(phi + coll))
        sptr = -np.sin(phi)
        cptr = np.cos(phi)
        rotation_axis = np.array((sptr, 0, cptr))
        result = rotateAroundAxisAtOriginRHS(tmp, rotation_axis, theta)
        return result

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

    directionsSet = []
    for angle_entry in beamAngles:
        axisBEV = np.array((0, 1, 0))
        axisPVCS = inverseRotateBeamAtOriginRHS(axisBEV,
            angle_entry[0], angle_entry[1], angle_entry[2])
        directionsSet.append(axisPVCS)  # order: (x, y, z)
    
    # calculate the coordinates
    coordsShape = PTVSegMask.shape + (3, )
    coords = np.zeros(coordsShape, dtype=float)
    axis_z = np.arange(coordsShape[0], dtype=float)
    axis_z = np.expand_dims(axis_z, axis=(1, 2))
    axis_y = np.arange(coordsShape[1], dtype=float)
    axis_y = np.expand_dims(axis_y, axis=(0, 2))
    axis_x = np.arange(coordsShape[2], dtype=float)
    axis_x = np.expand_dims(axis_x, axis=(0, 1))
    coords[:, :, :, 0] = axis_z
    coords[:, :, :, 1] = axis_y
    coords[:, :, :, 2] = axis_x
    PTVCentroid = calcCentroid(PTVSegMask)
    PTVCentroid = np.expand_dims(PTVCentroid, axis=(0, 1, 2))
    coords_minus_isocenter = coords - PTVCentroid

    beamsMask = None
    radius = 2
    barLength = 100
    for direction in directionsSet:
        # from (x, y, z) to (z, y, x)
        direction_ = np.array((direction[2], direction[1], direction[0]))
        direction_ = np.expand_dims(direction_, axis=(0, 1, 2))
        alongAxisProjection = np.sum(coords_minus_isocenter * direction_, axis=3, keepdims=True)
        perpendicular = coords_minus_isocenter - alongAxisProjection * direction_
        distance = np.linalg.norm(perpendicular, axis=3)

        alongAxisProjection = np.squeeze(alongAxisProjection)
        localMask = distance < radius
        localMask = np.logical_and(localMask, alongAxisProjection < 0)
        localMask = np.logical_and(localMask, alongAxisProjection > -barLength)

        if beamsMask is None:
            beamsMask = localMask
        else:
            beamsMask = np.logical_or(beamsMask, localMask)
    return beamsMask


def beamViewGrouping():
    """
    This function groups the beam view images
    """
    beamViewFastDoseFolder = os.path.join(sourceFolder, "nrrdFastDose")
    beamViewQihuiRyanFolder = os.path.join(sourceFolder, "nrrdQihuiRyan")
    targetHeight = 500
    targetWidth = 600
    FastDoseList = []
    QihuiRyanList = []
    imageShape = None
    for patient in patientList:
        beamViewFastDoseImage = os.path.join(beamViewFastDoseFolder, patient+".png")
        beamViewFastDoseImage = plt.imread(beamViewFastDoseImage)
        if imageShape is None:
            imageShape = beamViewFastDoseImage.shape
        else:
            assert imageShape == beamViewFastDoseImage.shape
        assert imageShape[1] >= targetWidth and imageShape[0] >= targetHeight
        idxBeginCol = int((imageShape[1] - targetWidth) / 2)
        idxBeginRow = int((imageShape[0] - targetHeight) / 2)
        beamViewFastDoseImage = beamViewFastDoseImage[idxBeginRow: idxBeginRow+targetHeight, idxBeginCol:idxBeginCol+targetWidth, :]
        FastDoseList.append(beamViewFastDoseImage)

        beamViewQihuiRyanImage = os.path.join(beamViewQihuiRyanFolder, patient+".png")
        beamViewQihuiRyanImage = plt.imread(beamViewQihuiRyanImage)
        assert(beamViewQihuiRyanImage.shape == imageShape)
        beamViewQihuiRyanImage = beamViewQihuiRyanImage[idxBeginRow: idxBeginRow+targetHeight, idxBeginCol:idxBeginCol+targetWidth, :]
        QihuiRyanList.append(beamViewQihuiRyanImage)
    
    patchShape = (targetHeight, targetWidth, 4)
    canvasShape = (targetHeight*4, targetWidth*4, 4)
    canvas = np.ones(canvasShape, dtype=np.float32)
    numPatients = len(patientList)
    for i in range(numPatients):
        rowIdx = i // 2
        colIdx = i % 2
        rowOffset = rowIdx * targetHeight
        colOffset = 2 * colIdx * targetWidth
        FastDosePatch = FastDoseList[i]
        canvas[rowOffset: rowOffset+targetHeight, colOffset: colOffset+targetWidth, :] = FastDosePatch

        QihuiRyanPatch = QihuiRyanList[i]
        colOffset = (2 * colIdx + 1) * targetWidth
        canvas[rowOffset: rowOffset+targetHeight, colOffset: colOffset+targetWidth, :] = QihuiRyanPatch
    
    fig, ax = plt.subplots(figsize=(canvas.shape[1]/100, canvas.shape[0]/100), dpi=100)
    ax.imshow(canvas)
    fontsize = 25
    rowOffset__ = 10
    for i in range(numPatients):
        legend = "Patient {}\nOurs".format(patientList[i])
        rowIdx = i // 2
        rowOffset = rowIdx * targetHeight + rowOffset__
        colIdx = i % 2
        colOffset = 2 * colIdx * targetWidth
        ax.text(colOffset, rowOffset, legend, ha="left", va="top", fontsize=fontsize)

        legend = "Patient {}\nBaseline".format(patientList[i])
        colOffset = (colIdx * 2 + 1) * targetWidth
        ax.text(colOffset, rowOffset, legend, ha="left", va="top", fontsize=fontsize)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    imageFile = os.path.join(figureFolder, "TCIASIBBeamsView.png")
    plt.savefig(imageFile)
    plt.close(fig)
    plt.clf()


def R50Calculation():
    """
    This function calculates R50
    """
    dosePercentile = 10
    content = ["| Patient | Ours | Baseline | Clinical |", "| - | - | - | - |"]
    for patientName in patientList:
        doseRef = os.path.join(rootFolder, patientName, "dose.bin")
        doseFastDose = os.path.join(sourceFolder, patientName, "FastDose", "plan1", "dose.bin")
        doseQihuiRyan = os.path.join(sourceFolder, patientName, "QihuiRyan", "doseRef.bin")
        doseRef = np.fromfile(doseRef, dtype=np.float32)
        doseFastDose = np.fromfile(doseFastDose, dtype=np.float32)
        doseQihuiRyan = np.fromfile(doseQihuiRyan, dtype=np.float32)

        ptv70 = os.path.join(rootFolder, patientName, "PlanMask", "PTV70.bin")
        ptv56 = os.path.join(rootFolder, patientName, "PlanMask", "PTV56.bin")
        ptv70 = np.fromfile(ptv70, dtype=np.uint8).astype(bool)
        ptv56 = np.fromfile(ptv56, dtype=np.uint8).astype(bool)
        ptvMerge = np.logical_or(ptv70, ptv56)
        ptvVoxels = np.sum(ptvMerge)

        body = os.path.join(rootFolder, patientName, "PlanMask", "SKIN.bin")
        body = np.fromfile(body, dtype=np.uint8).astype(bool)
        notBody = np.logical_not(body)

        doseRef[notBody] = 0
        doseFastDose[notBody] = 0
        doseQihuiRyan[notBody] = 0

        ptvDoseRef = doseRef[ptvMerge]
        ptvDoseThresh = np.percentile(ptvDoseRef, dosePercentile) * 0.5
        doseRefR50 = doseRef > ptvDoseThresh
        doseRefR50 = np.sum(doseRefR50) / ptvVoxels

        ptvDoseFastDose = doseFastDose[ptvMerge]
        ptvDoseThresh = np.percentile(ptvDoseFastDose, dosePercentile) * 0.5
        doseFastDoseR50 = doseFastDose > ptvDoseThresh
        doseFastDoseR50 = np.sum(doseFastDoseR50) / ptvVoxels

        ptvDoseQihuiRyan = doseQihuiRyan[ptvMerge]
        ptvDoseThresh = np.percentile(ptvDoseQihuiRyan, dosePercentile) * 0.5
        doseQihuiRyanR50 = doseQihuiRyan > ptvDoseThresh
        doseQihuiRyanR50 = np.sum(doseQihuiRyanR50) / ptvVoxels

        line = "| {} | {:.3f} | {:.3f} | {:.3f} |".format(
            patientName, doseFastDoseR50, doseQihuiRyanR50, doseRefR50)
        content.append(line)
    content = "\n".join(content)
    print(content)


def doseCalculationTimeComp():
    def doseMatTimeExtract(path):
        with open(path, "r") as f:
            lines = f.readlines()
        keyWords = "Dose calculation time: "
        lines = [a for a in lines if keyWords in a]
        assert len(lines) == 1
        line = lines[0]
        line = [a for a in line.split(" ") if "e+" in a]
        assert len(line) == 1
        return eval(line[0])
    
    def QihuiRyanDosecalcTime(path):
        with open(path, "r") as f:
            lines = f.readlines()
        keyWords = "real	"
        lines = [a for a in lines if keyWords in a]
        assert len(lines) == 1
        line = lines[0]
        line = line.split("	")
        line = line[1]
        result = ms2sec(line)
        return result
    
    def ms2sec(input: str):
        mIdx = input.find("m")
        sIdx = input.find("s")
        minutes = eval(input[:mIdx])
        seconds = eval(input[mIdx+1: sIdx])
        result = minutes * 60 + seconds
        return result
    
    content = ["| Patient | Ours | Baseline | Speedup |",
        "| - | - | - | - |"]
    FastDoseTimeList = []
    baselineTimeList = []
    speedupList = []
    for patientName in patientList:
        FastDoseFolder = os.path.join(sourceFolder, patientName, "FastDose")
        nMatrices = 4
        timeFastDose = 0
        for ii in range(nMatrices):
            file = os.path.join(FastDoseFolder, "doseMat{}.log".format(ii))
            localTime = doseMatTimeExtract(file)
            timeFastDose += localTime
        
        timeQihuiRyan = 0
        QihuiRyanFolder = os.path.join(sourceFolder, patientName, "QihuiRyan")
        for ii in range(nMatrices):
            file = os.path.join(QihuiRyanFolder, "preprocess{}".format(ii), "dosecalc.log")
            localTime = QihuiRyanDosecalcTime(file)
            timeQihuiRyan += localTime

        localSpeedup = timeQihuiRyan / timeFastDose
        newLine = "| {} | {:.3f} | {:.3f} | {:.3f} |".format(
            patientName, timeFastDose, timeQihuiRyan, localSpeedup)
        content.append(newLine)

        FastDoseTimeList.append(timeFastDose)
        baselineTimeList.append(timeQihuiRyan)
        speedupList.append(localSpeedup)
    
    FastDoseTimeAvg = np.mean(FastDoseTimeList)
    baselineTimeAvg = np.mean(baselineTimeList)
    speedupAvg = np.mean(speedupList)
    lastLine = "| Avg | {:.3f} | {:.3f} | {:.3f} |".format(
        FastDoseTimeAvg, baselineTimeAvg, speedupAvg, speedupAvg)
    content.append(lastLine)
    content = "\n".join(content)
    print(content)


def BOOTimeComp():
    """
    This function compares the dose calculation time between our method and the baseline method
    """
    BOOTimeFile = os.path.join(os.getcwd(), "manucode", "BOOTimeTCIABaseline.txt")
    with open(BOOTimeFile, "r") as f:
        BOOlines = f.readlines()
    BOOlines = [eval(a) for a in BOOlines]
    content = ["| Patient | Ours | Baseline | Speedup |",
        "| - | - | - | - |"]
    oursTimeList = []
    speedupList = []
    for i, patientName in enumerate(patientList):
        FastDoseOptLogFile = os.path.join(sourceFolder, patientName, "FastDose", "optimize.log")
        with open(FastDoseOptLogFile, "r") as f:
            lines = f.readlines()
        keyWords = "Optimization iterations:"
        lines = [a for a in lines if keyWords in a]
        assert len(lines) == 2
        line = lines[0]
        line = line.split(" ")
        keyWords = "e+"
        line = [a for a in line if keyWords in a]
        assert len(line) == 1
        FastDoseOptTime = eval(line[0])

        localSpeedup = BOOlines[i] / FastDoseOptTime
        currentLine = "| {} | {:.3f} | {:.3f} | {:.3f} |".format(
            patientName, FastDoseOptTime, BOOlines[i], localSpeedup)
        content.append(currentLine)
        oursTimeList.append(FastDoseOptTime)
        speedupList.append(localSpeedup)
    
    oursTimeAvg = np.mean(oursTimeList)
    baselineTimeAvg = np.mean(BOOlines)
    speedupAvg = np.mean(speedupList)

    lastLine = "| Avg | {:.3f} | {:.3f} | {:.3f} |".format(oursTimeAvg, baselineTimeAvg, speedupAvg)
    content.append(lastLine)
    content = "\n".join(content)
    print(content)


if __name__ == "__main__":
    # StructsInit()
    # DVH_comp()
    # drawDoseWash()
    # nrrdVerification()
    # beamViewGrouping()
    # R50Calculation()
    # doseCalculationTimeComp()
    BOOTimeComp()