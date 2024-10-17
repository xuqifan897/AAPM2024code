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

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
sourceFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB"
numPatients = 5
isoRes = 2.5  # mm
figureFolder = "/data/qifan/projects/AAPM2024/manufigures/PancreasSIB"
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
        FastDoseDose = os.path.join(patientFolder, "FastDose", "plan1", "dose.bin")
        FastDoseDose = np.fromfile(FastDoseDose, dtype=np.float32)
        QihuiRyanDose = os.path.join(patientFolder, "QihuiRyan", "doseQihuiRyan.bin")
        QihuiRyanDose = np.fromfile(QihuiRyanDose, dtype=np.float32)
        
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


def drawDoseWash():
    doseWashFolder = os.path.join(figureFolder, "doseWash")
    if not os.path.isdir(doseWashFolder):
        os.mkdir(doseWashFolder)
    
    ImageHeight = 80
    AxialWidth = 80
    CoronalWidth = 80
    SagittalWidth = 80

    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)

        dimension = os.path.join(patientFolder, "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = eval(dimension.replace(" ", ", "))  # (x, y, z)
        dimension_flip = np.flip(dimension)  # (z, y, x)

        density = os.path.join(patientFolder, "density_raw.bin")
        density = np.fromfile(density, dtype=np.uint16)
        density = np.reshape(density, dimension_flip)

        clinicalDose = os.path.join(patientFolder, "doseNorm.bin")
        clinicalDose = np.fromfile(clinicalDose, dtype=np.float32)
        clinicalDose = np.reshape(clinicalDose, dimension_flip)
        FastDoseDose = os.path.join(patientFolder, "FastDose", "plan1", "dose.bin")
        FastDoseDose = np.fromfile(FastDoseDose, dtype=np.float32)
        FastDoseDose = np.reshape(FastDoseDose, dimension_flip)
        QihuiRyanDose = os.path.join(patientFolder, "QihuiRyan", "doseQihuiRyan.bin")
        QihuiRyanDose = np.fromfile(QihuiRyanDose, dtype=np.float32)
        QihuiRyanDose = np.reshape(QihuiRyanDose, dimension_flip)
        
        masks = {}
        relevantStructures = ["PTV", "Stomach_duo_planCT", "Bowel_sm_planCT",
            "kidney_left", "kidney_right", "liver", "SKIN"]
        colorMap = {}
        for i, struct in enumerate(relevantStructures):
            colorMap[struct] = colors[i]
        for name in relevantStructures:
            filename = name
            if name == "PTV":
                filename = "ROI"
            filename = os.path.join(patientFolder, "InputMask", filename+".bin")
            maskArray = np.fromfile(filename, dtype=np.uint8).astype(bool)
            maskArray = np.reshape(maskArray, dimension_flip)
            masks[name] = maskArray

        # normalize
        ptv = masks["PTV"].astype(bool)
        body = masks["SKIN"].astype(bool)

        # mask dose
        clinicalDose[np.logical_not(body)] = 0
        FastDoseDose[np.logical_not(body)] = 0
        QihuiRyanDose[np.logical_not(body)] = 0

        centroid = calcCentroid(ptv)  # (z, y, x)
        z, y, x = centroid.astype(int)
        
        # doseList = [clinicalDose, FastDoseDose, QihuiRyanDose]
        doseList = [("clinical", clinicalDose), ("FastDose", FastDoseDose),
            ("QihuiRyan", QihuiRyanDose)]
        imageList = []
        doseShowMax = np.max(clinicalDose)
        print(patientName)
        for name, doseArray in doseList:
            ptvDose = doseArray[ptv]
            ptvThresh = np.percentile(ptvDose, 5)
            doseWashThresh = 0.2 * ptvThresh  # dose wash threshold is set to be 20% of the ptv dose
            threshMask = doseArray > doseWashThresh

            densityAxial = density[z, :, :]
            doseAxial = doseArray[z, :, :]
            masksAxial = [(name, array[z, :, :]) for name, array in masks.items()]
            bodyAxial = body[z, :, :]
            threshMaskAxial = threshMask[z, :, :]
            axialImage = drawSlice(densityAxial, doseAxial, masksAxial, bodyAxial,
                ImageHeight, AxialWidth, colorMap, doseShowMax, threshMaskAxial)
            
            densityCoronal = np.flip(density[:, y, :], axis=0)
            doseCoronal = np.flip(doseArray[:, y, :], axis=0)
            masksCoronal = [(name, np.flip(array[:, y, :], axis=0)) for name, array in masks.items()]
            bodyCoronal = np.flip(body[:, y, :], axis=0)
            threshMaskCoronal = np.flip(threshMask[:, y, :], axis=0)
            coronalImage = drawSlice(densityCoronal, doseCoronal, masksCoronal, bodyCoronal,
                ImageHeight, CoronalWidth, colorMap, doseShowMax, threshMaskCoronal)
            
            densitySagittal = np.flip(density[:, :, x], axis=0)
            doseSagittal = np.flip(doseArray[:, :, x], axis=0)
            masksSagittal = [(name, np.flip(array[:, :, x], axis=0)) for name, array in masks.items()]
            bodySagittal = np.flip(body[:, :, x], axis=0)
            threshMaskSagittal = np.flip(threshMask[:, :, x], axis=0)
            sagittalImage = drawSlice(densitySagittal, doseSagittal, masksSagittal, bodySagittal,
                ImageHeight, SagittalWidth, colorMap, doseShowMax, threshMaskSagittal)
            
            ImageRow = np.concatenate((axialImage, coronalImage, sagittalImage), axis=1)
            imageList.append(ImageRow)
            print(name)
        patientImage = np.concatenate(imageList, axis=0)

        # generate colorbar
        colorBarLocal = colorBarGen(doseShowMax, patientImage.shape[0])
        patientImage = np.concatenate((patientImage, colorBarLocal), axis=1)

        patientImageFile = os.path.join(doseWashFolder, patientName + ".png")
        plt.imsave(patientImageFile, patientImage)
        print(patientImageFile, "\n")


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


def hex_to_rgb(hex_color):
    """Converts a color from hexadecimal format to RGB."""
    hex_color = hex_color.lstrip('#')
    result = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    result = np.array(result) / 255
    result = "{} {} {}".format(*result)
    return result


def nrrdVerification():
    nrrdFastDose = os.path.join(sourceFolder, "nrrdFastDose")
    if not os.path.isdir(nrrdFastDose):
        os.mkdir(nrrdFastDose)
    nrrdQihuiRyan = os.path.join(sourceFolder, "nrrdQihuiRyan")
    if not os.path.isdir(nrrdQihuiRyan):
        os.mkdir(nrrdQihuiRyan)
    structsList = []
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        maskFolder = os.path.join(sourceFolder, patientName, "InputMask")
        structsLocal = [a.split(".")[0] for a in os.listdir(maskFolder)]
        for a in structsLocal:
            if a not in structsList:
                structsList.append(a)
    ptv = "ROI"
    body = "SKIN"
    beams = "BEAMS"
    assert ptv in structsList and body in structsList
    structsList.remove(ptv)
    structsList.remove(body)
    structsList.sort()
    structsList.insert(0, ptv)
    structsList.append(beams)
    structsList.append(body)
    colorMap = {}
    for i, a in enumerate(structsList):
        colorMap[a] = hex_to_rgb(colors[i])
    
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)
        dimension = os.path.join(patientFolder, "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension_flip = np.flip(dimension)

        maskFolder = os.path.join(patientFolder, "InputMask")
        structs = [a.split(".")[0] for a in os.listdir(maskFolder)]
        maskDict = {}
        for struct in structs:
            maskArray = os.path.join(maskFolder, struct + ".bin")
            maskArray = np.fromfile(maskArray, dtype=np.uint8)
            maskArray = np.reshape(maskArray, dimension_flip)
            maskDict[struct] = maskArray

        validBeams = os.path.join(patientFolder, "FastDose", "beamlist.txt")
        with open(validBeams, "r") as f:
            validBeams = f.readlines()
        for j in range(len(validBeams)):
            line = validBeams[j]
            line = line.replace(" ", ", ")
            line = np.array(eval(line)) * np.pi / 180  # degree to radian
            validBeams[j] = line
        
        ptvMask = maskDict["ROI"]
        beamListFastDose = os.path.join(patientFolder, "FastDose", "plan1", "metadata.txt")
        with open(beamListFastDose, "r") as f:
            beamListFastDose = f.readlines()
        beamListFastDose = beamListFastDose[3]
        beamListFastDose = eval(beamListFastDose.replace("  ", ", "))
        beamListFastDose = [validBeams[k] for k in beamListFastDose]
        beamMaskFastDose = genBeamsMask(ptvMask, beamListFastDose)
        maskDictFastDose = maskDict.copy()
        maskDictFastDose[beams] = beamMaskFastDose
        maskFastDose, headerFastDose = nrrdGen(maskDictFastDose, colorMap, dimension_flip)
        file = os.path.join(nrrdFastDose, patientName + ".nrrd")
        nrrd.write(file, maskFastDose, headerFastDose)
        print(file)
        
        beamFileQihuiRyan = os.path.join(patientFolder, "QihuiRyan", "selected_angles.csv")
        with open(beamFileQihuiRyan, "r") as f:
            beamListQihuiRyan = f.readlines()
        beamListQihuiRyan = beamListQihuiRyan[1:]  # remove the header
        beamListQihuiRyan = [a for a in beamListQihuiRyan if a != "\n"]
        for j in range(len(beamListQihuiRyan)):
            line = eval(beamListQihuiRyan[j])
            line = line + (0.0000, )
            line = np.array(line[1:]) * np.pi / 180  # leave only the angles, and convert to radian
            beamListQihuiRyan[j] = line
        beamMaskQihuiRyan = genBeamsMask(ptvMask, beamListQihuiRyan)
        maskDictQihuiRyan = maskDict.copy()
        maskDictQihuiRyan[beams] = beamMaskQihuiRyan
        maskQihuiRyan, headerQihuiRyan = nrrdGen(maskDictQihuiRyan, colorMap, dimension_flip)
        file = os.path.join(nrrdQihuiRyan, patientName + ".nrrd")
        nrrd.write(file, maskQihuiRyan, headerQihuiRyan)
        print(file)


def nrrdGen(maskDict, colorMap, dimensionOrg):
    nStructs = len(maskDict)
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
        
        color = colorMap[name]
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


def rotateAroundAxisAtOrigin(p: np.ndarray, r: np.ndarray, t: float):
    # ASSUMES r IS NORMALIZED ALREADY and center is (0, 0, 0)
    # p - vector to rotate
    # r - rotation axis
    # t - rotation angle
    sptr = np.sin(t)
    cptr = np.cos(t)
    result = np.array((
        (-r[0]*(-r[0]*p[0] - r[1]*p[1] - r[2]*p[2]))*(1-cptr) + p[0]*cptr + (-r[2]*p[1] + r[1]*p[2])*sptr,
        (-r[1]*(-r[0]*p[0] - r[1]*p[1] - r[2]*p[2]))*(1-cptr) + p[1]*cptr + (+r[2]*p[0] - r[0]*p[2])*sptr,
        (-r[2]*(-r[0]*p[0] - r[1]*p[1] - r[2]*p[2]))*(1-cptr) + p[2]*cptr + (-r[1]*p[0] + r[0]*p[1])*sptr
    ))
    return result


def inverseRotateBeamAtOriginRHS(vec: np.ndarray, theta: float, phi: float, coll: float):
    tmp = rotateAroundAxisAtOrigin(vec, np.array((0., 1., 0.)), -(phi+coll))  # coll rotation + correction
    sptr = np.sin(-phi)
    cptr = np.cos(-phi)
    rotation_axis = np.array((sptr, 0., cptr))
    result = rotateAroundAxisAtOrigin(tmp, rotation_axis, theta)
    return result


def centroidCalc(ptv):
    ptv = ptv > 0
    totalVoxels = np.sum(ptv)
    
    ptvShape = ptv.shape
    xScale = np.arange(ptvShape[0])
    xScale = np.expand_dims(xScale, axis=(1, 2))
    xCoord = np.sum(ptv * xScale) / totalVoxels

    yScale = np.arange(ptvShape[1])
    yScale = np.expand_dims(yScale, axis=(0, 2))
    yCoord = np.sum(ptv * yScale) / totalVoxels

    zScale = np.arange(ptvShape[2])
    zScale = np.expand_dims(zScale, axis=(0, 1))
    zCoord = np.sum(ptv * zScale) / totalVoxels

    return np.array((xCoord, yCoord, zCoord))


def genBeamsMask(PTVMask, beamsSelect):
    directionsSelect = []
    for angle in beamsSelect:
        axisBEV = np.array((0, 1, 0))
        axisPVCS = inverseRotateBeamAtOriginRHS(axisBEV, angle[0], angle[1], angle[2])
        directionsSelect.append(axisPVCS)
    
    # calculate the coordinates
    coordsShape = PTVMask.shape + (3, )
    coords = np.zeros(coordsShape, dtype=float)
    axis_z = np.arange(coordsShape[0])
    axis_z = np.expand_dims(axis_z, axis=(1, 2))
    axis_y = np.arange(coordsShape[1])
    axis_y = np.expand_dims(axis_y, axis=(0, 2))
    axis_x = np.arange(coordsShape[2])
    axis_x = np.expand_dims(axis_x, axis=(0, 1))
    coords[:, :, :, 0] = axis_z
    coords[:, :, :, 1] = axis_y
    coords[:, :, :, 2] = axis_x
    PTVCentroid = centroidCalc(PTVMask)
    PTVCentroid = np.expand_dims(PTVCentroid, axis=(0, 1, 2))
    coords_minus_isocenter = coords - PTVCentroid

    beamsMask = None
    radius = 2
    barLength = 100
    for direction in directionsSelect:
        # from (x, y, z) to (z, y, x)
        direction = np.flip(direction)
        direction_ = np.expand_dims(direction, axis=(0, 1, 2))
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
    beamViewFastDoseFolder = os.path.join(sourceFolder, "beamViewFastDose")
    beamViewQihuiRyanFolder = os.path.join(sourceFolder, "beamViewQihuiRyan")
    targetWidth = 500
    FastDoseList = []
    QihuiRyanList = []
    imageShape = None
    for i in range(numPatients):
        beamViewFastDoseImage = os.path.join(beamViewFastDoseFolder, "Patient{:03d}.png".format(i+1))
        beamViewFastDoseImage = plt.imread(beamViewFastDoseImage)
        if imageShape is None:
            imageShape = beamViewFastDoseImage.shape
        else:
            assert imageShape == beamViewFastDoseImage.shape
        idxBegin = int((imageShape[1] - targetWidth) / 2)
        if i == 0:
            idxBegin -= 30
        beamViewFastDoseImage = beamViewFastDoseImage[:, idxBegin:idxBegin+targetWidth, :]
        FastDoseList.append(beamViewFastDoseImage)

        beamViewQihuiRyanImage = os.path.join(beamViewQihuiRyanFolder, "Patient{:03d}.png".format(i+1))
        beamViewQihuiRyanImage = plt.imread(beamViewQihuiRyanImage)
        assert(beamViewQihuiRyanImage.shape == imageShape)
        beamViewQihuiRyanImage = beamViewQihuiRyanImage[:, idxBegin:idxBegin+targetWidth, :]
        QihuiRyanList.append(beamViewQihuiRyanImage)

    patchShape = (imageShape[0], targetWidth, 4)
    canvasShape = (patchShape[0]*3, patchShape[1]*4, 4)
    canvas = np.ones(canvasShape, dtype=np.float32)
    for i in range(numPatients):
        rowIdx = i // 2
        colIdx = i % 2
        rowOffset = rowIdx * patchShape[0]
        colOffset = 2 * colIdx * patchShape[1]
        FastDosePatch = FastDoseList[i]
        canvas[rowOffset: rowOffset+patchShape[0], colOffset: colOffset+patchShape[1], :] = FastDosePatch
    
        QihuiRyanPatch = QihuiRyanList[i]
        colOffset = (2 * colIdx + 1) * patchShape[1]
        canvas[rowOffset: rowOffset+patchShape[0], colOffset: colOffset+patchShape[1], :] = QihuiRyanPatch
    
    # add legend
    fig, ax = plt.subplots(figsize=(canvas.shape[1]/100, canvas.shape[0]/100), dpi=100)
    ax.imshow(canvas)
    fontsize = 25
    rowOffset__ = 10
    for i in range(numPatients):
        legend = "Patient {:03d}\nOurs".format(i+1)
        rowIdx = i // 2
        rowOffset = rowIdx * patchShape[0] + rowOffset__
        colIdx = i % 2
        colOffset = colIdx * 2 * patchShape[1]
        ax.text(colOffset, rowOffset, legend, ha="left", va="top", fontsize=fontsize)

        legend = "Patient {:03d}\nBaseline".format(i+1)
        colOffset = (colIdx * 2 + 1) * patchShape[1]
        ax.text(colOffset, rowOffset, legend, ha="left", va="top", fontsize=fontsize)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis("off")
    imageFile = os.path.join(sourceFolder, "PancreasSIBBeamsView.png")
    plt.savefig(imageFile)
    plt.close(fig)
    plt.clf()


def R50Calculation():
    """
    This function calculates R 50
    """
    dosePercentile = 10
    content = ["| Patient | Ours | Baseline | Clinical |", "| - | - | - | - |"]
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)
        doseRef = os.path.join(patientFolder, "doseNorm.bin")
        doseFastDose = os.path.join(patientFolder, "FastDose", "plan1", "dose.bin")
        doseQihuiRyan = os.path.join(patientFolder, "QihuiRyan", "doseQihuiRyan.bin")
        doseRef = np.fromfile(doseRef, dtype=np.float32)
        doseFastDose = np.fromfile(doseFastDose, dtype=np.float32)
        doseQihuiRyan = np.fromfile(doseQihuiRyan, dtype=np.float32)

        # mask out body
        bodyMask = os.path.join(patientFolder, "InputMask", "SKIN.bin")
        bodyMask = np.fromfile(bodyMask, dtype=np.uint8).astype(bool)
        notBodyMask = np.logical_not(bodyMask)
        doseRef[notBodyMask] = 0
        doseFastDose[notBodyMask] = 0
        doseQihuiRyan[notBodyMask] = 0

        ptvMask = os.path.join(patientFolder, "InputMask", "ROI.bin")
        ptvMask = np.fromfile(ptvMask, dtype=np.uint8).astype(bool)
        ptvVoxels = np.sum(ptvMask)

        if True:
            ptvDoseRef = doseRef[ptvMask]
            ptvDoseThresh = np.percentile(ptvDoseRef, dosePercentile) * 0.5  # half of the prescription dose
            doseRefR50 = doseRef > ptvDoseThresh
            doseRefR50 = np.sum(doseRefR50) / ptvVoxels

            ptvDoseFastDose = doseFastDose[ptvMask]
            ptvDoseThresh = np.percentile(ptvDoseFastDose, dosePercentile) * 0.5
            doseFastDoseR50 = doseFastDose > ptvDoseThresh
            doseFastDoseR50 = np.sum(doseFastDoseR50) / ptvVoxels

            ptvDoseQihuiRyan = doseQihuiRyan[ptvMask]
            ptvDoseThresh = np.percentile(ptvDoseQihuiRyan, dosePercentile) * 0.5
            doseQihuiRyanR50 = doseQihuiRyan > ptvDoseThresh
            doseQihuiRyanR50 = np.sum(doseQihuiRyanR50) / ptvVoxels
        else:
            ptvDoseRef = doseRef[ptvMask]
            ptvDoseThresh = np.percentile(ptvDoseRef, dosePercentile)
            halfPTVDose = ptvDoseThresh * 0.5
            ptvVoxelsRef = np.sum(doseRef > ptvDoseThresh)
            halfPTVVoxels = np.sum(doseRef > halfPTVDose)
            doseRefR50 = halfPTVVoxels / ptvVoxelsRef

            ptvDoseFastDose = doseFastDose[ptvMask]
            ptvDoseThresh = np.percentile(ptvDoseFastDose, dosePercentile)
            halfPTVDose = ptvDoseThresh * 0.5
            ptvVoxelsFastDose = np.sum(doseFastDose > ptvDoseThresh)
            halfPTVVoxels = np.sum(doseFastDose > halfPTVDose)
            doseFastDoseR50 = halfPTVVoxels / ptvVoxelsFastDose

            ptvDoseQihuiRyan = doseQihuiRyan[ptvMask]
            ptvDoseThresh = np.percentile(ptvDoseQihuiRyan, dosePercentile)
            halfPTVDose = ptvDoseThresh * 0.5
            ptvVoxelsQihuiRyan = np.sum(doseQihuiRyan > ptvDoseThresh)
            halfPTVVoxels = np.sum(doseQihuiRyan > halfPTVDose)
            doseQihuiRyanR50 = halfPTVVoxels / ptvVoxelsQihuiRyan
            # print(np.sum(ptvMask), ptvVoxelsRef, ptvVoxelsFastDose, ptvVoxelsQihuiRyan)
            print(np.mean(doseRef), np.mean(doseFastDose), np.mean(doseQihuiRyan))

        line = "| {:03d} | {:.3f} | {:.3f} | {:.3f} |".format(i+1, doseFastDoseR50, doseQihuiRyanR50, doseRefR50)
        content.append(line)
    content = "\n".join(content)
    print(content)


def R50ExaminePlot():
    """
    This function does the sanity check of the R50 calculation
    """
    SanityCheckImageFolder = os.path.join(sourceFolder, "R50SanityCheck")
    if not os.path.isdir(SanityCheckImageFolder):
        os.mkdir(SanityCheckImageFolder)
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

        doseRef = os.path.join(patientFolder, "doseNorm.bin")
        doseRef = np.fromfile(doseRef, dtype=np.float32)
        doseRef = np.reshape(doseRef, dimension_flip)

        doseFastDose = os.path.join(patientFolder, "FastDose", "plan1", "dose.bin")
        doseFastDose = np.fromfile(doseFastDose, dtype=np.float32)
        doseFastDose = np.reshape(doseFastDose, dimension_flip)

        doseQihuiRyan = os.path.join(patientFolder, "QihuiRyan", "doseQihuiRyan.bin")
        doseQihuiRyan = np.fromfile(doseQihuiRyan, dtype=np.float32)
        doseQihuiRyan = np.reshape(doseQihuiRyan, dimension_flip)

        bodyMask = os.path.join(patientFolder, "InputMask", "SKIN.bin")
        bodyMask = np.fromfile(bodyMask, dtype=np.uint8) > 0
        bodyMask = np.reshape(bodyMask, dimension_flip)

        ptvMask = os.path.join(patientFolder, "InputMask", "ROI.bin")
        ptvMask = np.fromfile(ptvMask, dtype=np.uint8) > 0
        ptvMask = np.reshape(ptvMask, dimension_flip)

        # normalize prescription dose to 20 Gy
        prescriptionDose = 20
        doseShowMax = 50
        prescriptionPercentile = 10
        notBodyMask = np.logical_not(bodyMask)

        doseRefPTV = doseRef[ptvMask]
        doseRefThresh = np.percentile(doseRefPTV, prescriptionPercentile)
        doseRef *= prescriptionDose / doseRefThresh
        doseRef[notBodyMask] = 0

        doseFastDosePTV = doseFastDose[ptvMask]
        doseFastDoseThresh = np.percentile(doseFastDosePTV, prescriptionPercentile)
        doseFastDose *= prescriptionDose / doseFastDoseThresh
        doseFastDose[notBodyMask] = 0

        doseQihuiRyanPTV = doseQihuiRyan[ptvMask]
        doseQihuiRyanThresh = np.percentile(doseQihuiRyanPTV, prescriptionPercentile)
        doseQihuiRyan *= prescriptionDose / doseQihuiRyanThresh
        doseQihuiRyan[notBodyMask] = 0

        patientImageFolder = os.path.join(SanityCheckImageFolder, patientName)
        if not os.path.isdir(patientImageFolder):
            os.mkdir(patientImageFolder)
        dimZ = dimension_flip[0]

        doseList = [doseRef, doseFastDose, doseQihuiRyan]
        for j in range(dimZ):
            densitySlice = density[j, :, :]
            ptvSlice = ptvMask[j, :, :]
            ptvCounters = measure.find_contours(ptvSlice)

            fig = plt.figure(figsize=(12, 4))
            gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
            for k in range(3):
                current_block = fig.add_subplot(gs[0, k])
                current_block.imshow(densitySlice, vmin=0, vmax=1200, cmap="gray")

                doseSlice = doseList[k][j, :, :]
                current_block.imshow(doseSlice, vmin=0, vmax=doseShowMax, cmap="jet", alpha=0.3)
                for contour in ptvCounters:
                    current_block.plot(contour[:, 1], contour[:, 0],
                        linewidth=1, linestyle="--", color=colors[0])

                R50Region = doseSlice > prescriptionDose * 0.5
                R50RegionContours = measure.find_contours(R50Region)
                for contour in R50RegionContours:
                    current_block.plot(contour[:, 1], contour[:, 0],
                        linewidth=1, linestyle="--", color=colors[1])


            figureFile = os.path.join(patientImageFolder, "{:03d}.png".format(j))
            fig.tight_layout()
            fig.savefig(figureFile)
            plt.close(fig)
            plt.clf()
            print(figureFile)
        break


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
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)
        doseMat1Log = os.path.join(patientFolder, "FastDose", "doseMat1.log")
        doseMat1CalcTime = doseMatTimeExtract(doseMat1Log)
        doseMat2Log = os.path.join(patientFolder, "FastDose", "doseMat2.log")
        doseMat2CalcTime = doseMatTimeExtract(doseMat2Log)
        FastDoseTime = doseMat1CalcTime + doseMat2CalcTime

        baselineDosecalcTime = os.path.join(patientFolder, "QihuiRyan", "preprocess.log")
        baselineDosecalcTime = QihuiRyanDosecalcTime(baselineDosecalcTime)
        localSpeedup = baselineDosecalcTime / FastDoseTime
        newLine = "| {:03d} | {:.3f} | {:.3f} | {:.3f} |".format(i+1, FastDoseTime, baselineDosecalcTime, localSpeedup)
        content.append(newLine)
        FastDoseTimeList.append(FastDoseTime)
        baselineTimeList.append(baselineDosecalcTime)
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
    BOOTimeFile = os.path.join(os.getcwd(), "BOOTimePancreasSIBBaseline.txt")
    with open(BOOTimeFile, "r") as f:
        BOOlines = f.readlines()
    BOOlines = [eval(a) for a in BOOlines]
    content = ["| Patient | Ours | Baseline | Speedup |",
        "| - | - | - | - |"]
    oursTimeList = []
    speedupList = []
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join("/data/qifan/projects/FastDoseWorkplace/Pancreas", patientName)
        FastDoseOptLogFile = os.path.join(patientFolder, "FastDose", "optimize.log")
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
        currentLine = "| {:03d} | {:.3f} | {:.3f} | {:.3f} |".format(
            i+1, FastDoseOptTime, BOOlines[i], localSpeedup)
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
    # DVH_comp()
    # drawDoseWash()
    # nrrdVerification()
    # beamViewGrouping()
    R50Calculation()
    # R50ExaminePlot()
    # doseCalculationTimeComp()
    # BOOTimeComp()