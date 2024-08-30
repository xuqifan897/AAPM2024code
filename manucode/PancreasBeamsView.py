import os
import numpy as np
import nrrd
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import convolve
from skimage import measure
from io import BytesIO

rootFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas"
figureFolder = "/data/qifan/projects/AAPM2024/manufigures"
numPatients = 5
StructuresGlobal = None
isoRes = 2.5  # mm
doseShowMax = 30

def hex_to_rgb(hex_color):
    """Converts a color from hexadecimal format to RGB."""
    hex_color = hex_color.lstrip('#')
    result = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    result = np.array(result) / 255
    result = "{} {} {}".format(*result)
    return result

def StructsInit():
    global StructuresGlobal
    StructuresGlobal = []
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i + 1)
        maskFolder = os.path.join(rootFolder, patientName, "InputMask")
        StructuresLocal = [a.split(".")[0] for a in os.listdir(maskFolder)]
        for a in StructuresLocal:
            if a not in StructuresGlobal:
                StructuresGlobal.append(a)
    PTVName = "PTV"
    skinName = "SKIN"
    BeamsName = "BEAMS"
    assert PTVName in StructuresGlobal and skinName in StructuresGlobal
    StructuresGlobal.remove(PTVName)
    StructuresGlobal.remove(skinName)
    StructuresGlobal.sort()
    StructuresGlobal.append(skinName)
    StructuresGlobal.append(BeamsName)
    StructuresGlobal.insert(0, PTVName)

    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
    # colors = [hex_to_rgb(a) for a in colors]
    StructuresGlobal = {StructuresGlobal[i]: colors[i] for i in range(len(StructuresGlobal))}
    # print(StructuresGlobal)

def masks2nrrd():
    """
    This function converts the original binary masks into nrrd
    """
    for i in range(5, 6):
        patientName = "Patient{:03d}".format(i)
        patientFolder = os.path.join(rootFolder, patientName)
        dimensionFile = os.path.join(patientFolder, "FastDoseCorrect", "prep_output", "dimension.txt")
        with open(dimensionFile, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension = np.flip(dimension)
        
        MaskFolder = os.path.join(patientFolder, "InputMask")
        maskDict = {}
        for struct in StructuresGlobal:
            fileName = os.path.join(MaskFolder, struct + ".bin")
            if not os.path.isfile(fileName):
                continue
            maskArray = np.fromfile(fileName, dtype=np.uint8)
            maskArray = np.reshape(maskArray, dimension)
            maskDict[struct] = maskArray

        
        beamlistFastDose = os.path.join(rootFolder, "plansAngleCorrect",
            patientName, "FastDose", "plan1", "metadata.txt")
        with open(beamlistFastDose, "r") as f:
            beamsSelect = f.readlines()
        beamsSelect = beamsSelect[3]
        beamsSelect = beamsSelect.replace("  ", ", ")
        beamsSelect = eval(beamsSelect)

        beamsMask = genBeamsMask(patientName, maskDict["PTV"], beamsSelect)
        maskDict["BEAMS"] = beamsMask
        mask, header = nrrdGen(maskDict)
        file = os.path.join(rootFolder, "plansAngleCorrect", patientName, "beamMasksFastDose.nrrd")
        nrrd.write(file, mask, header)
        print(file)


def genBeamsMask(patientName, PTVMask, beamsSelect):
    # patientFolder = os.path.join(rootFolder, patientName)
    patientFolder = os.path.join(rootFolder, "plansAngleCorrect", patientName)
    beamListFile = os.path.join(patientFolder, "beamlist.txt")
    with open(beamListFile, "r") as f:
        angles = f.readlines()
    for i in range(len(angles)):
        currentLine = angles[i]
        currentLine = currentLine.replace(" ", ", ")
        currentLine = np.array(eval(currentLine))
        currentLine = currentLine * np.pi / 180  # convert degree to rad
        angles[i] = currentLine
    beamsSelect = [angles[i] for i in beamsSelect]

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
    
    directionsSelect = []
    for angle_entry in beamsSelect:
        axisBEV = np.array((0, 1, 0))
        axisPVCS = inverseRotateBeamAtOriginRHS(axisBEV,
            angle_entry[0], angle_entry[1], angle_entry[2])
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
    PTVCentroid = calcCentroid(PTVMask)
    PTVCentroid = np.expand_dims(PTVCentroid, axis=(0, 1, 2))
    coords_minus_isocenter = coords - PTVCentroid

    beamsMask = None
    radius = 2
    barLength = 100
    for direction in directionsSelect:
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

        
def nrrdGen(maskDict):
    if False:
        nrrdTemplate = "/data/qifan/FastDoseWorkplace/TCIAAdd/002/RT_exp3.nrrd"
        seg, header = nrrd.read(nrrdTemplate)
        type_ = header["type"]
        print(type_, type(type_))
        dimension_ = header["dimension"]
        print(dimension_, type(dimension_))
        space = header["space"]
        print(space, type(space))
        spaceDirections = header["space directions"]
        print(spaceDirections, type(spaceDirections), spaceDirections.dtype)
        sizes = header["sizes"]
        print(sizes, type(sizes), sizes.dtype)
        spaceOrigin = header["space origin"]
        print(spaceOrigin, type(spaceOrigin), spaceOrigin.dtype)
        Segmentation_ReferenceImageExtentOffset = header["Segmentation_ReferenceImageExtentOffset"]
        print(Segmentation_ReferenceImageExtentOffset, type(Segmentation_ReferenceImageExtentOffset))
        kinds = header["kinds"]
        print(kinds, type(kinds))

    nStructs = len(maskDict)
    dimensionOrg = maskDict["PTV"].shape
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
        
        color = StructuresGlobal[name]
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


def beamViewPancreasGroup():
    """
    This function groups individual beam view figures into one figure
    """
    figureFolderTopic = os.path.join(figureFolder, "beamViewPancreas")
    targetWidth = 600
    imageSet = []
    imagePatchShape = None
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i + 1)
        patientImage = os.path.join(figureFolderTopic, "beamView{}.png".format(patientName))
        patientImage = plt.imread(patientImage)
        widthOrg = patientImage.shape[1]
        idxBegin = int((widthOrg - targetWidth) / 2)
        patientImage = patientImage[:, idxBegin: idxBegin + targetWidth]
        imageSet.append(patientImage)
        if imagePatchShape is None:
            imagePatchShape = patientImage.shape
        else:
            assert imagePatchShape == patientImage.shape
    
    numImagesPerRow = 3
    numRows = int(np.ceil(len(imageSet) / numImagesPerRow))
    numAppend = numRows * numImagesPerRow - len(imageSet)
    appendImage = np.ones(imagePatchShape, dtype=imageSet[-1].dtype)
    imageSet.extend([appendImage] * numAppend)

    imageRows = []
    for i in range(numRows):
        localRow = imageSet[i * numImagesPerRow: (i+1) * numImagesPerRow]
        localRow = np.concatenate(localRow, axis=1)
        imageRows.append(localRow)
    fullImage = np.concatenate(imageRows, axis=0)
    fig, ax = plt.subplots(figsize=(fullImage.shape[1] / 100, fullImage.shape[0] / 100), dpi=100)
    ax.imshow(fullImage)
    singleImageHeight, singleImageWidth = imageSet[-1].shape[:2]
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        rowIdx = i // numImagesPerRow
        colIdx = i % numImagesPerRow
        rowDisp = rowIdx * singleImageHeight + 50
        colDisp = colIdx * singleImageWidth + 30
        ax.text(colDisp, rowDisp, patientName, fontsize=30, color="black")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    imageFile = os.path.join(figureFolder, "beamViewsPancreas.png")
    fig.savefig(imageFile)
    plt.close(fig)
    plt.clf()


def beamViewPancreasGroupCorrect():
    """
    Merge the beam view figures of both the FastDose group and QihuiRyan group
    """
    figureFolderTopic = os.path.join(figureFolder, "beamViewPancreasCorrect")
    FastDoseImages = []
    QihuiRyanImages = []
    imageShape = None
    imageDtype = None
    idxBegin = None
    targetWidth = 600
    for i in range(numPatients):
        file = os.path.join(figureFolderTopic, "Patient{:03d}BeamsFastDose.png".format(i+1))
        image = plt.imread(file)
        if imageShape is None:
            imageShape = image.shape
            idxBegin = int((imageShape[1] - targetWidth) / 2)
            imageShape = (imageShape[0], targetWidth, imageShape[2])
            imageDtype = imageDtype
        image = image[:, idxBegin: idxBegin+targetWidth, :]
        assert imageShape == image.shape
        FastDoseImages.append(image)

        file = os.path.join(figureFolderTopic, "Patient{:03d}BeamsQihuiRyan.png".format(i+1))
        image = plt.imread(file)
        image = image[:, idxBegin: idxBegin+targetWidth, :]
        QihuiRyanImages.append(image)
        assert imageShape == image.shape
    
    # construct blank images
    imageList = []
    for i in range(numPatients):
        imageUpper = FastDoseImages[i]
        imageLower = QihuiRyanImages[i]
        localImage = np.concatenate((imageUpper, imageLower), axis=0)
        imageList.append(localImage)
    
    # concatenate images together
    canvasShape = ((4*imageShape[0], 3*imageShape[1], imageShape[2]))
    canvas = np.ones(canvasShape, dtype=imageDtype)
    rowSize = 3
    for i in range(numPatients):
        rowIdx = i // rowSize
        colIdx = i % rowSize
        rowOffset = rowIdx * imageShape[0] * 2
        colOffset = colIdx * imageShape[1]
        canvas[rowOffset: rowOffset + 2 * imageShape[0],
            colOffset: colOffset + imageShape[1], :] = imageList[i]
    
    # write label
    canvasShape = canvas.shape
    fig, ax = plt.subplots(figsize=(canvasShape[1]/100, canvasShape[0]/100))
    ax.imshow(canvas)
    verticalDisplacement = 100
    horizontalDisplacement = 30
    for i in range(numPatients):
        rowIdx = i // rowSize
        colIdx = i % rowSize
        rowOffset = rowIdx * imageShape[0] * 2 + verticalDisplacement
        colOffset = colIdx * imageShape[1] + horizontalDisplacement
        text = "Patient{:03d}\nOurs".format(i+1)
        ax.text(colOffset, rowOffset, text, fontsize=30, color="black")

        rowOffset = (rowIdx * 2 + 1) * imageShape[0] + verticalDisplacement
        colOffset = colIdx * imageShape[1] + horizontalDisplacement
        text = "Patient{:03d}\nBaseline".format(i+1)
        ax.text(colOffset, rowOffset, text, fontsize=30, color="black")

    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    imageFile = os.path.join(figureFolder, "beamViewsPancreasCorrect.png")
    plt.savefig(imageFile)
    plt.close(fig)
    plt.clf()
    print(imageFile)


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


def drawDoseWashPancreas():
    topicFolder = os.path.join(figureFolder, "DoseWashPancreasCorrect")
    if not os.path.isdir(topicFolder):
        os.mkdir(topicFolder)
    
    ImageHeight = 80
    AxialWidth = 80
    CoronalWidth = 80
    SagittalWidth = 60
    
    targetFolder = os.path.join(rootFolder, "plansAngleCorrect")
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)

        # initialize masks
        dimension = os.path.join(rootFolder, patientName,
            "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension_flip = np.flip(dimension)

        density = os.path.join(rootFolder, patientName, "density_raw.bin")
        density = np.fromfile(density, dtype=np.uint16)
        density = np.reshape(density, dimension_flip)  # (z, y, x)
        maskFolder = os.path.join(rootFolder, patientName, "InputMask")
        maskDict = {}
        for file in os.listdir(maskFolder):
            name = file.split(".")[0]
            file_ = os.path.join(maskFolder, file)
            mask = np.fromfile(file_, dtype=np.uint8)
            mask = np.reshape(mask, dimension_flip)  # (z, y, x)
            maskDict[name] = mask
        
        patientFolder = os.path.join(targetFolder, patientName)
        doseFastDose = os.path.join(patientFolder, "FastDose", "plan1", "dose.bin")
        doseFastDose = np.fromfile(doseFastDose, dtype=np.float32)
        doseFastDose = np.reshape(doseFastDose, dimension_flip)  # (z, y, x)
        doseQihuiRyan = os.path.join(patientFolder, "QihuiRyan", "doseRef.bin")
        doseQihuiRyan = np.fromfile(doseQihuiRyan, dtype=np.float32)
        doseQihuiRyan = np.reshape(doseQihuiRyan, dimension_flip)

        # normalize
        ptv = "PTV"
        ptv = maskDict[ptv] > 0
        body = "SKIN"
        body = maskDict[body] > 0

        # mask dose
        doseFastDose[np.logical_not(body)] = 0
        doseQihuiRyan[np.logical_not(body)] = 0

        percentile = 10
        targetDose = 20
        ptvDoseFastDose = doseFastDose[ptv]
        threshFastDose = np.percentile(ptvDoseFastDose, percentile)
        doseFastDose *= targetDose / threshFastDose

        ptvDoseQihuiRyan = doseQihuiRyan[ptv]
        threshQihuiRyan = np.percentile(ptvDoseQihuiRyan, percentile)
        doseQihuiRyan *= targetDose / threshQihuiRyan
        
        # calculate centroid
        centroid = calcCentroid(ptv)  # (z, y, x)
        z, y, x = centroid.astype(int)
        if i == 1:
            z -= 3
        if i == 3:
            z -= 3
        bodyAxial = body[z, :, :]
        densityAxial = density[z, :, :]
        doseFastDoseAxial = doseFastDose[z, :, :]
        doseQihuiRyanAxial = doseQihuiRyan[z, :, :]
        maskSlices = [(a, b[z, :, :]) for a, b in maskDict.items()]
        axialPlotFastDose = drawSlice(densityAxial, doseFastDoseAxial, maskSlices,
            bodyAxial, ImageHeight, AxialWidth)
        axialPlotQihuiRyan = drawSlice(densityAxial, doseQihuiRyanAxial, maskSlices,
            bodyAxial, ImageHeight, AxialWidth)
        axialPlotCombined = np.concatenate((axialPlotFastDose, axialPlotQihuiRyan), axis=0)

        bodyCoronal = body[:, y, :]
        densityCoronal = density[:, y, :]
        doseFastDoseCoronal = doseFastDose[:, y, :]
        doseQihuiRyanCoronal = doseQihuiRyan[:, y, :]
        maskSlices = [(a, b[:, y, :]) for a, b in maskDict.items()]
        coronalPlotFastDose = drawSlice(densityCoronal, doseFastDoseCoronal, maskSlices,
            bodyCoronal, ImageHeight, CoronalWidth)
        coronalPlotFastDose = np.flip(coronalPlotFastDose, axis=0)
        coronalPlotQihuiRyan = drawSlice(densityCoronal, doseQihuiRyanCoronal, maskSlices,
            bodyCoronal, ImageHeight, CoronalWidth)
        coronalPlotQihuiRyan = np.flip(coronalPlotQihuiRyan, axis=0)
        coronalPlotCombined = np.concatenate((coronalPlotFastDose, coronalPlotQihuiRyan), axis=0)

        bodySagittal = body[:, :, x]
        densitySagittal = density[:, :, x]
        doseFastDoseSagittal = doseFastDose[:, :, x]
        doseQihuiRyanSagittal = doseQihuiRyan[:, :, x]
        maskSlices = [(a, b[:, :, x]) for a, b in maskDict.items()]
        sagittalPlotFastDose = drawSlice(densitySagittal, doseFastDoseSagittal, maskSlices,
            bodySagittal, ImageHeight, SagittalWidth)
        sagittalPlotFastDose = np.flip(sagittalPlotFastDose, axis=0)
        sagittalPlotQihuiRyan = drawSlice(densitySagittal, doseQihuiRyanSagittal, maskSlices,
            bodySagittal, ImageHeight, SagittalWidth)
        sagittalPlotQihuiRyan = np.flip(sagittalPlotQihuiRyan, axis=0)
        sagittalPlotCombined = np.concatenate((sagittalPlotFastDose, sagittalPlotQihuiRyan), axis=0)

        fullPlot = np.concatenate((axialPlotCombined, coronalPlotCombined, sagittalPlotCombined), axis=1)
        fullPlotFile = os.path.join(topicFolder, "doseWash{}.png".format(patientName))
        plt.imsave(fullPlotFile, fullPlot)
        print(fullPlotFile)


def drawSlice(densitySlice, doseSlice, maskSlice, bodySlice, height, width):
    doseThresh = 1
    maskCentroid = calcCentroid2d(bodySlice)
    densityCrop = crop_and_fill(densitySlice, maskCentroid, height, width)
    doseCrop = crop_and_fill(doseSlice, maskCentroid, height, width)
    maskSliceCrop = []
    for name, mask_slice in maskSlice:
        mask_slice_crop = crop_and_fill(mask_slice, maskCentroid, height, width)
        maskSliceCrop.append((name, mask_slice_crop))
    fig, ax = plt.subplots(figsize=(width/50, height/50), dpi=200)
    ax.imshow(densityCrop, cmap="gray", vmin=0, vmax=2000)
    for name, mask in maskSliceCrop:
        color = StructuresGlobal[name]
        contours = measure.find_contours(mask)
        for contour in contours:
            # ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=4)
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=0.5)
    ax.imshow(doseCrop, cmap="jet", vmin=0, vmax=doseShowMax, alpha=(doseCrop>doseThresh)*0.3)
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


def colorBarGen():
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

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    plt.clf()
    buf.seek(0)
    image = plt.imread(buf)
    buf.close()
    ImageSliced = image[:, -160:-40, :]
    if False:
        figureFile = os.path.join(figureFolder, "DoseWashSample", "colorbar.png")
        plt.imsave(figureFile, ImageSliced)
        print(figureFile)

    return ImageSliced


def doseWashCombine():
    # patients selected: Patient002 and Patient004
    topicFolder = os.path.join(figureFolder, "DoseWashPancreasCorrect")
    patient002 = os.path.join(topicFolder, "doseWashPatient002.png")
    patient002 = plt.imread(patient002)
    patient004 = os.path.join(topicFolder, "doseWashPatient004.png")
    patient004 = plt.imread(patient004)
    
    ImageHeight = 80 * 4
    AxialWidth = 80 * 4
    CoronalWidth = 80 * 4
    SagittalWidth = 60 * 4
    imageShape = (ImageHeight * 2, AxialWidth + CoronalWidth + SagittalWidth, 4)
    assert patient002.shape == imageShape and patient004.shape == imageShape
    imageConcat = np.concatenate((patient002, patient004), axis=0)
    concatShape = imageConcat.shape

    colorBar = colorBarGen()
    colorBarEnlarge = np.zeros((concatShape[0], colorBar.shape[1], 4), dtype=colorBar.dtype)
    colorBarEnlarge[:, :, -1] = 1
    offset = int((concatShape[0] - colorBar.shape[0]) / 2)
    colorBarEnlarge[offset: offset+colorBar.shape[0], :, :] = colorBar
    imageConcat = np.concatenate((imageConcat, colorBarEnlarge), axis=1)
    concatShape = imageConcat.shape

    fig, ax = plt.subplots(figsize=(concatShape[1]/100, concatShape[0]/100), dpi=200)
    ax.imshow(imageConcat)

    patientName = ["Patient002", "Patient002", "Patient004", "Patient004"]
    groupName = ["Ours", "Baseline"]
    for rowIdx in range(4):
        patName = patientName[rowIdx]
        groupIdx = rowIdx % len(groupName)
        grpName = groupName[groupIdx]
        text = patName + "\n" + grpName
        plt.text(0, rowIdx * ImageHeight, text, color="white", ha="left", va="top", fontsize=15)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    imageFile = os.path.join(figureFolder, "doseWashPancreasCorrect.png")
    plt.savefig(imageFile)
    plt.close(fig)
    plt.clf()


if __name__ == "__main__":
    StructsInit()
    # masks2nrrd()
    # beamViewPancreasGroup()
    # beamViewPancreasGroupCorrect()
    # drawDoseWashPancreas()
    doseWashCombine()