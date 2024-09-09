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
        block.set_title(patientName, fontsize=20)
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

    figureFile = os.path.join(figureFolder, "FastDosePancreasCorrect.png")
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
        ptv = masks["PTV"]
        body = masks["SKIN"]

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
            densityAxial = density[z, :, :]
            doseAxial = doseArray[z, :, :]
            masksAxial = [(name, array[z, :, :]) for name, array in masks.items()]
            bodyAxial = body[z, :, :]
            axialImage = drawSlice(densityAxial, doseAxial, masksAxial, bodyAxial,
                ImageHeight, AxialWidth, colorMap, doseShowMax)
            
            densityCoronal = np.flip(density[:, y, :], axis=0)
            doseCoronal = np.flip(doseArray[:, y, :], axis=0)
            masksCoronal = [(name, np.flip(array[:, y, :], axis=0)) for name, array in masks.items()]
            bodyCoronal = np.flip(body[:, y, :], axis=0)
            coronalImage = drawSlice(densityCoronal, doseCoronal, masksCoronal, bodyCoronal,
                ImageHeight, CoronalWidth, colorMap, doseShowMax)
            
            densitySagittal = np.flip(density[:, :, x], axis=0)
            doseSagittal = np.flip(doseArray[:, :, x], axis=0)
            masksSagittal = [(name, np.flip(array[:, :, x], axis=0)) for name, array in masks.items()]
            bodySagittal = np.flip(body[:, :, x], axis=0)
            sagittalImage = drawSlice(densitySagittal, doseSagittal, masksSagittal, bodySagittal,
                ImageHeight, SagittalWidth, colorMap, doseShowMax)
            
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
    height, width, colorMap, doseShowMax):
    doseThresh = 10
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
        color = colorMap[name]
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


def nrrdGen():
    isoRes = 2.5  # mm
    structList = []
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)
        maskFolder = os.path.join(patientFolder, "InputMask")
        structsLocal = [a.split(".")[0] for a in os.listdir(maskFolder)]
        for a in structsLocal:
            if a not in structList:
                structList.append(a)
    ptv = "ROI"
    body = "SKIN"
    assert ptv in structList and body in structList
    structList.remove(ptv)
    structList.remove(body)
    structList.sort()
    structList.insert(0, ptv)
    structList.append(body)
    structList.append("beams")
    colorMap = {}
    for i, name in enumerate(structList):
        colorMap[name] = hex_to_rgb(colors[i])

    nrrdFastDose = os.path.join(sourceFolder, "nrrdFastDose")
    if not os.path.isdir(nrrdFastDose):
        os.mkdir(nrrdFastDose)
    nrrdQihuiRyan = os.path.join(sourceFolder, "nrrdQihuiRyan")
    if not os.path.isdir(nrrdQihuiRyan):
        os.mkdir(nrrdQihuiRyan)
    
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)
        dimension = os.path.join(patientFolder, "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = eval(dimension.replace(" ", ", "))  # (x, y, z)
        dimension_flip = np.flip(dimension)  # (z, y, x)

        maskFolder = os.path.join(patientFolder, "InputMask")
        structsLocal = [a.split(".")[0] for a in os.listdir(maskFolder)]

        nStructs = len(structsLocal)
        fullDimension = np.insert(dimension_flip, 0, nStructs).astype(np.int64)

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
        extent_str = "0 {} 0 {} 0 {}".format(*dimension_flip)

        header_middle = []
        seg_array = np.zeros(fullDimension, dtype=np.uint8)
        idx = 0
        for name in structList:
            if name not in structsLocal:
                continue
            localMask = os.path.join(maskFolder, name+".bin")
            localMask = np.fromfile(localMask, dtype=np.uint8)
            localMask = np.reshape(localMask, dimension_flip)
            seg_array[idx, :, :, :] = localMask
            
            key_header = "Segment{}_".format(idx)
            color = colorMap[name]
            header_middle.append((key_header + "Color", color))
            header_middle.append((key_header + "ColorAutoGenerated", "1"))
            header_middle.append((key_header + "Extent", extent_str))
            header_middle.append((key_header + "ID", name))
            header_middle.append((key_header + "LabelValue", "1"))
            header_middle.append((key_header + "Layer", idx))
            header_middle.append((key_header + "Name", name))
            header_middle.append((key_header + "NameAutoGenerated", "1"))
            header_middle.append((key_header + "Tags",
                "DicomRtImport.RoiNumber:{}|TerminologyEntry:Segmentation "
                    "category and type - 3D Slicer General Anatomy ".format(idx+1)))
            idx += 1
        header_middle.sort(key=lambda a: a[0])
        header_result = header_beginning + header_middle + header_ending
        header_result = OrderedDict(header_result)

        nrrdFile = os.path.join(nrrdFastDose, patientName + ".nrrd")
        nrrd.write(nrrdFile, seg_array, header_result)
        print(nrrdFile)
        break


if __name__ == "__main__":
    # DVH_comp()
    # drawDoseWash()
    nrrdGen()