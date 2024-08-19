import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import nrrd
from collections import OrderedDict
colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())

sourceFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas"
targetFolder = os.path.join(sourceFolder, "plansAngleCorrect")
figureFolder = "/data/qifan/projects/AAPM2024/manufigures"
manuFiguresFolder = "/data/qifan/projects/AAPM2024/manufigures"
numPatients = 5
prescriptionDose = 20
doseMaxShow = 25
isoRes = 2.5

StructureList = []
colorMap = {}
colorMapBeamsShow = {}

def DVH_comp():
    for i in range(numPatients):
        patientName = 'Patient{:03d}'.format(i+1)
        # prepare dimension
        dimension = os.path.join(sourceFolder, patientName,
            "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension_flip = np.flip(dimension)

        patientFolder = os.path.join(targetFolder, patientName)
        FastDoseFolder = os.path.join(patientFolder, "FastDose")
        doseFastDose = os.path.join(FastDoseFolder, 'plan1', 'dose.bin')
        doseFastDose = np.fromfile(doseFastDose, dtype=np.float32)
        doseFastDose = np.reshape(doseFastDose, dimension_flip)

        QihuiRyanFolder = os.path.join(patientFolder, "QihuiRyan")
        doseQihuiRyan = os.path.join(QihuiRyanFolder, 'doseRef.bin')
        doseQihuiRyan = np.fromfile(doseQihuiRyan, dtype=np.float32)
        doseQihuiRyan = np.reshape(doseQihuiRyan, dimension_flip)

        StructureList = os.path.join(FastDoseFolder, "StructureInfo.csv")
        with open(StructureList, "r") as f:
            StructureList = f.readlines()
        StructureList = StructureList[1:]  # remove the first line
        StructureList = [a.split(",")[0] for a in StructureList]
        # remove the ring structure
        ringStruct = "RingStructure"
        if ringStruct in StructureList:
            StructureList.remove(ringStruct)
        
        maskDict = {}
        maskFolder = os.path.join(sourceFolder, patientName, "InputMask")
        for name in StructureList:
            mask = os.path.join(maskFolder, name + ".bin")
            mask = np.fromfile(mask, dtype=np.uint8)
            mask = np.reshape(mask, dimension_flip)
            maskDict[name] = mask
        
        # normalize
        PTVMask = maskDict["PTV"].astype(bool)
        PTVDoseFastDose = doseFastDose[PTVMask]
        threshFastDose = np.percentile(PTVDoseFastDose, 10)
        doseFastDose *= prescriptionDose / threshFastDose

        PTVDoseQihuiRyan = doseQihuiRyan[PTVMask]
        threshQihuiRyan = np.percentile(PTVDoseQihuiRyan, 10)
        doseQihuiRyan *= prescriptionDose / threshQihuiRyan

        # make DVH
        fig, ax = plt.subplots()
        for idx, pair in enumerate(maskDict.items()):
            color = colors[idx]
            name, mask = pair
            mask = mask.astype(bool)
            structDoseFastDose = doseFastDose[mask].copy()
            structDoseFastDose = np.sort(structDoseFastDose)
            structDoseFastDose = np.insert(structDoseFastDose, 0, 0)

            structDoseQihuiRyan = doseQihuiRyan[mask].copy()
            structDoseQihuiRyan = np.sort(structDoseQihuiRyan)
            structDoseQihuiRyan = np.insert(structDoseQihuiRyan, 0, 0)

            y_axis = np.linspace(100, 0, np.sum(mask)+1)
            ax.plot(structDoseFastDose, y_axis, color=color, label=name)
            ax.plot(structDoseQihuiRyan, y_axis, color=color, linestyle="--")
        ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9))
        ax.set_xlim(0, 30)
        ax.set_xlabel("Dose (Gy)")
        ax.set_ylabel("Percentile")
        ax.set_title(patientName + " DVH")
        imageFile = os.path.join(FastDoseFolder, "DVHComp_" + patientName + ".png")
        plt.savefig(imageFile)
        plt.close(fig)
        plt.clf()
        print(patientName)


def StructsInit():
    global StructureList
    global colorMap
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        sourcePatientFolder = os.path.join(sourceFolder, patientName)
        maskFolder = os.path.join(sourcePatientFolder, "InputMask")
        structures = [a.split(".")[0] for a in os.listdir(maskFolder)]
        for a in structures:
            if a not in StructureList:
                StructureList.append(a)
    skinName = "SKIN"
    assert skinName in StructureList
    StructureList.remove(skinName)
    for i, name in enumerate(StructureList):
        colorMap[name] = colors[i]

    StructureListBeamsShow = StructureList + ["SKIN", "Beams"]
    for i, name in enumerate(StructureListBeamsShow):
        colorMapBeamsShow[name] = colors[i]


def DVH_plot():
    rowSize = 4
    colSize = 3
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(colSize, rowSize, width_ratios=[0.2, 5, 5, 5],
        height_ratios=[4, 4, 0.2])

    # create the common y label
    ylabel_block = fig.add_subplot(gs[:-1, 0])
    ylabel_block.text(0.9, 0.5, "Fractional Volume (%)", ha="center", va="center",
        rotation="vertical", fontsize=20)
    ylabel_block.axis("off")
    
    # create the common x label
    xlabel_block = fig.add_subplot(gs[-1, 1:])
    xlabel_block.text(0.5, 0.9, "Dose (Gy)", ha="center", va="center", fontsize=20)
    xlabel_block.axis("off")

    # create the DVH plots
    localRowSize = rowSize - 1
    localColSize = colSize - 1
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        sourcePatientFolder = os.path.join(sourceFolder, patientName)
        targetPatientFolder = os.path.join(targetFolder, patientName)
        inputFolder = os.path.join(sourcePatientFolder, "FastDose", "prep_output")
        dimension = os.path.join(inputFolder, "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension_flip = np.flip(dimension)

        MaskFolder = os.path.join(sourcePatientFolder, "InputMask")
        StructuresLocal = [b for a in os.listdir(MaskFolder) if
            (b:=a.split(".")[0]) in StructureList]
        maskDict = {}
        for struct in StructuresLocal:
            maskFile = os.path.join(MaskFolder, "{}.bin".format(struct))
            maskArray = np.fromfile(maskFile, dtype=np.uint8)
            maskArray = np.reshape(maskArray, dimension_flip)
            maskDict[struct] = maskArray
        
        doseFastDose = os.path.join(targetPatientFolder, "FastDose", "plan1", "dose.bin")
        doseFastDose = np.fromfile(doseFastDose, dtype=np.float32)
        doseFastDose = np.reshape(doseFastDose, dimension_flip)
        doseQihuiRyan = os.path.join(targetPatientFolder, "QihuiRyan", "doseRef.bin")
        doseQihuiRyan = np.fromfile(doseQihuiRyan, dtype=np.float32)
        doseQihuiRyan = np.reshape(doseQihuiRyan, dimension_flip)

        # normalize
        percentile_value = 10
        primaryPTVName = "PTV"
        assert primaryPTVName in maskDict
        ptvMask = maskDict[primaryPTVName].astype(bool)
        ptvDose = doseFastDose[ptvMask]
        thresh = np.percentile(ptvDose, percentile_value)
        doseFastDose *= prescriptionDose / thresh
        ptvDose = doseQihuiRyan[ptvMask]
        thresh = np.percentile(ptvDose, percentile_value)
        doseQihuiRyan *= prescriptionDose / thresh

        rowIdx = i % localRowSize + 1
        colIdx = i // localRowSize
        assert colIdx < localColSize, "Figure index ({}, {}) error".format(rowIdx, colIdx)
        block = fig.add_subplot(gs[colIdx, rowIdx])

        for name, mask in maskDict.items():
            color = colorMap[name]
            mask = mask.astype(bool)
            structDoseFastDose = doseFastDose[mask]
            structDoseFastDose = np.sort(structDoseFastDose)
            structDoseFastDose = np.insert(structDoseFastDose, 0, 0)

            structDoseQihuiRyan = doseQihuiRyan[mask]
            structDoseQihuiRyan = np.sort(structDoseQihuiRyan)
            structDoseQihuiRyan = np.insert(structDoseQihuiRyan, 0, 0)

            y_axis = np.linspace(100, 0, np.sum(mask)+1)
            block.plot(structDoseFastDose, y_axis, color=color, linewidth=2.0)
            block.plot(structDoseQihuiRyan, y_axis, color=color, linewidth=2.0, linestyle="--")
            print(name)
        
        block.set_xlim(0, doseMaxShow)
        block.tick_params(axis="x", labelsize=16)
        block.tick_params(axis="y", labelsize=16)
        block.set_title("Patient {:03d}".format(i+1), fontsize=20)
    
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

    figureFile = os.path.join(figureFolder, "FastDosePancreasCorrect.png")
    plt.savefig(figureFile)
    figureFile = os.path.join(figureFolder, "FastDosePancreasCorrect.eps")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()


def masks2nrrd():
    FastDoseBeamFolder = os.path.join(targetFolder, "FastDoseBeams")
    if not os.path.isdir(FastDoseBeamFolder):
        os.mkdir(FastDoseBeamFolder)
    QihuiRyanBeamFolder = os.path.join(targetFolder, "QihuiRyanBeams")
    if not os.path.isdir(QihuiRyanBeamFolder):
        os.mkdir(QihuiRyanBeamFolder)
    
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        sourcePatientFolder = os.path.join(sourceFolder, patientName)
        targetPatientFolder = os.path.join(targetFolder, patientName)
        inputFolder = os.path.join(sourcePatientFolder, "FastDose", "prep_output")
        dimension = os.path.join(inputFolder, "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension_flip = np.flip(dimension)

        MaskFolder = os.path.join(sourcePatientFolder, "InputMask")
        StructuresLocal = [b for a in os.listdir(MaskFolder) if
            (b:=a.split(".")[0]) in StructureList]
        bodyName = "SKIN"
        StructuresLocal.append(bodyName)
        maskDict = {}
        for struct in StructuresLocal:
            maskFile = os.path.join(MaskFolder, "{}.bin".format(struct))
            maskArray = np.fromfile(maskFile, dtype=np.uint8)
            maskArray = np.reshape(maskArray, dimension_flip)
            maskDict[struct] = maskArray
        ptvName = "PTV"
        assert ptvName in maskDict
        ptvMask = maskDict[ptvName]

        # find the beam angles
        beamList = os.path.join(targetPatientFolder, "beamlist.txt")
        with open(beamList, "r") as f:
            beamList = f.readlines()
        beamList = [np.array(eval(a.replace(" ", ", "))) * np.pi / 180 for a in beamList]
        
        selectedBeamsFastDose = os.path.join(targetPatientFolder, "FastDose", "plan1", "metadata.txt")
        with open(selectedBeamsFastDose, "r") as f:
            selectedBeamsFastDose = f.readlines()
        selectedBeamsFastDose = selectedBeamsFastDose[3]
        selectedBeamsFastDose = selectedBeamsFastDose.replace("  ", ", ")
        selectedBeamsFastDose = eval(selectedBeamsFastDose)
        selectedBeamsFastDose = [beamList[idx] for idx in selectedBeamsFastDose]
        selectedBeamsFastDose = genBeamsMask(ptvMask, selectedBeamsFastDose)
        maskDictFastDose = maskDict.copy()
        maskDictFastDose["Beams"] = selectedBeamsFastDose
        seg_array, header_result = nrrdGen(maskDictFastDose)
        nrrdFile = os.path.join(FastDoseBeamFolder, "{}.nrrd".format(patientName))
        nrrd.write(nrrdFile, seg_array, header_result)

        selectedBeamsQihuiRyan = os.path.join(targetPatientFolder, "QihuiRyan", "selected_angles.csv")
        with open(selectedBeamsQihuiRyan, "r") as f:
            selectedBeamsQihuiRyan = f.readlines()
        # remove the first and last line
        selectedBeamsQihuiRyan = selectedBeamsQihuiRyan[1: -1]
        for j in range(len(selectedBeamsQihuiRyan)):
            line = selectedBeamsQihuiRyan[j]
            line = eval(line)
            line = np.array(line[1:])
            line *= np.pi / 180
            line = np.append(line, 0)
            selectedBeamsQihuiRyan[j] = line
        selectedBeamsQihuiRyan = genBeamsMask(ptvMask, selectedBeamsQihuiRyan)
        maskDictQihuiRyan = maskDict.copy()
        maskDictQihuiRyan["Beams"] = selectedBeamsQihuiRyan
        seg_array, header_result = nrrdGen(maskDictQihuiRyan)
        nrrdFile = os.path.join(QihuiRyanBeamFolder, "{}.nrrd".format(patientName))
        nrrd.write(nrrdFile, seg_array, header_result)
        print(patientName)


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


def nrrdGen(maskDict):
    def hex_to_rgb(hex_color):
        """Converts a color from hexadecimal format to RGB."""
        hex_color = hex_color.lstrip('#')
        result = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        result = np.array(result) / 255
        result = "{} {} {}".format(*result)
        return result
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
        
        color = hex_to_rgb(colorMapBeamsShow[name])
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


if __name__ == "__main__":
    StructsInit()
    # DVH_comp()
    # DVH_plot()
    masks2nrrd()