import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import nrrd
from collections import OrderedDict

sourceFolder = "/data/qifan/projects/FastDoseWorkplace/TCIAAdd"
figureFolder = "/data/qifan/projects/AAPM2024/manufigures"
targetFolder = os.path.join(sourceFolder, "plansAngleCorrect")
patients = ["002", "003", "009", "013", "070", "125", "132", "190"]

StructureList = []
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

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
colors_skip = [11, 13, 14, 16, 18]
idx = 18
for i in colors_skip:
    colors[i] = colors[idx]
    idx += 1
colorMap = {}

isoRes = 2.5
doseShowMax = 95

def StructsInit():
    """
    This function is to generate a coherent structure list for all patients
    """
    global StructureList, colorMap
    for patient in patients:
        if ".txt" in patient:
            continue
        patientFolder = os.path.join(sourceFolder, patient)
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

    # bring PTV70 and PTV56 forward
    StructureList.remove("PTV70")
    StructureList.remove("PTV56")
    StructureList.insert(0, "PTV56")
    StructureList.insert(0, "PTV70")

    # add the four sets of beams for the four isocenters
    additional = ["SKIN"] + ["PTVSeg{}".format(i) for i in range(4)]
    allStructs = StructureList + additional
    for i in range(len(allStructs)):
        colorMap[allStructs[i]] = colors[i]


def DVH_plot_single_patient():
    patientName = "002"
    for patientName in patients:
        sourcePatientFolder = os.path.join(sourceFolder, patientName)
        targetPatientFolder = os.path.join(targetFolder, patientName)
        inputFolder = os.path.join(sourcePatientFolder, "FastDose", "prep_output")
        dimension = os.path.join(inputFolder, "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = np.array(eval(dimension))
        dimension_flip = np.flip(dimension)

        MaskFolder = os.path.join(sourcePatientFolder, "PlanMask")
        StructuresLocal = [b for a in os.listdir(MaskFolder) if
            (b:=a.split(".")[0]).replace(" ", "") not in exclude and "+" not in a]
        maskDict = {}
        for struct in StructuresLocal:
            maskFile = os.path.join(MaskFolder, "{}.bin".format(struct))
            maskArray = np.fromfile(maskFile, dtype=np.uint8)
            maskArray = np.reshape(maskArray, dimension_flip)
            name = struct.replace(" ", "")
            if name in ConvergeReverse:
                name = ConvergeReverse[name]
            maskDict[name] = maskArray
        
        expFolder = os.path.join(targetPatientFolder, "FastDose", "plan1")
        doseExp = os.path.join(expFolder, "dose.bin")
        doseExp = np.fromfile(doseExp, dtype=np.float32)
        doseExp = np.reshape(doseExp, dimension_flip)

        doseRef = os.path.join(sourcePatientFolder, "dose.bin")
        doseRef = np.fromfile(doseRef, dtype=np.float32)
        doseRef = np.reshape(doseRef, dimension_flip)

        # normalize
        percentile_value = 10
        primaryPTVName = "PTV70"
        assert primaryPTVName in maskDict
        primaryMask = maskDict[primaryPTVName].astype(bool)
        thresh = np.percentile(doseExp[primaryMask], percentile_value)
        doseExp *= 70 / thresh
        thresh = np.percentile(doseRef[primaryMask], percentile_value)
        doseRef *= 70 / thresh

        fig, ax = plt.subplots()
        for name, mask in maskDict.items():
            color = colorMap[name]
            mask = mask.astype(bool)
            structDoseExp = doseExp[mask]
            structDoseExp = np.sort(structDoseExp)
            structDoseExp = np.insert(structDoseExp, 0, 0.0)

            structDoseRef = doseRef[mask]
            structDoseRef = np.sort(structDoseRef)
            structDoseRef = np.insert(structDoseRef, 0, 0.0)

            y_axis = np.linspace(100, 0, np.sum(mask)+1)
            ax.plot(structDoseExp, y_axis, color=color, label=name)
            ax.plot(structDoseRef, y_axis, color=color, linestyle="--")

        ax.legend()
        ax.set_xlabel("Dose (Gy)")
        ax.set_ylabel("Percentile (%)")
        file = os.path.join(targetPatientFolder, "DVH{}.png".format(patientName))
        plt.savefig(file)
        plt.close(fig)
        plt.clf()
        print(file)


def masks2nrrd():
    """
    This function converts the original binary masks into nrrd
    """
    numIsocenters = 4
    for patientName in patients:
        sourcePatientFolder = os.path.join(sourceFolder, patientName)
        targetPatientFolder = os.path.join(targetFolder, patientName)
        inputFolder = os.path.join(sourcePatientFolder, "FastDose", "prep_output")
        dimension = os.path.join(inputFolder, "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension_flip = np.flip(dimension)

        maskFolder = os.path.join(sourcePatientFolder, "PlanMask")
        maskDict = {}
        for struct in StructureList + ["SKIN"]:
            fileName = os.path.join(maskFolder, struct + ".bin")
            if not os.path.isfile(fileName):
                continue
            maskArray = np.fromfile(fileName, dtype=np.uint8)
            maskArray = np.reshape(maskArray, dimension_flip)
            maskDict[struct] = maskArray
        
        # there are 4 isocenters
        isocenterMapping = {}
        cumuIdx = 0
        for segIdx in range(numIsocenters):
            beamListFile = os.path.join(targetPatientFolder, "beamlist{}.txt".format(segIdx))
            with open(beamListFile, "r") as f:
                lines = f.readlines()
            nBeams = len(lines)
            for i in range(nBeams):
                lineContent = lines[i]
                lineContent = lineContent.replace(" ", ", ")
                lineContent = np.array(eval(lineContent)) * np.pi / 180  # convert degree to rad
                isocenterMapping[cumuIdx] = (segIdx, lineContent)
                cumuIdx += 1
        
        beamList = os.path.join(targetPatientFolder, "FastDose", "plan1", "metadata.txt")
        with open(beamList, "r") as f:
            beamList = f.readlines()
        beamList = beamList[3]
        beamList = beamList.replace("  ", ", ")
        beamList = eval(beamList)
        
        for segIdx in range(numIsocenters):
            # filter out the beams that belongs to the isocenter
            currentIsocenterList = []
            for idx in beamList:
                if isocenterMapping[idx][0] == segIdx:
                    currentIsocenterList.append(isocenterMapping[idx][1])
            
            # then generate masks for the beams
            PTVSegMaskName = os.path.join(maskFolder, "PTVSeg{}.bin".format(segIdx))
            PTVSegMask = np.fromfile(PTVSegMaskName, dtype=np.uint8)
            PTVSegMask = np.reshape(PTVSegMask, dimension_flip)
            PTVBeamsMask = genBeamsMask(PTVSegMask, currentIsocenterList)
            maskDict["PTVSeg{}".format(segIdx)] = PTVBeamsMask

        mask, header = nrrdGen(maskDict)
        file = os.path.join(targetPatientFolder, "beamMasks{}.nrrd".format(patientName))
        nrrd.write(file, mask, header)
        print(file)


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
        
        color = hex_to_rgb(colorMap[name])
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


def DVH_plot():
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
        sourcePatientFolder = os.path.join(sourceFolder, patients[i])
        targetPatientFolder = os.path.join(targetFolder, patients[i])
        inputFolder = os.path.join(sourcePatientFolder, "FastDose", "prep_output")
        dimension = os.path.join(inputFolder, "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension_flip = np.flip(dimension)

        MaskFolder = os.path.join(sourcePatientFolder, "PlanMask")
        StructuresLocal = [b for a in os.listdir(MaskFolder) if
            (b:=a.split(".")[0]).replace(" ", "") not in exclude and "+" not in a]
        maskDict = {}
        for struct in StructuresLocal:
            maskFile = os.path.join(MaskFolder, "{}.bin".format(struct))
            maskArray = np.fromfile(maskFile, dtype=np.uint8)
            maskArray = np.reshape(maskArray, dimension_flip)
            name = struct.replace(" ", "")
            if name in ConvergeReverse:
                name = ConvergeReverse[name]
            maskDict[name] = maskArray
        
        expFolder = os.path.join(targetPatientFolder, "FastDose", "plan1")
        doseExp = os.path.join(expFolder, "dose.bin")
        doseExp = np.fromfile(doseExp, dtype=np.float32)
        doseExp = np.reshape(doseExp, dimension_flip)

        doseRef = os.path.join(sourcePatientFolder, "dose.bin")
        doseRef = np.fromfile(doseRef, dtype=np.float32)
        doseRef = np.reshape(doseRef, dimension_flip)

        # normalize
        percentile_value = 10
        primaryPTVName = "PTV70"
        assert primaryPTVName in maskDict
        primaryMask = maskDict[primaryPTVName].astype(bool)
        thresh = np.percentile(doseExp[primaryMask], percentile_value)
        doseExp *= 70 / thresh
        thresh = np.percentile(doseRef[primaryMask], percentile_value)
        doseRef *= 70 / thresh

        rowIdx = i % localRowSize + 1
        colIdx = i // localRowSize
        assert colIdx < localColSize, "Figure index ({}, {}) error".format(rowIdx, colIdx)
        block = fig.add_subplot(gs[colIdx, rowIdx])

        for name, mask in maskDict.items():
            color = colorMap[name]
            mask = mask.astype(bool)
            structDoseExp = doseExp[mask]
            structDoseExp = np.sort(structDoseExp)
            structDoseExp = np.insert(structDoseExp, 0, 0.0)

            structDoseRef = doseRef[mask]
            structDoseRef = np.sort(structDoseRef)
            structDoseRef = np.insert(structDoseRef, 0, 0.0)

            y_axis = np.linspace(100, 0, np.sum(mask)+1)
            block.plot(structDoseExp, y_axis, color=color, linewidth=2.0)
            block.plot(structDoseRef, y_axis, color=color, linewidth=2.0, linestyle="--")
            print(name)
        
        block.set_xlim(0, doseShowMax)
        block.tick_params(axis="x", labelsize=16)
        block.tick_params(axis="y", labelsize=16)
        block.set_title("Patient {}".format(patients[i]), fontsize=20)
        print()
    
    # prepare legend
    legendBlock = fig.add_subplot(gs[colSize-2, rowSize-1])
    legendBlock.axis("off")
    handles = []
    labels = []
    namesSkip = ['SKIN', "PTVSeg0", "PTVSeg1", "PTVSeg2", "PTVSeg3"]
    for name, color in colorMap.items():
        if name in namesSkip:
            continue
        handleEntry = plt.Line2D([0], [0], color=color, lw=2)
        handles.append(handleEntry)
        labels.append(name)
    legendBlock.legend(handles, labels, loc="center", ncol=2, fontsize=16)
    plt.tight_layout()

    figureFile = os.path.join(figureFolder, "FastDoseTCIABeamCorrect.png")
    plt.savefig(figureFile)
    figureFile = os.path.join(figureFolder, "FastDoseTCIABeamCorrect.eps")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()


if __name__ == "__main__":
    StructsInit()
    # DVH_plot_single_patient()
    # masks2nrrd()
    DVH_plot()