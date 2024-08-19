import os
import numpy as np
import nrrd
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator

isoRes = 2.5
StructuresGlobal = None

sourceFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas"
resultFolder = os.path.join(sourceFolder, "plansAngleCorrect")
if not os.path.isdir(resultFolder):
    os.mkdir(resultFolder)
numPatients = 5

def fpanglesGen(angleRes: float):
    """
    Generate angles with angle resolution angleRes
    """
    # angleRes unit: rad
    eps = 1e-4
    angleList = []
    numTheta = round(np.pi / angleRes)
    for thetaIdx in range(numTheta + 1):
        theta = thetaIdx * angleRes
        phiTotal = 2 * np.pi * np.sin(theta)
        numPhi = int(np.ceil(phiTotal / angleRes))
        if numPhi == 0:
            numPhi = 1
        deltaPhi = 2 * np.pi / numPhi
        for phiIdx in range(numPhi):
            phi = phiIdx * deltaPhi
            sinTheta = np.sin(theta)
            cosTheta = np.cos(theta)
            sinPhi = np.sin(phi)
            cosPhi = np.cos(phi)
            direction = np.array((sinTheta * cosPhi, sinTheta * sinPhi, cosTheta))
            angleList.append((np.array((theta, phiIdx * deltaPhi)), direction))
    return angleList


def direction2VarianIEC(direction: np.array):
    eps = 1e-4
    cosGantry = direction[1]
    sinGantry = np.sqrt(1 - cosGantry ** 2)
    gantry = np.arccos(cosGantry)
    if sinGantry < eps:
        minusCouch = 0
    else:
        sinMinusCouch = direction[2] / sinGantry
        cosMinusCouch = - direction[0] / sinGantry
        cosMinusCouch = np.clip(cosMinusCouch, -1, 1)
        minusCouch = np.arccos(cosMinusCouch)
        if sinMinusCouch < 0:
            minusCouch = - minusCouch
    couch = - minusCouch
    return (gantry, couch)


def fpanglesGenTest():
    angleRes = 6 * np.pi / 180  # 6 degrees
    angleListInit = fpanglesGen(angleRes)
    
    if False:
        VarianIECList = []
        for i in range(len(angleListInit)):
            _, direction = angleListInit[i]
            gantry, couch = direction2VarianIEC(direction)
            
            gantryAngle = gantry * 180 / np.pi
            couchAngle = couch * 180 / np.pi
            entry = "{:.4f} {:.4f} {:.4f}".format(gantryAngle, couchAngle, 0)
            VarianIECList.append(entry)

        beamList = "\n".join(VarianIECList)
        file = os.path.join(resultFolder, "beamListFull.txt")
        with open(file, "w") as f:
            f.write(beamList)
    else:
        vectorBEV = np.array((0, 1, 0))
        for i in range(len(angleListInit)):
            _, direction = angleListInit[i]
            gantry, couch = direction2VarianIEC(direction)
            vectorCartesian = inverseRotateBeamAtOriginRHS(vectorBEV, gantry, couch, 0)
            print(direction, vectorCartesian)


def validAngleListGen():
    """
    This function generates the number of valid beams for each patient
    """
    eps = 1e-4
    angleResInit = 6 * np.pi / 180
    angleListInit = fpanglesGen(angleResInit)
    numBeamsDesired = 400

    for idx in range(1, numPatients + 1):
        patientName = "Patient{:03d}".format(idx)
        maskFolder = os.path.join(sourceFolder, patientName, "InputMask")
        patientTargetFolder = os.path.join(resultFolder, patientName)

        patientDimension = os.path.join(sourceFolder, patientName, "FastDose", "prep_output", "dimension.txt")
        with open(patientDimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension_flip = np.flip(dimension)
        
        # load the masks
        ptv = os.path.join(maskFolder, "PTV.bin")
        ptv = np.fromfile(ptv, dtype=np.uint8)
        ptv = np.reshape(ptv, dimension_flip)
        ptv = np.transpose(ptv, axes=(2, 1, 0))
        ptvCentroid = centroidCalc(ptv)

        skin = os.path.join(maskFolder, "SKIN.bin")
        skin = np.fromfile(skin, dtype=np.uint8)
        skin = np.reshape(skin, dimension_flip)
        skin = np.transpose(skin, axes=(2, 1, 0))
        skin = (skin > 0).astype(float)
        # generate the first and bottom slices
        skinProjZ = np.sum(skin, axis=(0, 1))
        idxValid = [i for i in range(skin.shape[2]) if skinProjZ[i] > 0]
        idxMin, idxMax = min(idxValid), max(idxValid)
        sliceBottom = skin[:, :, idxMin].copy()
        sliceBottom = RegularGridInterpolator((np.arange(dimension[0]), np.arange(dimension[1])),
            sliceBottom, bounds_error=False, fill_value=0)
        sliceTop = skin[:, :, idxMax]
        sliceTop = RegularGridInterpolator((np.arange(dimension[0]), np.arange(dimension[1])),
            sliceTop, bounds_error=False, fill_value=0)

        # calculate the list of valid angles
        validAnglesLocal = []
        for j in range(len(angleListInit)):
            angle, direction = angleListInit[j]
            if abs(direction[2]) < eps:
                validAnglesLocal.append((angle, direction))
                continue
            # calculate the intersection with the top slice
            k_value = (idxMax - ptvCentroid[2]) / direction[2]
            if k_value < 0:
                intersectionTop = ptvCentroid + k_value * direction
                intersectionTop = intersectionTop[:2]
                intersectionValue = sliceTop(intersectionTop)
                if intersectionValue < eps:
                    validAnglesLocal.append((angle, direction))
                continue
            
            # calculate the intersection with the bottom slice
            k_value = (idxMin - ptvCentroid[2]) / direction[2]
            if k_value < 0:
                intersectionBottom = ptvCentroid + k_value * direction
                intersectionBottom = intersectionBottom[:2]
                intersectionValue = sliceBottom(intersectionBottom)
                if intersectionValue < eps:
                    validAnglesLocal.append((angle, direction))
        numPreSelect = len(validAnglesLocal)

        # modify the angular resolution to get the desired number of beams
        angleResAdjust = angleResInit * np.sqrt(numPreSelect / numBeamsDesired)
        angleListAdjust = fpanglesGen(angleResAdjust)
        validAnglesAdjust = []
        for j in range(len(angleListAdjust)):
            angle, direction = angleListAdjust[j]
            if abs(direction[2]) < eps:
                validAnglesAdjust.append((angle, direction))
                continue
            # calculate the intersection with the top slice
            k_value = (idxMax - ptvCentroid[2]) / direction[2]
            if k_value < 0:
                intersectionTop = ptvCentroid + k_value * direction
                intersectionTop = intersectionTop[:2]
                intersectionValue = sliceTop(intersectionTop)
                if intersectionValue < eps:
                    validAnglesAdjust.append((angle, direction))
                continue
            
            # calculate the intersection with the bottom slice
            k_value = (idxMin - ptvCentroid[2]) / direction[2]
            if k_value < 0:
                intersectionBottom = ptvCentroid + k_value * direction
                intersectionBottom = intersectionBottom[:2]
                intersectionValue = sliceBottom(intersectionBottom)
                if intersectionValue < eps:
                    validAnglesAdjust.append((angle, direction))
        
        # calculate the gantry and couch angles
        VarianIECList = []
        for i in range(len(validAnglesAdjust)):
            _, direction = validAnglesAdjust[i]
            gantry, phi = direction2VarianIEC(direction)

            # then convert to degrees
            gantryDegree = gantry * 180 / np.pi
            phiDegree = phi * 180 / np.pi
            entry = "{:.4f} {:.4f} {:.4f}".format(gantryDegree, phiDegree, 0)
            VarianIECList.append(entry)
        
        beamList1 = VarianIECList[:200]
        beamList2 = VarianIECList[200:]
        patientTargetFolder = os.path.join(resultFolder, patientName)
        if not os.path.isdir(patientTargetFolder):
            os.mkdir(patientTargetFolder)
        beamList = "\n".join(VarianIECList)
        beamListList = [beamList1, beamList2]
        beamListList = ["\n".join(a) for a in beamListList]
        beamListFile = os.path.join(patientTargetFolder, "beamlist.txt")
        with open(beamListFile, "w") as f:
            f.write(beamList)
        for i in range(len(beamListList)):
            content = beamListList[i]
            file = os.path.join(patientTargetFolder, "beamlist{}.txt".format(i+1))
            with open(file, "w") as f:
                f.write(content)
        print(patientName)


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


def filePrep():
    files = ["params.txt", "StructureInfo.csv"]
    for i in range(5):
        patientName = "Patient{:03d}".format(i+1)
        sourceExpFolder = os.path.join(sourceFolder, patientName, "FastDose")
        targetExpFolder = os.path.join(resultFolder, patientName, "FastDose")
        if not os.path.isdir(targetExpFolder):
            os.mkdir(targetExpFolder)
        for file in files:
            sourceFile = os.path.join(sourceExpFolder, file)
            targetFile = os.path.join(targetExpFolder, file)
            command = "cp {} {}".format(sourceFile, targetFile)
            os.system(command)


def nrrdVerification():
    def hex_to_rgb(hex_color):
        """Converts a color from hexadecimal format to RGB."""
        hex_color = hex_color.lstrip('#')
        result = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        result = np.array(result) / 255
        result = "{} {} {}".format(*result)
        return result

    patientName = "Patient{:03d}".format(1)
    global StructuresGlobal
    StructuresGlobal = []
    for i in range(numPatients):
        patientName_ = "Patient{:03d}".format(i + 1)
        maskFolder = os.path.join(sourceFolder, patientName_, "InputMask")
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
    colors = [hex_to_rgb(a) for a in colors]
    StructuresGlobal = {StructuresGlobal[i]: colors[i] for i in range(len(StructuresGlobal))}

    patientSourceFolder = os.path.join(sourceFolder, patientName)
    dimensionFile = os.path.join(patientSourceFolder, "FastDoseCorrect", "prep_output", "dimension.txt")
    with open(dimensionFile, "r") as f:
        dimension = f.readline()
    dimension = dimension.replace(" ", ", ")
    dimension = eval(dimension)
    dimension = np.flip(dimension)
    
    MaskFolder = os.path.join(patientSourceFolder, "InputMask")
    maskDict = {}
    for struct in StructuresGlobal:
        fileName = os.path.join(MaskFolder, struct + ".bin")
        if not os.path.isfile(fileName):
            continue
        maskArray = np.fromfile(fileName, dtype=np.uint8)
        maskArray = np.reshape(maskArray, dimension)
        maskDict[struct] = maskArray
    
    beamListFile = os.path.join(resultFolder, patientName, "beamlist.txt")
    with open(beamListFile, "r") as f:
        lines = f.readlines()
    nBeams2Show = 20
    lines = lines[:nBeams2Show]
    lines = [np.array(eval(a.replace(" ", ", "))) for a in lines]
    lines = [a * np.pi / 180 for a in lines]
    beamsMask = genBeamsMask(patientName, maskDict["PTV"], lines)
    maskDict["BEAMS"] = beamsMask
    mask, header = nrrdGen(maskDict)
    file = os.path.join(resultFolder, patientName, "beamMasksFastDose.nrrd")
    nrrd.write(file, mask, header)
    print(file)


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


def genBeamsMask(patientName, PTVMask, beamsSelect):
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

def drawBeams():
    pass


if __name__ == "__main__":
    # fpanglesGenTest()
    validAngleListGen()
    # filePrep()
    # nrrdVerification()