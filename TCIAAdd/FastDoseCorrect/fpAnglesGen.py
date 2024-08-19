import os
import numpy as np
import nrrd
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())

patients = ["002", "003", "009", "013", "070", "125", "132", "190"]
RootFolder = "/data/qifan/projects/FastDoseWorkplace/TCIAAdd"
resultFolder = os.path.join(RootFolder, "plansAngleCorrect")
if not os.path.isdir(resultFolder):
    os.mkdir(resultFolder)
isoRes = 2.5
global StructsMetadata


def StructsExclude():
    """
    This function removes the structures that are irrelevant in the optimization
    """
    global StructsMetadata
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
        # "159": {"exclude": ["CTV56", "CTV63", "CTV70", "GTV"],
        #       "PTV": ["PTV56", "PTV63", "PTV70"],
        #       "BODY": "SKIN"},
        "190": {"exclude": ["CTV70", "GTV"],
              "PTV": ["CTV56", "PTV56", "PTV70"],
              "BODY": "SKIN"}
    }


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


def validAngleListGen():
    """
    This function generates the number of valid beams for each patient
    """
    eps = 1e-4
    angleResInit = 6 * np.pi / 180
    angleListInit = fpanglesGen(angleResInit)
    numBeamsDesired = 400

    for patientName in StructsMetadata:
        patientResultFolder = os.path.join(resultFolder, patientName)
        if not os.path.isdir(patientResultFolder):
            os.mkdir(patientResultFolder)

        rootPatientFolder = os.path.join(RootFolder, patientName)
        patientDimension = os.path.join(rootPatientFolder, "FastDose", "prep_output", "dimension.txt")
        with open(patientDimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension_flip = np.flip(dimension)

        # There are four PTVs
        maskFolder = os.path.join(rootPatientFolder, "PlanMask")
        PTVCentroidList = []
        for i in range(4):
            fileName = os.path.join(maskFolder, "PTVSeg{}.bin".format(i))
            maskLocal = np.fromfile(fileName, dtype=np.uint8)
            maskLocal = np.reshape(maskLocal, dimension_flip)  # order: (z, y, x)
            maskLocal = np.transpose(maskLocal, axes=(2, 1, 0))  # order: (x, y, z)
            centroidLocal = centroidCalc(maskLocal)  # order: (x, y, z)
            if False:
                # for test purposes
                zCoords = int(centroidLocal[0])
                slice = maskLocal[zCoords, :, :]
                sliceFile = os.path.join(patientResultFolder, "axialPTVSeg{}.png".format(i))
                plt.imsave(sliceFile, slice)
                print(sliceFile)
            PTVCentroidList.append(centroidLocal)

        # load SKIN masks
        fileName = os.path.join(maskFolder, "SKIN.bin")
        skin = np.fromfile(fileName, dtype=np.uint8)
        skin = np.reshape(skin, dimension_flip)
        skin = np.transpose(skin, axes=(2, 1, 0))  # order: (x, y, z)
        partialSkin = np.any(skin, axis=(0, 1))
        idxZ = [i for i in range(partialSkin.size) if partialSkin[i]]
        idxZ_min, idxZ_max = min(idxZ), max(idxZ)
        sliceBottom = skin[:, :, idxZ_min].astype(float)  # order: (x, y)
        sliceBottom = RegularGridInterpolator((np.arange(sliceBottom.shape[0]),
            np.arange(sliceBottom.shape[1])), sliceBottom, bounds_error=False, fill_value=0)
        sliceTop = skin[:, :, idxZ_max].astype(float)
        sliceTop = RegularGridInterpolator((np.arange(sliceTop.shape[0]),
            np.arange(sliceTop.shape[1])), sliceTop, bounds_error=False, fill_value=0)
        
        validBeamList = []
        for i in range(len(angleListInit)):
            angle, direction = angleListInit[i]  # direction order: (x, y, z)
            # to distribute the angles among PTVs uniformly
            ptvIdx = i % len(PTVCentroidList)
            ptvCentroid = PTVCentroidList[ptvIdx]

            if abs(direction[2]) < eps:
                validBeamList.append((ptvIdx, ptvCentroid, angle, direction))
                continue

            # calculate the intersection with the top slice
            k_value = (idxZ_max - ptvCentroid[2]) / direction[2]
            if k_value < 0:
                intersectionTop = ptvCentroid + k_value * direction
                intersectionTop = intersectionTop[:2]
                intersectionValue = sliceTop(intersectionTop)
                if intersectionValue < eps:
                    validBeamList.append((ptvIdx, ptvCentroid, angle, direction))
                continue
            
            k_value = (idxZ_min - ptvCentroid[2]) / direction[2]
            if k_value < 0:
                intersectionBottom = ptvCentroid + k_value * direction
                intersectionBottom = intersectionBottom[: 2]
                intersectionValue = sliceBottom(intersectionBottom)
                if intersectionValue < eps:
                    validBeamList.append((ptvIdx, ptvCentroid, angle, direction))
        numPreSelect = len(validBeamList)

        # modify the angular resolution to get the desired number of beams
        angleResAdjust = angleResInit * np.sqrt(numPreSelect / numBeamsDesired)
        angleListAdjust = fpanglesGen(angleResAdjust)
        validAnglesAdjust = []
        for j in range(len(angleListAdjust)):
            angle, direction = angleListAdjust[j]
            ptvIdx = j % len(PTVCentroidList)
            ptvCentroid = PTVCentroidList[ptvIdx]
            if abs(direction[2]) < eps:
                validAnglesAdjust.append((ptvIdx, ptvCentroid, angle, direction))
                continue
            # calculate the intersection with the top slice
            k_value = (idxZ_max - ptvCentroid[2]) / direction[2]
            if k_value < 0:
                intersectionTop = ptvCentroid + k_value * direction
                intersectionTop = intersectionTop[:2]
                intersectionValue = sliceTop(intersectionTop)
                if intersectionValue < eps:
                    validAnglesAdjust.append((ptvIdx, ptvCentroid, angle, direction))
                continue
            
            k_value = (idxZ_min - ptvCentroid[2]) / direction[2]
            if k_value < 0:
                intersectionBottom = ptvCentroid + k_value * direction
                intersectionBottom = intersectionBottom[:2]
                intersectionValue = sliceBottom(intersectionBottom)
                if intersectionValue < eps:
                    validAnglesAdjust.append((ptvIdx, ptvCentroid, angle, direction))
        
        # then separate the list "validAnglesAdjust" into different lists, each
        # corresponding to a PTV segment
        beamListList = [[], [], [], []]
        for i in range(len(validAnglesAdjust)):
            entry = validAnglesAdjust[i]
            ptvIdx, direction = entry[0], entry[3]
            gantry, phi = direction2VarianIEC(direction)
            gantryDegree = gantry * 180 / np.pi
            phiDegree = phi * 180 / np.pi
            textLine = "{:.4f} {:.4f} {:.4f}".format(gantryDegree, phiDegree, 0)
            beamListList[ptvIdx].append(textLine)
        for i in range(4):
            fullText = "\n".join(beamListList[i])
            file = os.path.join(patientResultFolder, "beamlist{}.txt".format(i))
            with open(file, "w") as f:
                f.write(fullText)
        print(patientName)
        


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


def nrrdVerification():
    patientName = "002"
    patientMetaData = StructsMetadata[patientName]
    excludeList = patientMetaData["exclude"]
    ptvList = patientMetaData["PTV"]
    bodyName = patientMetaData["BODY"]
    
    rootPatientFolder = os.path.join(RootFolder, patientName)
    patientResultFolder = os.path.join(resultFolder, patientName)
    maskFolder = os.path.join(rootPatientFolder, "PlanMask")
    masks = [a.split(".")[0] for a in os.listdir(maskFolder)]
    masks = [a for a in masks if a not in excludeList]
    assert bodyName in masks
    masks_clean = [a for a in masks if "PTV" not in a and a != bodyName]
    masks_clean.sort()
    masks_clean.insert(0, "PTV56")
    masks_clean.insert(0, "PTV70")
    masks_clean.append(bodyName)
    
    dimension = os.path.join(rootPatientFolder, "FastDose", "prep_output", "dimension.txt")
    with open(dimension, "r") as f:
        dimension = f.readline()
    dimension = dimension.replace(" ", ", ")
    dimension = eval(dimension)
    dimension_flip = np.flip(dimension)

    maskDict = {}
    for name in masks_clean:
        maskFile = os.path.join(maskFolder, name + ".bin")
        mask = np.fromfile(maskFile, dtype=np.uint8)
        mask = np.reshape(mask, dimension_flip)  # (z, y, x)
        maskDict[name] = mask
    
    # generate the beamMask
    for i in range(4):
        PTVSegMask = os.path.join(maskFolder, "PTVSeg{}.bin".format(i))
        PTVSegMask = np.fromfile(PTVSegMask, dtype=np.uint8)
        PTVSegMask = np.reshape(PTVSegMask, dimension_flip)  # (z, y, x)
        localBeamList = os.path.join(patientResultFolder, "beamlist{}.txt".format(i))
        with open(localBeamList, "r") as f:
            localBeamList = f.readlines()
        localBeamList = localBeamList[: 5]
        localBeamList = [np.array(eval(a.replace(" ", ", "))) for a in localBeamList]
        localBeamList = [a * np.pi / 180 for a in localBeamList]
        localBeamMask = genBeamsMask(patientName, PTVSegMask, localBeamList)
        maskDict["beamsPTV{}".format(i)] = localBeamMask
    maskResult, header = nrrdGen(maskDict)
    file = os.path.join(patientResultFolder, "beamMasksFastDose.nrrd")
    nrrd.write(file, maskResult, header)
    print(file)


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
        
        color = hex_to_rgb(colors[i])
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


def hex_to_rgb(hex_color):
    """Converts a color from hexadecimal format to RGB."""
    hex_color = hex_color.lstrip('#')
    result = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    result = np.array(result) / 255
    result = "{} {} {}".format(*result)
    return result


def filePrep():
    files = ["params.txt", "StructureInfo.csv"]
    for patientName in StructsMetadata:
        sourceExpFolder = os.path.join(RootFolder, patientName, "FastDose")
        targetExpFolder = os.path.join(resultFolder, patientName, "FastDose")
        if not os.path.isdir(targetExpFolder):
            os.mkdir(targetExpFolder)
        command1 = "cp {} {}".format(os.path.join(sourceExpFolder, "params1.txt"), targetExpFolder)
        os.system(command1)
        command2 = "cp {} {}".format(os.path.join(sourceExpFolder, "StructureInfo1.csv"), targetExpFolder)
        os.system(command2)


if __name__ == "__main__":
    StructsExclude()
    # validAngleListGen()
    # nrrdVerification()
    filePrep()