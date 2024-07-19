import os
import numpy as np
import nrrd
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import convolve

rootFolder = "/data/qifan/FastDoseWorkplace/Pancreas"
numPatients = 5
StructuresGlobal = None
isoRes = 2.5  # mm

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
    colors = [hex_to_rgb(a) for a in colors]
    StructuresGlobal = {StructuresGlobal[i]: colors[i] for i in range(len(StructuresGlobal))}
    # print(StructuresGlobal)

def masks2nrrd():
    """
    This function converts the original binary masks into nrrd
    """
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i + 1)
        patientFolder = os.path.join(rootFolder, patientName)
        dimensionFile = os.path.join(patientFolder, "FastDose", "prep_output", "dimension.txt")
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

        
        beamlistFastDose = os.path.join(patientFolder, "FastDose", "plan1", "metadata.txt")
        with open(beamlistFastDose, "r") as f:
            beamsSelect = f.readlines()
        beamsSelect = beamsSelect[3]
        beamsSelect = beamsSelect.replace("  ", ", ")
        beamsSelect = eval(beamsSelect)

        beamsMask = genBeamsMask(patientName, maskDict["PTV"], beamsSelect)
        maskDict["BEAMS"] = beamsMask
        mask, header = nrrdGen(maskDict)
        file = os.path.join(patientFolder, "beamMasksFastDose.nrrd")
        nrrd.write(file, mask, header)
        print(file)


def genBeamsMask(patientName, PTVMask, beamsSelect):
    patientFolder = os.path.join(rootFolder, patientName)
    beamListFile = os.path.join(patientFolder, "FastDose", "beamlist.txt")
    with open(beamListFile, "r") as f:
        angles = f.readlines()
    for i in range(len(angles)):
        currentLine = angles[i]
        currentLine = currentLine.replace(" ", ", ")
        currentLine = eval(currentLine)
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
    
    directionsSelect = []
    for angle_entry in beamsSelect:
        axisBEV = np.array((0, 1, 0))
        axisPVCS = inverseRotateBeamAtOriginRHS(axisBEV,
            angle_entry[0], angle_entry[1], angle_entry[2])
        directionsSelect.append(axisPVCS)
    
    # calculate the coordinates
    coordsShape = PTVMask.shape + (3, )
    coords = np.zeros(coordsShape, dtype=float)
    axis_x = np.arange(coordsShape[0])
    axis_x = np.expand_dims(axis_x, axis=(1, 2))
    axis_y = np.arange(coordsShape[1])
    axis_y = np.expand_dims(axis_y, axis=(0, 2))
    axis_z = np.arange(coordsShape[2])
    axis_z = np.expand_dims(axis_z, axis=(0, 1))
    coords[:, :, :, 0] = axis_x
    coords[:, :, :, 1] = axis_y
    coords[:, :, :, 2] = axis_z
    PTVCentroid = calcCentroid(PTVMask)
    PTVCentroid = np.expand_dims(PTVCentroid, axis=(0, 1, 2))
    coords_minus_isocenter = coords - PTVCentroid

    beamsMask = None
    radius = 2
    barLength = 100
    for direction in directionsSelect:
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


if __name__ == "__main__":
    StructsInit()
    masks2nrrd()