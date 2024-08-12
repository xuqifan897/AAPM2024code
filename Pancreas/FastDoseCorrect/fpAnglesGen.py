import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

sourceFolder = "/data/qifan/FastDoseWorkplace/Pancreas"
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
    direction = - direction  # invert direction
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
            print(-direction, vectorCartesian)


def validAngleListGen():
    """
    This function generates the number of valid beams for each patient
    """
    eps = 1e-4
    angleResInit = 6 * np.pi / 180
    angleListInit = fpanglesGen(angleResInit)
    numBeamsDesired = 800

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
            if k_value > 0:
                intersectionTop = ptvCentroid + k_value * direction
                intersectionTop = intersectionTop[:2]
                intersectionValue = sliceTop(intersectionTop)
                if intersectionValue < eps:
                    validAnglesLocal.append((angle, direction))
                    continue
            
            # calculate the intersection with the bottom slice
            k_value = (idxMin - ptvCentroid[2]) / direction[2]
            if k_value > 0:
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
            if k_value > 0:
                intersectionTop = ptvCentroid + k_value * direction
                intersectionTop = intersectionTop[:2]
                intersectionValue = sliceTop(intersectionTop)
                if intersectionValue < eps:
                    validAnglesAdjust.append((angle, direction))
                    continue
            
            # calculate the intersection with the bottom slice
            k_value = (idxMin - ptvCentroid[2]) / direction[2]
            if k_value > 0:
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
        beamList2 = VarianIECList[200: 400]
        beamList3 = VarianIECList[400: 600]
        beamList4 = VarianIECList[600:]
        patientTargetFolder = os.path.join(resultFolder, patientName)
        if not os.path.isdir(patientTargetFolder):
            os.mkdir(patientTargetFolder)
        beamList = "\n".join(VarianIECList)
        beamListList = [beamList1, beamList2, beamList3, beamList4]
        beamListList = ["\n".join(a) for a in beamListList]
        beamListFile = os.path.join(patientTargetFolder, "beamlist.txt")
        with open(beamListFile, "w") as f:
            f.write(beamList)
        for i in range(len(beamListList)):
            content = beamListList[i]
            file = os.path.join(patientTargetFolder, "beamlist{}.txt".format(i))
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
    sourceFolder = "/data/qifan/FastDoseWorkplace/Pancreas"
    targetFolder = "/data/qifan/FastDoseWorkplace/Pancreas/plansAngleCorrect"
    files = ["params.txt", "StructureInfo.csv"]
    for i in range(5):
        patientName = "Patient{:03d}".format(i+1)
        sourceExpFolder = os.path.join(sourceFolder, patientName, "FastDose")
        targetExpFolder = os.path.join(targetFolder, patientName, "FastDose")
        if not os.path.isdir(targetExpFolder):
            os.mkdir(targetExpFolder)
        for file in files:
            sourceFile = os.path.join(sourceExpFolder, file)
            targetFile = os.path.join(targetExpFolder, file)
            command = "cp {} {}".format(sourceFile, targetFile)
            os.system(command)


if __name__ == "__main__":
    # fpanglesGenTest()
    # validAngleListGen()
    filePrep()