# This file is created during the preparation of the manuscript, to correct for the beam angle problem
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import random

rootFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas"
beamListFullPath = "/data/qifan/projects/FastDoseWorkplace/TCIAAdd/beamlist_full.txt"
numPatients = 5


def dataPrep():
    for i in range(numPatients):
        patientFolder = os.path.join(rootFolder, "Patient{:03d}".format(i + 1))
        sourceFolder = os.path.join(patientFolder, "FastDose")
        targetFolder = os.path.join(patientFolder, "FastDoseCorrect")
        if not os.path.isdir(targetFolder):
            os.mkdir(targetFolder)
        command1 = "cp -r {}/prep_output {}/prep_output".format(sourceFolder, targetFolder)
        command2 = "cp {}/params.txt {}/params.txt".format(sourceFolder, targetFolder)
        command3 = "cp {}/StructureInfo.csv {}/StructureInfo.csv".format(sourceFolder, targetFolder)
        os.system(command1)
        os.system(command2)
        os.system(command3)

def beamListGen():
    with open(beamListFullPath, "r") as f:
        beamListFull = f.readlines()

    eps = 1e-4
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i + 1)
        patientFolder = os.path.join(rootFolder, patientName)
        dimensionFile = os.path.join(patientFolder, "FastDoseCorrect", "prep_output", "dimension.txt")
        with open(dimensionFile, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension = np.flip(dimension)

        bodyMaskFile = os.path.join(patientFolder, "InputMask", "SKIN.bin")
        bodyMask = np.fromfile(bodyMaskFile, dtype=np.uint8)
        bodyMask = np.reshape(bodyMask, dimension)

        lowerSlice = bodyMask[1, :, :]
        upperSlice = bodyMask[-1, :, :]
        print(patientName, np.sum(lowerSlice), np.sum(upperSlice))
        lowerSlice = RegularGridInterpolator(
            (np.arange(lowerSlice.shape[0]), np.arange(lowerSlice.shape[1])),
            lowerSlice, bounds_error=False, fill_value=0)
        upperSlice = RegularGridInterpolator(
            (np.arange(upperSlice.shape[0]), np.arange(upperSlice.shape[1])),
            upperSlice, bounds_error=False, fill_value=0)

        PTVMaskFile = os.path.join(patientFolder, "InputMask", "PTV.bin")
        PTVMask = np.fromfile(PTVMaskFile, dtype=np.uint8)
        PTVMask = np.reshape(PTVMask, dimension)
        PTVCentroid = calcCentroid(PTVMask)

        beamListSelect = []
        for line in beamListFull:
            lineOrg = line
            line = line.replace(" ", ", ")
            angle = eval(line)
            axisBEV = np.array((0, 1, 0))
            direction = inverseRotateBeamAtOriginRHS(axisBEV, angle[0], angle[1], angle[2])

            if direction[0] > 0:
                scale = (dimension[0] - PTVCentroid[0]) / direction[0]
                point = PTVCentroid + scale * direction
                point = point[1:]
                value = upperSlice(point)
                if value < eps:
                    beamListSelect.append(lineOrg)
                else:
                    # print(direction)
                    pass
            elif direction[0] < 0:
                scale = (0 - PTVCentroid[0]) / direction[0]
                point = PTVCentroid + scale * direction
                point = point[1:]
                value = lowerSlice(point)
                if value < eps:
                    beamListSelect.append(lineOrg)
                else:
                    # print(direction)
                    pass
            elif direction[0] == 0:
                beamListSelect.append(lineOrg)
        print(len(beamListSelect), len(beamListFull), "\n")

        # reduce the number of valid beams by half
        nBeamsWeWant = len(beamListSelect)
        nBeamsWeWant = int(nBeamsWeWant / 2)
        random.shuffle(beamListSelect)
        beamListSelect = beamListSelect[:nBeamsWeWant]
        
        # split beamListSelect into 4 subsets
        beamListFullText = "".join(beamListSelect)
        if beamListFullText[-1] == "\n":
            beamListFullText = beamListFullText[:-1]
        beamListFullFile = os.path.join(patientFolder, "FastDoseCorrect", "beamlist.txt")
        with open(beamListFullFile, "w") as f:
            f.write(beamListFullText)
        
        numSplits = 4
        nLinesPerGroup = int(np.ceil(len(beamListSelect) / numSplits))
        for j in range(numSplits):
            subGroup = beamListSelect[j*nLinesPerGroup: (j+1)*nLinesPerGroup]
            subGroupText = "".join(subGroup)
            if subGroupText[-1] == "\n":
                subGroupText = subGroupText[:-1]
            subGroupFile = os.path.join(patientFolder, "FastDoseCorrect", "beamlist{}.txt".format(j+1))
            with open(subGroupFile, "w") as f:
                f.write(subGroupText)


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


if __name__ == "__main__":
    # dataPrep()
    beamListGen()