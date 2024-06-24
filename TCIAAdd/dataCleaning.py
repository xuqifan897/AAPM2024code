import os
import glob
import numpy as np
import nrrd
import random
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import measure, transform
from collections import OrderedDict

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())

patients = ["002", "003", "009", "013", "070", "125", "132", "190"]
RootFolder = "/data/qifan/FastDoseWorkplace/TCIAAdd"
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
        "159": {"exclude": ["CTV56", "CTV63", "CTV70", "GTV"],
              "PTV": ["PTV56", "PTV63", "PTV70"],
              "BODY": "SKIN"},
        "190": {"exclude": ["CTV70", "GTV"],
              "PTV": ["CTV56", "PTV56", "PTV70"],
              "BODY": "SKIN"}
    }


def CheckData():
    """
    Check the data format    
    """
    global StructsMetadata
    for patient in patients:
        PatientFolder = os.path.join(RootFolder, patient)
        metadataFile = os.path.join(PatientFolder, "metadata.txt")
        with open(metadataFile, "r") as f:
            lines = f.readlines()
        dimension = eval(lines[0].replace(" ", ", "))
        RescaleSlope = eval(lines[1])
        RescaleIntercept = eval(lines[2])

        CTFile = os.path.join(PatientFolder, "CT.nrrd")
        CTArray, header = nrrd.read(CTFile)
        dimOrg = header["sizes"]
        CTArray = np.reshape(CTArray, dimOrg)
        CTArray -= RescaleIntercept
        CTArray = np.transpose(CTArray, axes=(2, 1, 0))
        dimOrg = np.flip(dimOrg)
        
        segFile = os.path.join(PatientFolder, "RTSTRUCT.nrrd")
        masks = readSegFile(segFile)

        if False:
            # visualize CT slices and masks
            viewFolder = os.path.join(PatientFolder, "view")
            if not os.path.isdir(viewFolder):
                os.mkdir(viewFolder)
            for i in range(dimOrg[0]):
                CTSlice = CTArray[i, :, :]
                plt.imshow(CTSlice, cmap="gray", vmin=500, vmax=2000)
                for j, entry in enumerate(masks.items()):
                    color = colors[j]
                    name, maskArray = entry
                    maskSlice = maskArray[i, :, :]
                    if np.sum(maskSlice) == 0:
                        continue
                    contours = measure.find_contours(maskSlice)
                    initial = True
                    for contour in contours:
                        if initial:
                            plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                            initial = False
                        else:
                            plt.plot(contour[:, 1], contour[:, 0], color=color)
                plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
                plt.tight_layout()
                file = os.path.join(viewFolder, "{:03d}.png".format(i))
                plt.savefig(file)
                plt.clf()
                print(file)
        else:
            CTArray = transform.resize(CTArray, dimension)
            CTArray = CTArray.astype(np.uint16)
            CTArray_file = os.path.join(PatientFolder, "density_raw.bin")
            CTArray.tofile(CTArray_file)

            MaskFolder = os.path.join(PatientFolder, "InputMask")
            if not os.path.isdir(MaskFolder):
                os.mkdir(MaskFolder)
            patient_exclude = StructsMetadata[patient]["exclude"]
            for key in masks:
                if key in patient_exclude:
                    continue
                mask_input = masks[key]
                mask_input = mask_input.astype(float)
                mask_output = transform.resize(mask_input, dimension)
                mask_output = (mask_output > 0).astype(np.uint8)
                mask_file = os.path.join(MaskFolder, "{}.bin".format(key))
                mask_output.tofile(mask_file)
            print(patient)


def readSegFile(file: str):
    seg, header = nrrd.read(file)
    seg = np.transpose(seg, axes=(0, 3, 2, 1))
    result = {}
    idx = 0
    while True:
        keyRoot = "Segment{}_".format(idx)
        nameKey = keyRoot + "Name"
        layerKey = keyRoot + "Layer"
        labelValueKey = keyRoot + "LabelValue"
        if nameKey not in header:
            break
        name = header[nameKey]
        layer = int(header[layerKey])
        labelValue = int(header[labelValueKey])
        mask = seg[layer, :, :, :] == labelValue
        result[name] = mask
        idx += 1
    return result


def postCheck():
    """
    This function checks the generated density array, mask array, and dose array
    """
    for patient in patients:
        if patient in ["002", "003"]:
            continue
        patientFolder = os.path.join(RootFolder, patient)
        dimensionFile = os.path.join(patientFolder, "metadata.txt")
        with open(dimensionFile, "r") as f:
            dimension = f.readline()
        dimension = eval(dimension.replace(" ", ", "))
        
        densityFile = os.path.join(patientFolder, "density_raw.bin")
        densityArray = np.fromfile(densityFile, dtype=np.uint16)
        densityArray = np.reshape(densityArray, dimension)

        doseFile = os.path.join(patientFolder, "dose.bin")
        doseArray = np.fromfile(doseFile, dtype=np.float32)
        doseArray = np.reshape(doseArray, dimension)

        maskFolder = os.path.join(patientFolder, "InputMask")
        files = [a.split(".")[0] for a in os.listdir(maskFolder)]
        masks = {}
        for a in files:
            file = os.path.join(maskFolder, "{}.bin".format(a))
            maskArray = np.fromfile(file, dtype=np.uint8)
            maskArray = np.reshape(maskArray, dimension)
            masks[a] = maskArray

        # normalize
        primaryMask = masks["PTV70"].astype(bool)
        thresh = np.percentile(doseArray[primaryMask], 5)
        doseArray *= 70 / thresh
        showThresh = 5
        
        imageFolder = os.path.join(patientFolder, "view")
        if not os.path.isdir(imageFolder):
            os.mkdir(imageFolder)
        nSlices = dimension[0]
        for i in range(nSlices):
            densitySlice = densityArray[i, :, :]
            plt.imshow(densitySlice, cmap="gray", vmin=500, vmax=2000)
            for j, entry in enumerate(masks.items()):
                color = colors[j]
                name, maskArray = entry
                maskSlice = maskArray[i, :, :]
                if np.sum(maskSlice) == 0:
                    continue
                contours = measure.find_contours(maskSlice)
                initial = True
                for contour in contours:
                    if initial:
                        plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                        initial = False
                    else:
                        plt.plot(contour[:, 1], contour[:, 0], color=color)
            doseSlice = doseArray[i, :, :]
            plt.imshow(doseSlice, cmap="jet", vmin=0, vmax=80, alpha=(doseSlice>showThresh)*0.3)
            plt.legend(loc="upper left", bbox_to_anchor=[1.02, 1.0])
            sliceFile = os.path.join(imageFolder, "{:03d}.png".format(i))
            plt.savefig(sliceFile)
            plt.clf()
            print(sliceFile)


def MaskTrim():
    "This function merges PTV masks of the same dose, and crops masks so that PTV masks do not " \
    "overlap with each other, and OAR masks do not overlap with PTV masks"
    for patient in patients:
        PatientFolder = os.path.join(RootFolder, patient)
        InputMaskFolder = os.path.join(PatientFolder, "InputMask")
        OutputMaskFolder = os.path.join(PatientFolder, "PlanMask")
        if not os.path.isdir(OutputMaskFolder):
            os.mkdir(OutputMaskFolder)
        
        ExcludeList = (a:=StructsMetadata[patient])["exclude"]
        PTVList = a["PTV"]
        BODY = a["BODY"]

        SpecialComb = ExcludeList + PTVList + [BODY]
        OARs = [b for a in os.listdir(InputMaskFolder) if (b:=a.split(".")[0]).replace(" ", "") not in SpecialComb]
        # group PTVs into different dose levels
        PTVGroups = {}
        for ptv in PTVList:
            dose = "".join(a for a in ptv if a.isdigit())
            dose = eval(dose)
            if dose not in PTVGroups:
                PTVGroups[dose] = [ptv]
            else:
                PTVGroups[dose].append(ptv)
        
        PTVMasksMerge = []
        for dose, group in PTVGroups.items():
            canvas = None
            for name in group:
                MaskFile = os.path.join(InputMaskFolder, name + ".bin")
                MaskArray = np.fromfile(MaskFile, dtype=np.uint8)
                if canvas is None:
                    canvas = MaskArray
                else:
                    canvas = np.logical_or(canvas, MaskArray)
            PTVMasksMerge.append([dose, canvas])
        PTVMasksMerge.sort(key=lambda a: a[0], reverse=True)

        # deal with overlap
        canvas = None
        for i in range(len(PTVMasksMerge)):
            PTVMask = PTVMasksMerge[i][1]
            if canvas is None:
                canvas = PTVMask
            else:
                PTVMask = np.logical_and(PTVMask, np.logical_not(canvas))
                canvas = np.logical_or(PTVMask, canvas)
                PTVMask = PTVMask.astype(np.uint8)
                PTVMasksMerge[i][1] = PTVMask
        
        OARMaskDict = {}
        for name in OARs:
            OARMaskFile = os.path.join(InputMaskFolder, "{}.bin".format(name))
            OARMask = np.fromfile(OARMaskFile, dtype=np.uint8)
            OARMask = np.logical_and(OARMask, np.logical_not(canvas))
            OARMask = OARMask.astype(np.uint8)
            OARMaskDict[name] = OARMask
        
        # write results
        for dose, mask in PTVMasksMerge:
            destFile = os.path.join(OutputMaskFolder, "PTV{}.bin".format(dose))
            mask.tofile(destFile)
        for name, mask in OARMaskDict.items():
            destFile = os.path.join(OutputMaskFolder, "{}.bin".format(name.replace(" ", "")))
            mask.tofile(destFile)
        BODYSource = os.path.join(InputMaskFolder, "{}.bin".format(BODY))
        BODYDest = os.path.join(OutputMaskFolder, "{}.bin".format(BODY))
        command = "cp \"{}\" \"{}\"".format(BODYSource, BODYDest)
        os.system(command)
        print("Patient {} done!".format(patient))


def PTVSeg():
    """
    This function follows the method proposed by Qihui et cl in the paper
    "Many-isocenter optimization for robotic radiotherpay"
    """
    for patient in patients:
        PatientFolder = os.path.join(RootFolder, patient)
        metadata = os.path.join(PatientFolder, "metadata.txt")
        with open(metadata, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        MaskPattern = os.path.join(PatientFolder, "PlanMask", "PTV*[0-9][0-9]*.bin")
        PTVList = [os.path.basename(a).split(".")[0] for a in glob.glob(MaskPattern)]

        MaskFolder = os.path.join(PatientFolder, "PlanMask")
        PTVMaskMerge = None
        for name in PTVList:
            file = os.path.join(MaskFolder, "{}.bin".format(name))
            maskArray = np.fromfile(file, dtype=np.uint8)
            if PTVMaskMerge is None:
                PTVMaskMerge = maskArray
            else:
                PTVMaskMerge = np.logical_or(PTVMaskMerge, maskArray)
        PTVMaskMerge = np.reshape(PTVMaskMerge, dimension)
        
        # we find the minimum bounding box that encapsulate the whole PTV area
        # and then divide the whole PTV volume into 2 x 2 sub-blocks
        AxisX = np.any(PTVMaskMerge, axis=(0, 1))
        indices = [a for a in range(AxisX.size) if AxisX[a]]
        AxisXLower = min(indices)
        AxisXUpper = max(indices) + 1
        AxisXMiddle = int((AxisXLower + AxisXUpper) / 2)
        AxisXPoints = [AxisXLower, AxisXMiddle, AxisXUpper]

        AxisY = np.any(PTVMaskMerge, axis=(0, 2))
        indices = [a for a in range(AxisY.size) if AxisY[a]]
        AxisYLower = min(indices)
        AxisYUpper = max(indices) + 1

        AxisZ = np.any(PTVMaskMerge, axis=(1, 2))
        indices = [a for a in range(AxisZ.size) if AxisZ[a]]
        AxisZLower = min(indices)
        AxisZUpper = max(indices) + 1
        AxisZMiddle = int((AxisZLower + AxisZUpper) / 2)
        AxisZPoints = [AxisZLower, AxisZMiddle, AxisZUpper]

        for i in range(2):
            IdxXBegin = AxisXPoints[i]
            IdxXEnd = AxisXPoints[i+1]
            for j in range(2):
                IdxZBegin = AxisZPoints[j]
                IdxZEnd = AxisZPoints[j+1]
                Mask = np.zeros_like(PTVMaskMerge)
                Mask[IdxZBegin: IdxZEnd, AxisYLower:AxisYUpper, IdxXBegin: IdxXEnd] = 1
                PTVAndMask = np.logical_and(PTVMaskMerge, Mask)
                PTVAndMask = PTVAndMask.astype(np.uint8)

                PTVSegIdx = i * 2 + j
                OutputFile = os.path.join(MaskFolder, "PTVSeg{}.bin".format(PTVSegIdx))
                PTVAndMask.tofile(OutputFile)
                print(OutputFile)

        PTVMaskMerge = PTVMaskMerge.astype(np.uint8)
        PTVMergeFile = os.path.join(MaskFolder, "PTVMerge.bin")
        PTVMaskMerge.tofile(PTVMergeFile)
        print(PTVMergeFile)
        print()


def ShowPTVSeg():
    """
    This function shows the PTVsegs
    """
    for patient in patients:
        PatientFolder = os.path.join(RootFolder, patient)
        metadata = os.path.join(PatientFolder, "metadata.txt")
        with open(metadata, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)

        CTArrayFile = os.path.join(PatientFolder, "density_raw.bin")
        CTArray = np.fromfile(CTArrayFile, dtype=np.uint16)
        CTArray = np.reshape(CTArray, dimension)

        DoseArrayFile = os.path.join(PatientFolder, "dose.bin")
        DoseArray = np.fromfile(DoseArrayFile, dtype=np.float32)
        DoseArray = np.reshape(DoseArray, dimension)
        DoseRoof = np.percentile(DoseArray, 99)
        DoseArray *= 70 / DoseRoof

        MaskFiles = [os.path.join(PatientFolder, "PlanMask", "PTVSeg{}.bin".format(i)) for i in range(4)]
        Masks = []
        for i, file in enumerate(MaskFiles):
            MaskArray = np.fromfile(file, dtype=np.uint8)
            MaskArray = np.reshape(MaskArray, dimension)
            print(np.sum(MaskArray))
            Masks.append(("PTVSeg{}".format(i), MaskArray))

        FigureFolder = os.path.join(PatientFolder, "view")
        if not os.path.isdir(FigureFolder):
            os.mkdir(FigureFolder)
        for i in range(dimension[0]):
            CTSlice = CTArray[i, :, :]
            DoseSlice = DoseArray[i, :, :]
            plt.imshow(CTSlice, cmap="gray", vmin=500, vmax=1200)
            LegendFlag = False
            for j, entry in enumerate(Masks):
                color = colors[j]
                name, MaskArray = entry
                MaskSlice = MaskArray[i, :, :]
                if np.sum(MaskSlice) == 0:
                    continue
                LegendFlag = True
                contours = measure.find_contours(MaskSlice)
                Initial = True
                for contour in contours:
                    if Initial:
                        plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                        Initial = False
                    else:
                        plt.plot(contour[:, 1], contour[:, 0], color=color)
            plt.imshow(DoseSlice, cmap="jet", vmin=0, vmax=80, alpha=0.3)
            if LegendFlag:
                plt.legend()
            FigureFile = os.path.join(FigureFolder, "{:03d}.png".format(i))
            plt.savefig(FigureFile)
            plt.clf()
            print(FigureFile)


def beamListGen():
    """
    This function generates a set of beams for each patient
    """
    nCandidatesPerSeg = 200
    fullBeamList = os.path.join(RootFolder, "beamlist_full.txt")
    with open(fullBeamList, "r") as f:
        fullBeamList = f.readlines()
    for i in range(len(fullBeamList)):
        entry = fullBeamList[i]
        while entry[-1] in [" ", "\n"]:
            entry = entry[:-1]
        entry = entry.replace(" ", ", ")
        entry = eval(entry)
        # convert degree to radian
        entry = np.array(entry) * np.pi / 180
        fullBeamList[i] = entry
    
    for patient in patients:
        patientFolder = os.path.join(RootFolder, patient)
        dimension = os.path.join(patientFolder, "metadata.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        
        SKIN = os.path.join(patientFolder, "PlanMask", "SKIN.bin")
        SKIN = np.fromfile(SKIN, dtype=np.uint8)
        SKIN = np.reshape(SKIN, dimension)

        # find the top and bottom slices
        partialSKIN = np.any(SKIN, axis=(1, 2))
        idxZ = [i for i in range(partialSKIN.size) if partialSKIN[i]]
        idxZ_min = min(idxZ)
        idxZ_max = max(idxZ)
        bottomSlice = SKIN[idxZ_min, :, :]
        topSlice = SKIN[idxZ_max, :, :]
        
        if True:
            # to exclude all beams irradiating from the top of the head
            radius = 30
            center = (np.array(topSlice.shape) - 1) / 2
            center = np.expand_dims(center, axis=(0, 1))
            coordsShape = topSlice.shape + (2, )
            coords = np.zeros(coordsShape)
            coordsY = np.arange(topSlice.shape[0])
            coordsY = np.expand_dims(coordsY, axis=(1))
            coordsX = np.arange(topSlice.shape[1])
            coordsX = np.expand_dims(coordsX, axis=(0))
            coords[:, :, 0] = coordsY
            coords[:, :, 1] = coordsX
            diff = coords - center
            norm = np.linalg.norm(diff, axis=2)
            topSlice = norm < radius

        # used for calculating the centroid of each PTV segment
        coordsShape = dimension + (3, )
        coords = np.zeros(coordsShape, dtype=float)
        axis_z = np.arange(dimension[0])
        axis_z = np.expand_dims(axis_z, axis=(1, 2))
        axis_y = np.arange(dimension[1])
        axis_y = np.expand_dims(axis_y, axis=(0, 2))
        axis_x = np.arange(dimension[2])
        axis_x = np.expand_dims(axis_x, axis=(0, 1))
        coords[:, :, :, 0] = axis_z
        coords[:, :, :, 1] = axis_y
        coords[:, :, :, 2] = axis_x

        # angle, orientation, 4 flags
        indicator = []
        directionOrg = np.array((0, 1, 0))
        for i in range(len(fullBeamList)):
            angle = fullBeamList[i]
            directionNew = inverseRotateBeamAtOriginRHS(directionOrg, *angle)
            indicator.append((angle, directionNew, np.zeros((4,), dtype=bool)))

        for segIdx in range(4):
            PTVSegMask = os.path.join(patientFolder, "PlanMask", "PTVSeg{}.bin".format(segIdx))
            PTVSegMask = np.fromfile(PTVSegMask, dtype=np.uint8)
            PTVSegMask = np.reshape(PTVSegMask, dimension)
            PTVSegMask = np.expand_dims(PTVSegMask, axis=3)
            nVoxels = np.sum(PTVSegMask)
            centroid = coords * PTVSegMask
            centroid = np.sum(centroid, axis=(0, 1, 2)) / nVoxels
            # print("Patient {}, PTVSeg{}, centroid {}".format(patient, segIdx, centroid))

            # for each beam, calculate the intersection of each beam to the bottom and top slice
            for beamIdx in range(len(indicator)):
                direction = indicator[beamIdx][1]

                # if the absolute diff is too small, True
                if np.abs(direction[0]) < 1e-4:
                    indicator[beamIdx][2][segIdx] = True
                    continue
                
                # judge on the bottom slice
                factor = (idxZ_min - centroid[0]) / direction[0]
                if factor <= 0:
                    # we only consider the case where the factor <= 0, as the beam is one-directional
                    intersectionPoint = centroid + factor * direction
                    intersectionY = int(intersectionPoint[1])
                    intersectionX = int(intersectionPoint[2])
                    if (intersectionY < 0 or intersectionY >= dimension[1] or \
                        intersectionX < 0 or intersectionX >= dimension[2]):
                        indicator[beamIdx][2][segIdx] = True
                    elif bottomSlice[intersectionY, intersectionX] == False:
                        indicator[beamIdx][2][segIdx] = True

                factor = (idxZ_max - centroid[0]) / direction[0]
                if factor <= 0:
                    intersectionPoint = centroid + factor * direction
                    intersectionY = int(intersectionPoint[1])
                    intersectionX = int(intersectionPoint[2])
                    if (intersectionY < 0 or intersectionY >= dimension[1] or \
                        intersectionX < 0 or intersectionX >= dimension[2]):
                        indicator[beamIdx][2][segIdx] = True
                    elif topSlice[intersectionY, intersectionX] == False:
                        indicator[beamIdx][2][segIdx] = True

        # we only take the angles that are valid for all PTV segments
        valid = []
        for i in range(len(indicator)):
            angle_rad, direction, flags = indicator[i]
            if np.all(flags):
                angle_deg = angle_rad * 180 / np.pi
            valid.append(angle_deg)
        random.shuffle(valid)
        for i in range(4):
            localBeams = valid[i*nCandidatesPerSeg: (i+1)*nCandidatesPerSeg]
            content = []
            for entry in localBeams:
                content.append("{:.4f} {:.4f} {:.4f}".format(*entry))
            content = "\n".join(content)
            file = os.path.join(patientFolder, "beamListSeg{}.txt".format(i))
            with open(file, "w") as f:
                f.write(content)
            print(file)


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


def testBeamList():
    """
    This function tests whether the beam lists generated above are valid
    """
    isoRes = 2.5
    for patient in patients:
        patientFolder = os.path.join(RootFolder, patient)
        RTSTRUCT_file = os.path.join(patientFolder, "RTSTRUCT.nrrd")
        seg, header = nrrd.read(RTSTRUCT_file)

        # delete all mask - related entries
        idx = 0
        while True:
            beginning = "Segment{}_".format(idx)
            idx += 1
            key = beginning + "Color"
            if key not in header:
                break
            del header[key]
            del header[beginning + "ColorAutoGenerated"]
            del header[beginning + "Extent"]
            del header[beginning + "ID"]
            del header[beginning + "LabelValue"]
            del header[beginning + "Layer"]
            del header[beginning + "Name"]
            del header[beginning + "NameAutoGenerated"]
            del header[beginning + "Tags"]
        
        dimension = os.path.join(patientFolder, "metadata.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimensionFlip = np.flip(dimension)

        nStructures = 5  # SKIN and 4 sets of beams
        header["sizes"][:] = [nStructures, *dimensionFlip]
        header["space directions"][1, 0] = isoRes
        header["space directions"][2, 1] = isoRes
        header["space directions"][3, 2] = isoRes
        header = list(header.items())
        headerBegin = header[:8]
        headerEnd = header[8:]


        # initialize header
        order = ["SKIN"] + ["PTVSeg{}".format(i) for i in range(4)]
        globalExtent = "0 {} 0 {} 0 {}".format(*dimensionFlip)
        headerMiddle = []
        for idx, name in enumerate(order):
            color = hex_to_rgb(colors[idx])
            beginning = "Segment{}_".format(idx)
            localList = []
            localList.append((beginning + "Color", color))
            localList.append((beginning + "ColorAutoGenerated", "1"))
            localList.append((beginning + "Extent", globalExtent))
            localList.append((beginning + "ID", name))
            localList.append((beginning + "LabelValue", "1"))
            localList.append((beginning + "Layer", str(idx)))
            localList.append((beginning + "Name", name))
            localList.append((beginning + "NameAutoGenerated", 1))
            localList.append((beginning + "Tags", f"DicomRtImport.RoiNumber:{idx+1}|TerminologyEntry:Segmentation " \
                "category and type - 3D Slicer General Anatomy list~SCT^85756007^Tissue~SCT^85756007^Tissue~^^~" \
                "Anatomic codes - DICOM master list~^^~^^|"))
            headerMiddle.append((beginning, localList))
        headerMiddle.sort(key=lambda a: a[0])
        middle_result = []
        for _, localList in headerMiddle:
            middle_result.extend(localList)
        segHeaderNew = headerBegin + middle_result + headerEnd
        segHeaderNew = OrderedDict(segHeaderNew)
        if False:
            print(segHeaderNew)
            break

        maskFolder = os.path.join(patientFolder, "PlanMask")
        masks = {"SKIN": None, "PTVSeg0": None, "PTVSeg1": None,
            "PTVSeg2": None, "PTVSeg3": None}
        for name in masks:
            file = os.path.join(maskFolder, name+".bin")
            maskArray = np.fromfile(file, dtype=np.uint8)
            maskArray = np.reshape(maskArray, dimension)
            masks[name] = maskArray
        
        maskCentroidDict = {"PTVSeg{}".format(i): None for i in range(4)}
        for name in maskCentroidDict:
            maskArray = masks[name]
            maskCentroidDict[name] = calcCentroid(maskArray)
        
        coordsShape = dimension + (3, )
        coords = np.zeros(coordsShape, dtype=float)
        axis_x = np.arange(dimension[0])
        axis_x = np.expand_dims(axis_x, axis=(1, 2))
        axis_y = np.arange(dimension[1])
        axis_y = np.expand_dims(axis_y, axis=(0, 2))
        axis_z = np.arange(dimension[2])
        axis_z = np.expand_dims(axis_z, axis=(0, 1))
        coords[:, :, :, 0] = axis_x
        coords[:, :, :, 1] = axis_y
        coords[:, :, :, 2] = axis_z

        radius = 1
        beamsDict = {}
        for name, centroid in maskCentroidDict.items():
            centroid = np.expand_dims(centroid, axis=(0, 1, 2))
            coordsDiff = coords - centroid

            idx = "".join((a for a in name if a.isdigit()))
            beamList = os.path.join(patientFolder, "beamListSeg{}.txt".format(idx))
            with open(beamList, "r") as f:
                beamList = f.readlines()
            
            canvas = np.zeros(dimension, dtype=np.uint8)
            radius = 1
            for j in range(len(beamList)):
                angle = beamList[j]
                angle = angle.replace(" ", ", ")
                angle = np.array(eval(angle))
                angle *= np.pi / 180  # degree to rad
                axisBEV = np.array((0, 1, 0))
                axisPVCS = inverseRotateBeamAtOriginRHS(axisBEV, *angle)
                axisPVCS = np.expand_dims(axisPVCS, axis=(0, 1, 2))

                alongAxisProjection = np.sum(coordsDiff * axisPVCS, axis=3, keepdims=True)
                perpendicular = coordsDiff - alongAxisProjection * axisPVCS

                distance = np.linalg.norm(perpendicular, axis=3)
                localMask = distance < radius

                alongAxisProjection = np.squeeze(alongAxisProjection)
                localMask = np.logical_and(localMask, alongAxisProjection < 0)
                canvas = np.logical_or(canvas, localMask)
                print("patient {}, PTVSeg{}, beam {}".format(patient, idx, j))
            beamsDict[name] = canvas.astype(np.uint8)
        
        # add SKIN mask
        beamsDict["SKIN"] = masks["SKIN"]
        maskList = []
        for key in order:
            localMask = beamsDict[key]
            localMask = np.transpose(localMask, axes=(2, 1, 0))
            localMask = np.expand_dims(localMask, axis=0)
            maskList.append(localMask)
        segNew = np.concatenate(maskList, axis=0)
        file = os.path.join(patientFolder, "beamViewTest.nrrd")
        nrrd.write(file, segNew, segHeaderNew)
        print(file)
        break


def hex_to_rgb(hex_color):
    """Converts a color from hexadecimal format to RGB."""
    hex_color = hex_color.lstrip('#')
    result = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    result = np.array(result) / 255
    result = "{} {} {}".format(*result)
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


def structsInfoGen():
    """
    This function generates the "structures.json" and "StructureInfo.csv" files for all patients
    """
    PTVName = "PTVMerge"
    BBoxName = "SKIN"
    for patient in patients:
        patientFolder = os.path.join(RootFolder, patient)
        MaskFolder = os.path.join(patientFolder, "PlanMask")
        structures = [a.split(".")[0] for a in os.listdir(MaskFolder)]
        assert PTVName in structures and BBoxName in structures, \
            "Either PTV or BBox not in structures"
        structures.remove(PTVName)
        structures.remove(BBoxName)
        structures.insert(0, BBoxName)
        structures = [a.replace(" ", "") for a in structures]
        content = {
            "prescription": 70,
            "ptv": PTVName,
            "oar": structures
        }
        content = json.dumps(content, indent=4)

        FastDoseFolder = os.path.join(patientFolder, "FastDose")
        if not os.path.isdir(FastDoseFolder):
            os.mkdir(FastDoseFolder)
        contentFile = os.path.join(FastDoseFolder, "structures.json")
        with open(contentFile, "w") as f:
            f.write(content)

        auxiliary = ["PTVSeg0", "PTVSeg1", "PTVSeg2", "PTVSeg3", "SKIN"]
        structures = [a for a in structures if a not in auxiliary]
        PTVs = [a for a in structures if "PTV" in a]
        OARs = [a for a in structures if a not in PTVs]
        OARs.append("RingStructure")
        PTVDose = []
        for name in PTVs:
            dose = "".join(a for a in name if a.isdigit())
            dose = eval(dose)
            PTVDose.append((name, dose))
        PTVDose.sort(key=lambda a: a[1], reverse=True)
        content = "Name,maxWeights,maxDose,minDoseTargetWeights,minDoseTarget,OARWeights,IdealDose"
        for name, dose in PTVDose:
            line = "{},100,{},100,{},NaN,{}".format(name, dose, dose, dose)
            content = content + "\n" + line
        special = {"BRAIN": 0.5, "RingStructure": 0.5}
        for name in OARs:
            weight = 5
            if name in special:
                weight = special[name]
            line = "{},0,18,NaN,NaN,{},0".format(name, weight)
            content = content + "\n" + line
        contentFile = os.path.join(FastDoseFolder, "StructureInfo.csv")
        with open(contentFile, "w") as f:
            f.write(content)
        print("Patient {} done!".format(patient))


if __name__ == "__main__":
    StructsExclude()
    # CheckData()
    # postCheck()
    # MaskTrim()
    # PTVSeg()
    # ShowPTVSeg()
    # beamListGen()
    testBeamList()
    # structsInfoGen()