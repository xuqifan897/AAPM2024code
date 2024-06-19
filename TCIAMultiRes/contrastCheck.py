import os
import glob
import numpy as np
import random
import json
import pydicom
from rt_utils import RTStructBuilder
from scipy.interpolate import RegularGridInterpolator
import nrrd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import measure, transform

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
normThresh = 10

Folder1 = "/data/qifan/projects/FastDoseWorkplace/TCIASupp"
Folder2 = "/data/qifan/projects/FastDoseWorkplace/TCIAMultiRes"
patients = ["002", "003", "009", "013", "070", "125", "132", "190"]

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

def getFullDose():
    for patient in patients:
        Folder1Patient = os.path.join(Folder1, patient)
        Folder1CTFolder = os.path.join(Folder1Patient, "data")
        Folder1Dose = os.path.join(Folder1Patient, "dose.dcm")
        CTArray, doseValues, VoxelSize, RescaleSlope, RescaleIntercept \
            = getCTandDose(Folder1CTFolder, Folder1Dose)
        doseValues = doseValues.astype(np.float32)

        doseValues = np.flip(doseValues, axis=2)
        doseValues = np.transpose(doseValues, axes=(1, 0, 2))

        targetFile = os.path.join(Folder2, patient, "doseFullRes.bin")
        doseValues.tofile(targetFile)
        print(targetFile)


def getCTandDose(CTFolder, doseFile):
    """
    This function gets aligned CT array and dose array
    Returns CTArray, DoseArray, VoxelSize in the order (height, width, slice)
    """
    CTData = []
    SourceFiles = os.listdir(CTFolder)
    ImageOrientationPatient = None
    SliceThickness = None
    for file in SourceFiles:
        path = os.path.join(CTFolder, file)
        dataset = pydicom.dcmread(path)
        if dataset.Modality == "CT":
            InstanceNumber = int(dataset.InstanceNumber)
            CTData.append((InstanceNumber, dataset.ImagePositionPatient, dataset.pixel_array))
            if ImageOrientationPatient is None:
                ImageOrientationPatient = dataset.ImageOrientationPatient
                shape = CTData[0][2].shape
                SliceThickness = dataset.SliceThickness
                PixelSpacing = dataset.PixelSpacing
                RescaleSlope = dataset.RescaleSlope
                RescaleIntercept = dataset.RescaleIntercept
            else:
                assert SliceThickness == dataset.SliceThickness

    CTData.sort(key=lambda a: a[0])
    numSlices = len(CTData)
    shape_CT = (shape[0], shape[1], numSlices)
    coordsShape = shape_CT + (3,)
    coords_array = np.zeros(coordsShape, dtype=float)

    ImagePositionPatient = CTData[0][1]
    coords_x = ImagePositionPatient[0] + np.arange(shape[0]) * PixelSpacing[0]
    coords_y = ImagePositionPatient[1] + np.arange(shape[1]) * PixelSpacing[1]
    coords_z = np.zeros(numSlices, dtype=coords_x.dtype)
    for i in range(numSlices):
        coords_z[i] = CTData[i][1][2]

    if float(SliceThickness) == 0.0:
        # deal with special caase
        coords_z_diff = np.diff(coords_z)
        SliceThickness = coords_z_diff[0]
        flag = coords_z_diff == SliceThickness
        flag = np.all(flag)
        assert flag

    coords_x = np.expand_dims(coords_x, axis=(1, 2))
    coords_y = np.expand_dims(coords_y, axis=(0, 2))
    coords_z = np.expand_dims(coords_z, axis=(0, 1))
    coords_array[:, :, :, 0] = coords_x
    coords_array[:, :, :, 1] = coords_y
    coords_array[:, :, :, 2] = coords_z

    doseDataset = pydicom.dcmread(doseFile)
    GridFrameOffsetVector = doseDataset.GridFrameOffsetVector
    ImagePositionPatientDose = doseDataset.ImagePositionPatient
    PixelSpacing_Dose = doseDataset.PixelSpacing
    doseArray = doseDataset.pixel_array
    doseArray = np.transpose(doseArray, axes=(2, 1, 0))
    shape_Dose = doseDataset.pixel_array.shape
    shape_Dose = (shape_Dose[2], shape_Dose[1], shape_Dose[0])
    SliceThickness_Dose = GridFrameOffsetVector[1] - GridFrameOffsetVector[0]

    ImagePositionPatientDose = np.array(ImagePositionPatientDose)
    ImagePositionPatientDose = np.expand_dims(ImagePositionPatientDose, axis=(0, 1, 2))
    coords_array -= ImagePositionPatientDose
    res_dose = (PixelSpacing_Dose[0], PixelSpacing_Dose[1], SliceThickness_Dose)
    res_dose = np.array(res_dose)
    res_dose = np.expand_dims(res_dose, axis=(0, 1, 2))
    coords_array /= res_dose

    nPoints = shape_CT[0] * shape_CT[1] * shape_CT[2]
    coords_array = np.reshape(coords_array, (nPoints, 3))

    doseCoordsX = np.arange(shape_Dose[0])
    doseCoordsY = np.arange(shape_Dose[1])
    doseCoordsZ = np.arange(shape_Dose[2])
    doseInterpFunc = RegularGridInterpolator(
        (doseCoordsX, doseCoordsY, doseCoordsZ), doseArray,
        bounds_error=False, fill_value=0.0)
    doseValues = doseInterpFunc(coords_array)
    doseValues = np.reshape(doseValues, shape_CT)
    doseValues = np.transpose(doseValues, axes=(1, 0, 2))

    CTArray = np.zeros(shape_CT, dtype=CTData[0][2].dtype)
    for i in range(numSlices):
        CTArray[:, :, i] = CTData[i][2]

    VoxelSize = np.array((PixelSpacing[0], PixelSpacing[1], SliceThickness))
    return CTArray, doseValues, VoxelSize, RescaleSlope, RescaleIntercept


def drawDoseWash():
    RescaleIntercept = -1024
    for patient in patients:
        patientFolder = os.path.join(Folder2, patient)
        CTArray = os.path.join(patientFolder, "CT.nrrd")
        CTArray, CTHeader = nrrd.read(CTArray)
        dimOrg = CTHeader["sizes"]
        CTArray = np.reshape(CTArray, dimOrg)
        CTArray -= RescaleIntercept  # (x, y, z)

        doseFullRes = os.path.join(patientFolder, "doseFullRes.bin")
        doseFullRes = np.fromfile(doseFullRes, dtype=np.float32)
        doseFullRes = np.reshape(doseFullRes, dimOrg)

        segFile = os.path.join(patientFolder, "RTSTRUCT.nrrd")
        masks = readSegFile(segFile)

        viewFolder = os.path.join(patientFolder, "fullResView")
        if not os.path.isdir(viewFolder):
            os.mkdir(viewFolder)

        doseRoof = np.max(doseFullRes)
        for i in range(dimOrg[2]):
            CTSlice = CTArray[:, :, i]
            doseSlice = doseFullRes[:, :, i]
            plt.imshow(CTSlice, cmap="gray", vmin=500, vmax=1500)

            for j, entry in enumerate(masks.items()):
                color = colors[j]
                name, maskArray = entry
                maskSlice = maskArray[:, :, i]
                if np.any(maskSlice) == 0:
                    continue
                contours = measure.find_contours(maskSlice)
                initial = True
                for contour in contours:
                    if initial:
                        plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                        initial = False
                    else:
                        plt.plot(contour[:, 1], contour[:, 0], color=color)

            plt.imshow(doseSlice, cmap="jet", vmin=0, vmax=doseRoof, alpha=0.3)
            plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
            plt.tight_layout()
            file = os.path.join(viewFolder, "{:03d}.png".format(i))
            plt.savefig(file)
            plt.clf()
            print(file)
        break


def readSegFile(file: str):
    seg, header = nrrd.read(file)
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


def convert2binary():
    """
    This function converts CT, RT, and dose to binary
    """
    RescaleIntercept = -1024
    for patient in patients:
        patientFolder = os.path.join(Folder2, patient)
        CTFile = os.path.join(patientFolder, "CT.nrrd")
        RTFile = os.path.join(patientFolder, "RTSTRUCT.nrrd")
        doseFile = os.path.join(patientFolder, "doseFullRes.bin")

        resultFolder = os.path.join(patientFolder, "fullRes")
        if not os.path.isdir(resultFolder):
            os.mkdir(resultFolder)

        CTArray, CTHeader = nrrd.read(CTFile)
        dimOrg = CTHeader["sizes"]
        dimTranspose = np.flip(dimOrg)
        content = "{} {} {}".format(*dimTranspose)
        file = os.path.join(resultFolder, "dimension.txt")
        with open(file, "w") as f:
            f.write(content)

        masks = readSegFile(RTFile)
        dose = np.fromfile(doseFile, dtype=np.float32)
        dose = np.reshape(dose, dimOrg)

        CTArray -= RescaleIntercept
        CTArray = np.transpose(CTArray, axes=(2, 1, 0)).astype(np.uint16)
        densityFile = os.path.join(resultFolder, "density_raw.bin")
        CTArray.tofile(densityFile)
        print(densityFile)

        dose = np.transpose(dose, axes=(2, 1, 0))
        doseFile = os.path.join(resultFolder, "doseRef.bin")
        dose.tofile(doseFile)
        print(doseFile)

        MaskFolder = os.path.join(resultFolder, "InputMask")
        if not os.path.isdir(MaskFolder):
            os.mkdir(MaskFolder)

        localExclude = StructsMetadata[patient]["exclude"]
        for name in masks:
            name_shorten = name.replace(" ", "")
            if name_shorten in localExclude:
                continue
            maskArray = masks[name]
            maskArray = np.transpose(maskArray, axes=(2, 1, 0))
            maskArray = maskArray.astype(np.uint8)
            file = os.path.join(MaskFolder, name_shorten + ".bin")
            maskArray.tofile(file)
            print(file)


def MaskTrim():
    "This function merges PTV masks of the same dose, and crops masks so that PTV masks do not " \
    "overlap with each other, and OAR masks do not overlap with PTV masks"
    for patient in patients:
        PatientFolder = os.path.join(Folder2, patient, "fullRes")
        InputMaskFolder = os.path.join(PatientFolder, "InputMask")
        OutputMaskFolder = os.path.join(PatientFolder, "PlanMask")
        if not os.path.isdir(OutputMaskFolder):
            os.mkdir(OutputMaskFolder)
        
        ExcludeList = (a:=StructsMetadata[patient])["exclude"]
        structures = [b for a in os.listdir(InputMaskFolder) if (b:=a.split(".")[0]) not in ExcludeList]
        PTVList = [a for a in structures if "ptv" in a.lower() or "ctv" in a.lower()]
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


def binaryView():
    for patient in patients:
        patientFolder = os.path.join(Folder2, patient, "fullRes")
        dimension = os.path.join(patientFolder, "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)

        density = os.path.join(patientFolder, "density_raw.bin")
        density = np.fromfile(density, dtype=np.uint16)
        density = np.reshape(density, dimension)

        dose = os.path.join(patientFolder, "doseRef.bin")
        dose = np.fromfile(dose, dtype=np.float32)
        dose = np.reshape(dose, dimension)

        masks = {}
        localExclude = ["PTVSeg0", "PTVSeg1", "PTVSeg2", "PTVSeg3", "PTVMerge", "SKIN"]
        maskFolder = os.path.join(patientFolder, "PlanMask")
        for file in os.listdir(maskFolder):
            name = file.split(".")[0]
            if name in localExclude:
                continue
            file = os.path.join(maskFolder, file)
            maskArray = np.fromfile(file, dtype=np.uint8)
            maskArray = np.reshape(maskArray, dimension)
            masks[name] = maskArray
        
        primaryMask = "PTV70"
        primaryMask = masks[primaryMask] > 0
        thresh = dose[primaryMask]
        thresh = np.percentile(thresh, normThresh)
        factor = 70 / thresh
        dose *= factor

        figureFolder = os.path.join(patientFolder, "view")
        if not os.path.isdir(figureFolder):
            os.mkdir(figureFolder)
        
        for i in range(dimension[0]):
            densitySlice = density[i, :, :]
            doseSlice = dose[i, :, :]
            plt.imshow(densitySlice, cmap="gray", vmin=500, vmax=2000)

            for j, entry in enumerate(masks.items()):
                color = colors[j]
                name, maskArray = entry
                maskSlice = maskArray[i, :, :]
                if np.any(maskSlice) == 0:
                    continue
                contours = measure.find_contours(maskSlice)
                initial = True
                for contour in contours:
                    if initial:
                        plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                        initial = False
                    else:
                        plt.plot(contour[:, 1], contour[:, 0], color=color)

            plt.imshow(doseSlice, cmap="jet", vmin=0, vmax=85, alpha=0.3)
            plt.colorbar()
            plt.legend(loc="top right", bbox_to_anchor=(1.02, 1))
            figureFile = os.path.join(figureFolder, "{:03d}.png".format(i))
            plt.savefig(figureFile)
            plt.clf()
            print(figureFile)
        break


def drawBinaryDVH():
    for patient in patients:
        patientFolder = os.path.join(Folder2, patient, "fullRes")
        dimension = os.path.join(patientFolder, "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)

        density = os.path.join(patientFolder, "density_raw.bin")
        density = np.fromfile(density, dtype=np.uint16)
        density = np.reshape(density, dimension)

        dose = os.path.join(patientFolder, "doseRef.bin")
        dose = np.fromfile(dose, dtype=np.float32)
        dose = np.reshape(dose, dimension)

        masks = {}
        localExclude = ["PTVSeg0", "PTVSeg1", "PTVSeg2", "PTVSeg3", "PTVMerge", "SKIN"]
        maskFolder = os.path.join(patientFolder, "PlanMask")
        for file in os.listdir(maskFolder):
            name = file.split(".")[0]
            if name in localExclude:
                continue
            file = os.path.join(maskFolder, file)
            maskArray = np.fromfile(file, dtype=np.uint8)
            maskArray = np.reshape(maskArray, dimension)
            masks[name] = maskArray
        
        primaryMask = "PTV70"
        primaryMask = masks[primaryMask] > 0
        thresh = dose[primaryMask]
        thresh = np.percentile(thresh, normThresh)
        factor = 70 / thresh
        dose *= factor

        fig = plt.figure(figsize=(8, 5))
        for i, entry in enumerate(masks.items()):
            color = colors[i]
            name, mask = entry
            mask = mask > 0
            struct_dose = dose[mask]
            struct_dose = np.sort(struct_dose)
            struct_dose = np.insert(struct_dose, 0, 0.0)
            numPoints = struct_dose.size
            percentile = np.linspace(100, 0, numPoints)
            plt.plot(struct_dose, percentile, color=color, label=name)

        plt.xlim(0, 90)
        plt.xlabel("Dose (Gy)")
        plt.ylabel("Percent Volume (%)")
        plt.title("Dose Volume Histogram of patient {}".format(patient))
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        plt.tight_layout()
        figureFile = os.path.join(patientFolder, "DVHFullRes.png")
        plt.savefig(figureFile)
        plt.close(fig)
        plt.clf()
        print(figureFile)


def PTVSeg():
    """
    This function follows the method proposed by Qihui et cl in the paper
    "Many-isocenter optimization for robotic radiotherpay"
    """
    for patient in patients:
        PatientFolder = os.path.join(Folder2, patient, "fullRes")
        metadata = os.path.join(PatientFolder, "dimension.txt")
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


def globalResize():
    resNew = 5 #  mm
    for patient in patients:
        # skip the cases that have been processed
        if patient in ["002", "003", "009"]:
            continue
        # find the original resolution
        folder1Patient = os.path.join(Folder1, patient, "data")
        files = os.listdir(folder1Patient)
        random.shuffle(files)
        file = os.path.join(folder1Patient, files[0])
        assert os.path.isfile(file), file
        dataset = pydicom.dcmread(file)
        SliceThickness = dataset.SliceThickness

        PixelSpacing = dataset.PixelSpacing
        resOrg = [float(SliceThickness), float(PixelSpacing[1]), float(PixelSpacing[0])]
        resOrg = np.array(resOrg)

        folder2Patient = os.path.join(Folder2, patient, "fullRes")
        dimensionOrg = os.path.join(folder2Patient, "dimension.txt")
        with open(dimensionOrg, "r") as f:
            dimensionOrg = f.readline()
        dimensionOrg = dimensionOrg.replace(" ", ", ")
        dimensionOrg = eval(dimensionOrg)
        dimensionOrg = np.array(dimensionOrg)

        dimensionNew = resOrg * dimensionOrg / resNew
        dimensionNew = dimensionNew.astype(int)
        print("patient {}: {} {} -> {}".format(patient, resOrg, dimensionOrg, dimensionNew))
        
        resFolder = os.path.join(Folder2, patient, "folderRes{}".format(resNew))
        if not os.path.isdir(resFolder):
            os.mkdir(resFolder)
        content = "{} {} {}".format(*dimensionNew)
        dimensionFile = os.path.join(resFolder, "dimension.txt")
        with open(dimensionFile, "w") as f:
            f.write(content)
        
        densityOrg = os.path.join(folder2Patient, "density_raw.bin")
        densityOrg = np.fromfile(densityOrg, dtype=np.uint16)
        densityOrg = np.reshape(densityOrg, dimensionOrg).astype(np.float32)
        densityNew = transform.resize(densityOrg, dimensionNew).astype(np.uint16)
        densityFile = os.path.join(resFolder, "density_raw.bin")
        densityNew.tofile(densityFile)
        print(densityFile)

        doseOrg = os.path.join(folder2Patient, "doseRef.bin")
        doseOrg = np.fromfile(doseOrg, dtype=np.float32)
        doseOrg = np.reshape(doseOrg, dimensionOrg)
        doseNew = transform.resize(doseOrg, dimensionNew).astype(np.float32)
        doseFile = os.path.join(resFolder, "doseRef.bin")
        doseNew.tofile(doseFile)
        print(doseFile)

        newMaskFolder = os.path.join(resFolder, "PlanMask")
        if not os.path.isdir(newMaskFolder):
            os.mkdir(newMaskFolder)
        maskFolder = os.path.join(folder2Patient, "PlanMask")
        for base in os.listdir(maskFolder):
            file = os.path.join(maskFolder, base)
            maskArray = np.fromfile(file, dtype=np.uint8)
            maskArray = np.reshape(maskArray, dimensionOrg).astype(np.float32)
            newMaskArray = transform.resize(maskArray, dimensionNew)
            newMaskArray = (newMaskArray >= 0.5).astype(np.uint8)
            newFile = os.path.join(newMaskFolder, base)
            newMaskArray.tofile(newFile)
            print(newFile)


def resizeDVH():
    """
    This function draws the ptv for the resized masks
    """
    resNew = 5  # mm
    for patient in patients:
        patientFolder = os.path.join(Folder2, patient, "folderRes{}".format(resNew))
        doseFile = os.path.join(patientFolder, "doseRef.bin")
        doseArray = np.fromfile(doseFile, dtype=np.float32)

        maskFolder = os.path.join(patientFolder, "PlanMask")
        masks = {}
        for file in os.listdir(maskFolder):
            name = file.split(".")[0]
            if name in ["PTVSeg0", "PTVSeg1", "PTVSeg2", "PTVSeg3", "PTVMerge", "SKIN"]:
                continue
            file = os.path.join(maskFolder, file)
            maskArray = np.fromfile(file, dtype=np.uint8)
            assert maskArray.size == doseArray.size
            masks[name] = maskArray > 0
        
        primaryMask = "PTV70"
        primaryMask = masks[primaryMask]
        thresh = doseArray[primaryMask]
        thresh = np.percentile(thresh, normThresh)
        doseArray *= 70 / thresh

        fig = plt.figure(figsize=(8, 5))
        for i, entry in enumerate(masks.items()):
            color = colors[i]
            name, mask = entry
            mask = mask > 0
            struct_dose = doseArray[mask]
            struct_dose = np.sort(struct_dose)
            struct_dose = np.insert(struct_dose, 0, 0.0)
            numPoints = struct_dose.size
            percentile = np.linspace(100, 0, numPoints)
            plt.plot(struct_dose, percentile, color=color, label=name)

        plt.xlim(0, 90)
        plt.xlabel("Dose (Gy)")
        plt.ylabel("Percent Volume (%)")
        plt.title("Dose Volume Histogram of patient {}".format(patient))
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        plt.tight_layout()
        figureFile = os.path.join(patientFolder, "DVHRes{}.png".format(resNew))
        plt.savefig(figureFile)
        plt.close(fig)
        plt.clf()
        print(figureFile)


def structsInfoGen():
    """
    This function generates the "structures.json" and "StructureInfo.csv" files for all patients
    """
    newRes = 5
    PTVName = "PTVMerge"
    BBoxName = "SKIN"
    for patient in patients:
        patientFolder = os.path.join(Folder2, patient, "folderRes{}".format(newRes))
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
            weight = 3
            if name in special:
                weight = special[name]
            line = "{},0,18,NaN,NaN,{},0".format(name, weight)
            content = content + "\n" + line
        contentFile = os.path.join(FastDoseFolder, "StructureInfo.csv")
        with open(contentFile, "w") as f:
            f.write(content)
        print("Patient {} done!".format(patient))


def beamListGen():
    """
    Due to memory limit, we can not use the full set of beams
    """
    samplingRatio = 0.4
    beamListFullPath = os.path.join(Folder2, "beamlist.txt")
    with open(beamListFullPath, "r") as f:
        lines = f.readlines()
    lines = [a.replace("\n", "") for a in lines]
    numBeams = len(lines)
    numBeamsSelect = int(numBeams * samplingRatio)
    numSeg = 4
    for i in range(numSeg):
        random.shuffle(lines)
        split1 = lines[:numBeamsSelect]
        split1 = "\n".join(split1)
        split1File = os.path.join(Folder2, "beamlistSeg{}.txt".format(i))
        with open(split1File, "w") as f:
            f.write(split1)


if __name__ == "__main__":
    StructsExclude()
    # getFullDose()
    # drawDoseWash()
    # convert2binary()
    # MaskTrim()
    # binaryView()
    # drawBinaryDVH()
    # PTVSeg()
    # globalResize()
    # resizeDVH()
    # structsInfoGen()
    beamListGen()