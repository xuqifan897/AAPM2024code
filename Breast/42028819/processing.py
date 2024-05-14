import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rt_utils import RTStructBuilder
from skimage import measure, transform
import json
import h5py
from scipy.interpolate import RegularGridInterpolator

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
colors[7] = colors[-1]
rootFolder = "/data/qifan/projects/FastDoseWorkplace/Breast"
patientName = "42028819"
targetResolution = np.array((2.5, 2.5, 2.5))  # mm

names_useful = ["A_LAD", "Breast_R", "Chestwall_L_experimental", "Chestwall_R_experimental",
                "Esophagus", "Glnd_Thyroid", "Heart", "Humerus_L", "Shoulder_L", "Liver",
                "Lung_L", 'Lung_R', "Nipple", "SpinalCord", "Ventricle_L", "PTV_PBI_L",
                "External", "d_Breast-PTV"]
PTVName = "PTV_PBI_L"
BodyName = "External"

def nameTrim(name: str):
    name = name.replace(" ", "")
    name = name.replace("-PTV", "")
    name = name.replace("(1)", "")
    return name

def initialProcessing():
    patientFolder = os.path.join(rootFolder, patientName)
    dicomFolder = os.path.join(patientFolder, "dicom")
    dicomFiles = os.listdir(dicomFolder)
    rtFile = None
    ctData = []
    RescaleSlope = None
    RescaleIntercept = None
    base = None
    HUmin = -1000
    for file in dicomFiles:
        file = os.path.join(dicomFolder, file)
        dataset = pydicom.dcmread(file)
        if dataset.Modality == "RTSTRUCT":
            rtFile = file
            continue
        if dataset.Modality == "CT":
            InstanceNumber = dataset.InstanceNumber
            InstanceNumber = int(InstanceNumber)
            if RescaleSlope is None or  RescaleIntercept is None:
                RescaleSlope = dataset.RescaleSlope
                RescaleIntercept = dataset.RescaleIntercept
            pixel_array = dataset.pixel_array
            pixel_array = pixel_array * RescaleSlope + RescaleIntercept
            pixel_array -= HUmin
            pixel_array[pixel_array < 0] = 0
            ctData.append((InstanceNumber, pixel_array))
    ctData.sort(key=lambda a: a[0])
    
    assert rtFile is not None, "rtFile not found"
    RTStruct = RTStructBuilder.create_from(
        dicom_series_path=dicomFolder, rt_struct_path=rtFile)
    StructureNames = RTStruct.get_roi_names()
    print(StructureNames)

    structures = {}
    PTVMask = getMask(PTVName, RTStruct)
    PTVMask = np.flip(PTVMask, axis=0)
    StructureNames = names_useful.copy()
    for name in StructureNames:
        mask = getMask(name, RTStruct)
        if mask is None:
            print("Failed loading structure {}".format(name))
            continue
        print("Loading structure {}".format(name))
        mask = np.flip(mask, axis=0)
        if name not in [PTVName, BodyName]:
            mask = np.logical_and(mask, np.logical_not(PTVMask))
        name_new = nameTrim(name)
        assert name_new not in structures, "Double definition of structure {}".format(name_new)
        structures[name_new] = mask

    viewFolder = os.path.join(patientFolder, "dicomView")
    if not os.path.isdir(viewFolder):
        os.mkdir(viewFolder)
    for i in range(len(ctData)):
        slice = ctData[i][1]
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.imshow(slice, cmap="gray", vmin=0, vmax=2000)
        for j, entry in enumerate(structures.items()):
            name, array = entry
            color = colors[j]
            slice = array[i, :, :]
            contours = measure.find_contours(slice)
            initial = True
            for contour in contours:
                if initial:
                    plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                    initial = False
                else:
                    plt.plot(contour[:, 1], contour[:, 0], color=color)
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
        plt.tight_layout()
        file = os.path.join(viewFolder, "{:03d}.png".format(i))
        plt.savefig(file, bbox_inches="tight")
        plt.clf()
        plt.close(fig)
        print(file)


def getMask(name, RTStruct, dimNew=None):
    try:
        mask = RTStruct.get_roi_mask_by_name(name)
    except:
        return None
    mask = np.transpose(mask, axes=(2, 0, 1))
    mask = mask.astype(np.float32)
    if dimNew is not None:
        mask = transform.resize(mask, dimNew)
    mask = mask > 1e-4
    mask = mask.astype(np.uint8)
    return mask


def structureGen():
    patientFolder = os.path.join(rootFolder, patientName)
    expFolder = os.path.join(patientFolder, "expFolder")
    if not os.path.isdir(expFolder):
        os.mkdir(expFolder)

    dicomFolder = os.path.join(patientFolder, "dicom")
    dicomFiles = os.listdir(dicomFolder)
    rtFile = None
    ctData = []
    RescaleSlope = None
    RescaleIntercept = None
    shape = None
    voxelSize = None
    HUmin = -1000
    for file in dicomFiles:
        file = os.path.join(dicomFolder, file)
        dataset = pydicom.dcmread(file)
        if dataset.Modality == "RTSTRUCT":
            rtFile = file
            continue
        if dataset.Modality == "CT":
            InstanceNumber = dataset.InstanceNumber
            InstanceNumber = int(InstanceNumber)
            pixel_array = dataset.pixel_array
            if RescaleSlope is None or  RescaleIntercept is None:
                RescaleSlope = dataset.RescaleSlope
                RescaleIntercept = dataset.RescaleIntercept
                shape = pixel_array.shape
                SliceThickness = dataset.SliceThickness
                PixelSpacing = dataset.PixelSpacing
                voxelSize = (float(SliceThickness), float(PixelSpacing[0]), float(PixelSpacing[1]))
                voxelSize = np.array(voxelSize)
            pixel_array = pixel_array * RescaleSlope + RescaleIntercept
            pixel_array -= HUmin
            pixel_array[pixel_array < 0] = 0
            pixel_array = pixel_array.astype(np.uint16)
            ctData.append((InstanceNumber, pixel_array))
    ctData.sort(key=lambda a: a[0])
    CTShape = (len(ctData), shape[0], shape[1])
    CTArray = np.zeros(CTShape, dtype=np.uint16)
    for i in range(len(ctData)):
        CTArray[i, :, :] = ctData[i][1]
    CTArray = np.flip(CTArray, axis=0)

    # calculate the new homogeneous phantom dimension
    CTSize = np.array(CTArray.shape) * voxelSize
    dimNew = CTSize / targetResolution
    dimNew = dimNew.astype(int)
    print(dimNew)
    print(np.min(CTArray), np.max(CTArray))
    CTArray = CTArray.astype(np.float32)
    print("Resizing ...")
    CTArray = transform.resize(CTArray, dimNew)
    CTArray = CTArray.astype(np.uint16)
    CTFile = os.path.join(expFolder, "density_raw.bin")
    CTArray.tofile(CTFile)
    print(dimNew)

    RTStruct = RTStructBuilder.create_from(
        dicom_series_path=dicomFolder, rt_struct_path=rtFile)
    structures = {}
    PTVMask = getMask(PTVName, RTStruct, dimNew)
    for name in names_useful:
        mask = getMask(name, RTStruct, dimNew)
        if name not in [PTVName, BodyName]:
            mask = np.logical_and(mask, np.logical_not(PTVMask))
        name_new = nameTrim(name)
        assert name_new not in structures, "Double definition of structure {}".format(name_new)
        structures[name_new] = mask
    maskFolder = os.path.join(expFolder, "MaskInput")
    if not os.path.isdir(maskFolder):
        os.mkdir(maskFolder)
    for name, array in structures.items():
        file = os.path.join(maskFolder, "{}.bin".format(name))
        array.tofile(file)
        print(file)


def structuresFileGen():
    patientFolder = os.path.join(rootFolder, patientName)
    expFolder = os.path.join(patientFolder, "expFolder")
    maskFolder = os.path.join(expFolder, "MaskInput")
    structures = os.listdir(maskFolder)
    structures = [a.split(".")[0] for a in structures]
    OARNames = [a for a in structures if a not in [PTVName, BodyName]]
    OARNames.insert(0, BodyName)
    content = {
        "prescription": 30,
        "ptv": PTVName,
        "oar": OARNames
    }
    content = json.dumps(content, indent=4)
    structuresFile = os.path.join(expFolder, "structures.json")
    with open(structuresFile, "w") as f:
        f.write(content)
    print(content)


def structuresInfoGen():
    PTVs = {PTVName: 30}
    Special = {"RingStructure": 2}
    exclude = [BodyName]
    expFolder = os.path.join(rootFolder, patientName, "expFolder")
    prep_output = os.path.join(expFolder, "prep_output")
    roi_list_file = os.path.join(prep_output, "roi_list.h5")
    dataset = h5py.File(roi_list_file, "r")
    structure_names = list(dataset.keys())
    
    total_exclude = list(PTVs.keys()) + exclude
    for key in total_exclude:
        assert key in structure_names, "The structure {} not found".format(key)
    OARList = [a for a in structure_names if a not in total_exclude]
    
    content = "Name,maxWeights,maxDose,minDoseTargetWeights,minDoseTarget,OARWeights,IdealDose"
    for name, dose in PTVs.items():
        newline = "{},100,{},100,{},NaN,{}".format(name, dose, dose, dose)
        content = content + "\n" + newline
    for name in OARList:
        if name in Special:
            continue
        newline = "{},0,18,NaN,NaN,5,0".format(name)
        content = content + "\n" + newline
    for name in Special:
        newline = "{},0,18,NaN,NaN,{},0".format(name, Special[name])
        content = content + "\n" + newline
    StructureInfoFile = os.path.join(expFolder, "StructureInfo.csv")
    with open(StructureInfoFile, "w") as f:
        f.write(content)
    print(content)


def drawDVHComp():
    patientFolder = os.path.join(rootFolder, patientName)
    expFolder = os.path.join(patientFolder, "expFolder")
    prep_output = os.path.join(expFolder, "prep_output")

    dimensionFile = os.path.join(prep_output, "dimension.txt")
    with open(dimensionFile, "r") as f:
        lines = f.readlines()
    doseShape = lines[0]
    doseShape = doseShape.split(" ")
    doseShape = [int(a) for a in doseShape]
    doseShape.reverse()
    doseShape = tuple(doseShape)

    densityFile = os.path.join(prep_output, "density.raw")
    density = np.fromfile(densityFile, dtype=np.float32)
    density = np.reshape(density, doseShape)

    doseFile = os.path.join(expFolder, "plan1", "dose.bin")
    doseArray = np.fromfile(doseFile, dtype=np.float32)
    doseArray = np.reshape(doseArray, doseShape)

    structuresFile = os.path.join(prep_output, "roi_list.h5")
    structures = getStructures(structuresFile)
    print("Structures Loaded")

    CTFolder = os.path.join(patientFolder, "dicom")
    dicomDoseFile = os.path.join(patientFolder, "RTDose.dcm")
    dicomDoseArray = examineDose(dicomDoseFile, CTFolder)
    dicomDoseArray = np.flip(dicomDoseArray, axis=0)
    dicomDoseArray = transform.resize(dicomDoseArray, doseShape)

    PTVMask = [a[1] for a in structures if a[0] == PTVName]
    assert len(PTVMask) == 1, "None or more than one PTV is found"
    PTVMask = PTVMask[0]
    PTVDoseOpt = doseArray[PTVMask]
    threshOpt = np.percentile(PTVDoseOpt, 5)
    print(threshOpt)
    doseArray = doseArray * 30 / threshOpt
    PTVDoseDicom = dicomDoseArray[PTVMask]
    threshDicom = np.percentile(PTVDoseDicom, 5)
    print(threshDicom)
    dicomDoseArray = dicomDoseArray * 30 / threshDicom

    # bring PTV forward
    exclude = [PTVName, BodyName, "RingStructure"]
    structures = [a for a in structures if a[0] not in exclude]
    structures.insert(0, (PTVName, PTVMask))

    # draw DVH
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, entry in enumerate(structures):
        color = colors[i]
        name, mask = entry
        mask = mask > 0
        optDose = doseArray[mask]
        optDose = np.sort(optDose)
        optDose = np.insert(optDose, 0, 0)
        dicomDose = dicomDoseArray[mask]
        dicomDose = np.sort(dicomDose)
        dicomDose = np.insert(dicomDose, 0, 0)
        y_axis = np.linspace(100, 0, dicomDose.size)
        plt.plot(optDose, y_axis, color=color, linestyle="-", label=name)
        plt.plot(dicomDose, y_axis, color=color, linestyle="--")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.title("DVH comparison patient {}".format(patientName))
    plt.xlabel("Dose (Gy)")
    plt.ylabel("Percentage")
    plt.tight_layout()
    figFile = os.path.join(patientFolder, "DVHComp.png")
    plt.savefig(figFile)


def getStructures(maskFile:str):
    dataset = h5py.File(maskFile, "r")
    structures_names = list(dataset.keys())
    result = []
    for struct_name in structures_names:
        struct = dataset[struct_name]
        structProps = struct["ArrayProps"]
        structMask = struct["mask"]

        structSize = structProps.attrs["size"]
        structCropSize = structProps.attrs["crop_size"]
        structCropStart = structProps.attrs["crop_start"]

        structSize = np.flip(structSize, axis=0)
        structCropSize = np.flip(structCropSize, axis=0)
        structCropStart = np.flip(structCropStart, axis=0)

        structMask = np.array(structMask)
        structMask = np.reshape(structMask, structCropSize)
        struct_mask = np.zeros(structSize, dtype=bool)
        struct_mask[structCropStart[0]: structCropStart[0] + structCropSize[0],
            structCropStart[1]: structCropStart[1] + structCropSize[1],
            structCropStart[2]: structCropStart[2] + structCropSize[2]] = structMask
        result.append((struct_name, struct_mask))
    return result


def examineDose(doseFile: str, CTFolder: str):
    CTData = []
    SourceFiles = os.listdir(CTFolder)
    ImOrienPatient_CT = None
    for file in SourceFiles:
        file = os.path.join(CTFolder, file)
        dataset = pydicom.dcmread(file)
        if dataset.Modality == "CT":
            InstanceNumber = int(dataset.InstanceNumber)
            if ImOrienPatient_CT is None:
                ImOrienPatient_CT = dataset.ImageOrientationPatient
            elif ImOrienPatient_CT != dataset.ImageOrientationPatient:
                print("ImageOrientationPatient attribute inconsistent")
            CTData.append((InstanceNumber, dataset))
    CTData.sort(key=lambda a: a[0])
    dimY, dimX = CTData[0][1].pixel_array.shape
    PixelSpacingY, PixelSpacingX = CTData[0][1].PixelSpacing
    CTcoordsShape = (dimX, dimY, len(CTData), 3)
    coords_array = np.zeros(CTcoordsShape, dtype=float)

    for i in range(len(CTData)):
        baseCoords = np.array(CTData[i][1].ImagePositionPatient)
        baseCoords = np.expand_dims(baseCoords, axis=(0, 1))
        coords_array[:, :, i, :] = baseCoords
        
    # Take into account the influence of the x coordinates
    vector_x = np.array(ImOrienPatient_CT[:3]) * PixelSpacingX
    vector_x = np.expand_dims(vector_x, axis=(0, 1, 2))
    voxelIdx = np.arange(dimX)
    voxelIdx = np.expand_dims(voxelIdx, axis=(1, 2, 3))
    offset_x = vector_x * voxelIdx

    # Take into account the influence of the y coordinates
    vector_y = np.array(ImOrienPatient_CT[3:]) * PixelSpacingY
    vector_y = np.expand_dims(vector_y, axis=(0, 1, 2))
    voxelIdx = np.arange(dimY)
    voxelIdx = np.expand_dims(voxelIdx, axis=(0, 2, 3))
    offset_y = vector_y * voxelIdx

    coords_array = coords_array + offset_x + offset_y


    doseDataset = pydicom.dcmread(doseFile)
    ImOrienPatient_dose = doseDataset.ImageOrientationPatient
    doseArray = doseDataset.pixel_array
    doseArray = np.transpose(doseArray, axes=(2, 1, 0))
    doseShape = doseArray.shape
    ImagePositionPatient_dose = doseDataset.ImagePositionPatient
    DosePixelSpacing = doseDataset.PixelSpacing
    GridFrameOffsetVector = doseDataset.GridFrameOffsetVector
    SliceThickness_Dose = GridFrameOffsetVector[1] - GridFrameOffsetVector[0]
    res_dose = (DosePixelSpacing[0], DosePixelSpacing[1], SliceThickness_Dose)
    res_dose = np.array(res_dose)
    doseCoordsX = np.arange(doseShape[0]) * res_dose[0] * ImOrienPatient_dose[0]
    doseCoordsY = np.arange(doseShape[1]) * res_dose[1] * ImOrienPatient_dose[4]
    sign_z = ImOrienPatient_dose[0] * ImOrienPatient_dose[4]
    doseCoordsZ = np.arange(doseShape[2]) * res_dose[2] * sign_z
    doseInterpFunc = RegularGridInterpolator(
        (doseCoordsX, doseCoordsY, doseCoordsZ), doseArray,
        bounds_error=False, fill_value=0.0)
    
    ImagePositionPatient_dose = np.expand_dims(ImagePositionPatient_dose, axis=(0, 1, 2))
    coords_array = coords_array - ImagePositionPatient_dose
    nPoints = CTcoordsShape[0] * CTcoordsShape[1] * CTcoordsShape[2]
    coords_array = np.reshape(coords_array, (nPoints, 3))
    doseValues = doseInterpFunc(coords_array)
    doseValues = np.reshape(doseValues, tuple(CTcoordsShape[:3]))
    doseValues = np.transpose(doseValues, axes=(2, 1, 0))
    return doseValues


if __name__ == "__main__":
    # initialProcessing()
    # structureGen()
    # structuresFileGen()
    # structuresInfoGen()
    drawDVHComp()