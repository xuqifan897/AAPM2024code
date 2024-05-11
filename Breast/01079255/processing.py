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
patientName = "01079255"
targetResolution = np.array((2.5, 2.5, 2.5))  # mm

names_useful = ["A_LAD", "Breast_R", "Chestwall_L_experimental", "Chestwall_R_experimental",
                "Esophagus", "Glnd_Thyroid", 'Heart', "Humerus_L", "Liver", "Lung_L", "Lung_R",
                "Nipple", "SpinalCord", "Sternum", "Stomach", "Body", "d_eval_PTV",
                "Shoulder_L", "d_Breast_L - PTV"]

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
    structureNames = RTStruct.get_roi_names()
    for a in names_useful:
        assert a in structureNames, "The structure {} not included in the structure list".format(a)
    structureNames = names_useful

    # There are so many structures, I'd like to take a look, to see which are relevant.
    structures = {}
    count = 0
    limit = 10
    for name in structureNames:
        try:
            mask = RTStruct.get_roi_mask_by_name(name)
        except:
            print("Failed to load structure {}".format(name))
            continue
        mask = np.transpose(mask, axes=(2, 0, 1))
        mask = np.flip(mask, axis=0)
        structures[name] = mask
        print("Loading structure {}".format(name))
        count += 1
        # if  count == limit:
        #     break

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
        print(file)


def closerLook():
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
    structureNames = RTStruct.get_roi_names()

    names = ["A_LAD", "Scar_Breast_L", "Scar_Axilla_L", "CTV_PBI_L", "PTV_PBI_L", "d_Breast_L - PTV",
             "d_conform", "d_eval_PTV", "d_low dose", "d_low conform", "TumorBed"]
    for name in names:
        assert name in structureNames, "{} is not included in structureNames".format(name)
    structureNames = names

    structures = {}
    count = 0
    limit = 10
    validNames = []
    for name in structureNames:
        try:
            mask = RTStruct.get_roi_mask_by_name(name)
        except:
            print("Failed to load structure {}".format(name))
            continue
        validNames.append(name)
        mask = np.transpose(mask, axes=(2, 0, 1))
        mask = np.flip(mask, axis=0)
        structures[name] = mask
        print("Loading structure {}".format(name))
        count += 1

    viewFolder = os.path.join(patientFolder, "closerView")
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
        print(file)
    print("Valid names: ", validNames)


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
    PTVName = "d_eval_PTV"
    PTVMask = getMask(PTVName, RTStruct, dimNew)
    for name in names_useful:
        mask = getMask(name, RTStruct, dimNew)
        if name != PTVName:
            mask = np.logical_and(mask, np.logical_not(PTVMask))
        name_new = name.replace(" ", "")
        assert name_new not in structures, "Double definition of structure {}".format(name_new)
        structures[name_new] = mask
    maskFolder = os.path.join(expFolder, "MaskInput")
    if not os.path.isdir(maskFolder):
        os.mkdir(maskFolder)
    for name, array in structures.items():
        file = os.path.join(maskFolder, "{}.bin".format(name))
        array.tofile(file)
        print(file)


def getMask(name, RTStruct, dimNew):
    mask = RTStruct.get_roi_mask_by_name(name)
    mask = np.transpose(mask, axes=(2, 0, 1))
    mask = mask.astype(np.float32)
    mask = transform.resize(mask, dimNew)
    mask = mask > 1e-4
    mask = mask.astype(np.uint8)
    return mask


def structuresFileGen():
    patientFolder = os.path.join(rootFolder, patientName)
    expFolder = os.path.join(patientFolder, "expFolder")
    maskFolder = os.path.join(expFolder, "MaskInput")
    structures = os.listdir(maskFolder)
    structures = [a.split(".")[0] for a in structures]
    PTVName = "PTV_PBI_L"
    BodyName = "Body"
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
    PTVs = {"PTV_PBI_L": 30}
    Special = {"RingStructure": 2}
    exclude = ["Body", "d_eval_PTV"]
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
        weight = 2
        if name in Special:
            continue
        newline = "{},0,18,NaN,NaN,5,0".format(name)
        content = content + "\n" + newline
    for name in Special:
        newline = "{},0,18,NaN,NaN,5,{}".format(name, Special[name])
        content = content + "\n" + newline
    StructureInfoFile = os.path.join(expFolder, "StructureInfo.csv")
    with open(StructureInfoFile, "w") as f:
        f.write(content)
    print(content)


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


def drawDVH():
    expFolder = os.path.join(rootFolder, patientName, "expFolder")
    prep_output_folder = os.path.join(expFolder, "prep_output")
    dimensionFile = os.path.join(prep_output_folder, "dimension.txt")
    optResultFolder = os.path.join(expFolder, "plan1")
    with open(dimensionFile, "r") as f:
        lines = f.readlines()
    doseShape = lines[0]
    doseShape = doseShape.split(" ")
    doseShape = [int(a) for a in doseShape]
    doseShape.reverse()
    doseShape = tuple(doseShape)
    
    roi_listFile = os.path.join(prep_output_folder, "roi_list.h5")
    doseFile = os.path.join(optResultFolder, "dose.bin")

    doseArray = np.fromfile(doseFile, dtype=np.float32)
    doseArray = np.reshape(doseArray, doseShape)
    doseArray /= np.max(doseArray)  # normalize

    exclude = ["Body", "RingStructure"]
    PrimaryPTV = "d_eval_PTV"
    Masks = getStructures(roi_listFile)
    Masks = [a for a in Masks if a[0] not in exclude]
    primaryMask = None
    for name, mask in Masks:
        if name == PrimaryPTV:
            primaryMask = mask
            break

    primaryMask = primaryMask > 0
    primaryDose = doseArray[primaryMask]
    thresh = np.percentile(primaryDose, 5)
    doseArray = doseArray / thresh * 30

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, entry in enumerate(Masks):
        color = colors[i]
        name, mask = entry
        mask = mask > 0
        struct_dose = doseArray[mask]
        struct_dose = np.sort(struct_dose)
        struct_dose = np.insert(struct_dose, 0, 0.0)
        numPoints = struct_dose.size
        percentile = np.linspace(100, 0, numPoints)
        ax.plot(struct_dose, percentile, color=color, label=name)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.xlim(0, 40)
    plt.xlabel("Dose (Gy)")
    plt.ylabel("Percentage (%)")
    plt.title("DVH {}".format(patientName))
    DVH_opt_file = os.path.join(optResultFolder, "DVH_opt.png")
    plt.savefig(DVH_opt_file)
    plt.clf()
    print(DVH_opt_file)


def drawDoseWash():
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

    dose_vmax = 90
    imageFolder = os.path.join(expFolder, "doseWash")
    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)
    nSlices = doseShape[0]
    for i in range(nSlices):
        fig, ax = plt.subplots(figsize=(8, 6))
        slice = density[i, :, :]
        plt.imshow(slice, cmap="gray", vmin=0, vmax=2.0)
        slice = doseArray[i, :, :]
        plt.imshow(slice, cmap="jet", vmin=0, vmax=40, alpha=0.3)
        for j, entry in enumerate(structures):
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
        file = os.path.join(imageFolder, "{:03d}.png".format(i))
        plt.savefig(file, bbox_inches="tight")
        plt.clf()
        print(file)


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

    PTVName = "d_eval_PTV"
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
    exclude = [PTVName, "PTV_PBI_L"]
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


def readDicomStructure(dicomFolder: str):
    RTFile = None
    files = os.listdir(dicomFolder)
    for file in files:
        file = os.path.join(dicomFolder, file)
        dataset = pydicom.dcmread(file)
        if dataset.Modality == "RTSTRUCT":
            RTFile = file
            break
    assert RTFile is not None, "RTFile not found"
    RTStruct = RTStructBuilder.create_from(dicom_series_path=dicomFolder, rt_struct_path=RTFile)
    structureNames = RTStruct.get_roi_names()
    result = {}
    for name in structureNames:
        try: 
            mask = RTStruct.get_roi_mask_by_name(name)
        except:
            print("Error loading structure {}".format(name))
            continue
        print("Loading structure {}".format(name))
        result[name] = mask
    return result


def examineDose(doseFile: str, CTFolder: str):
    CTData = []
    SourceFiles = os.listdir(CTFolder)
    ImageOrientationPatient = None
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
    CTData.sort(key=lambda a: a[0])
    numSlices = len(CTData)
    shape_CT = (shape[0], shape[1], numSlices)
    coordsShape = shape_CT + (3,)
    coords_array = np.zeros(coordsShape, dtype=float)

    ImagePositionList = [a[1] for a in CTData]
    
    ImagePositionPatient = CTData[0][1]
    coords_x = ImagePositionPatient[0] + np.arange(shape[0]) * PixelSpacing[0]
    coords_y = ImagePositionPatient[1] + np.arange(shape[1]) * PixelSpacing[1]
    coords_z = np.zeros(numSlices, dtype=coords_x.dtype)
    for i in range(numSlices):
        coords_z[i] = CTData[i][1][2]
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
    doseValues = np.transpose(doseValues, axes=(2, 1, 0))
    return doseValues


if __name__ == "__main__":
    # initialProcessing()
    # closerLook()
    # structureGen()
    # structuresFileGen()
    # structuresInfoGen()
    # drawDVH()
    drawDVHComp()
    # drawDoseWash()