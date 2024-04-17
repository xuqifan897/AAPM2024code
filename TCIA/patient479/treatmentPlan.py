import os
import numpy as np
import pydicom
import nibabel as nib
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import transform, measure
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
import json

def examineDose_new():
    """The interpretation above was not correct"""
    patientFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/479"
    doseFile = os.path.join(patientFolder, "dose.dcm")
    CTFolder = os.path.join(patientFolder, "data")
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
    doseValues = np.transpose(doseValues, axes=(1, 0, 2))
    doseInterpPath = os.path.join(patientFolder, "doseExp1.npy")
    np.save(doseInterpPath, doseValues)


def dcm2nii():
    """
    This function converts the dicom format into nii
    """
    patientFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/479"
    dcmFolder = os.path.join(patientFolder, "data")
    CTData = []
    SourceFiles = os.listdir(dcmFolder)
    RescaleSlope = None
    for file in SourceFiles:
        path = os.path.join(dcmFolder, file)
        dataset = pydicom.dcmread(path)
        if dataset.Modality == "CT":
            InstanceNumber = int(dataset.InstanceNumber)
            CTData.append((InstanceNumber, dataset.pixel_array))
            if RescaleSlope is None:
                RescaleSlope = dataset.RescaleSlope
                RescaleIntercept = dataset.RescaleIntercept
                shape = CTData[0][1].shape
                datatype = CTData[0][1].dtype
    CTData.sort(key=lambda a: a[0])
    print("Checkpoint 1")

    numSlices = len(CTData)
    arrayShape = (numSlices, shape[0], shape[1])
    DataArray = np.zeros(arrayShape, datatype)
    for i in range(numSlices):
        DataArray[i, :, :] = CTData[i][1]
    DataArray = DataArray * RescaleSlope + RescaleIntercept
    DataArray = DataArray[:, 130: -150, 140: -140]
    newShape = (numSlices, 512, 512)
    DataArray = transform.resize(DataArray, newShape)

    DataArray = np.transpose(DataArray, axes=(2, 1, 0))
    nifti_img = nib.Nifti1Image(DataArray, affine=np.eye(4))
    targetFile = os.path.join(patientFolder, "HeadNeck.nii")
    nib.save(nifti_img, targetFile)
    print(nifti_img)


def examine_nifti():
    patientFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/479"
    sourceFile = os.path.join(patientFolder, "HeadNeck.nii")
    image = nib.load(sourceFile).get_fdata()
    image = np.transpose(image, axes=(2, 1, 0))
    targetFolder = os.path.join(patientFolder, "niiView")
    if not os.path.isdir(targetFolder):
        os.mkdir(targetFolder)
    numSlice = image.shape[0]
    for i in range(numSlice):
        slice = image[i, :, :]
        file = os.path.join(targetFolder, "{:03d}.png".format(i))
        plt.imsave(file, slice, cmap="gray")
        print(file)


def examineMask():
    patientFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/479"
    maskFile = os.path.join(patientFolder, "Prediction_HeadNeck.nii")
    maskArray = nib.load(maskFile).get_fdata()
    maskArray = maskArray > 0
    maskArray = np.transpose(maskArray, axes=(2, 1, 0))
    maskArray = brainMaskProcessing(maskArray)

    if True:
        # Fill the holes in the mask slicewise
        for i in range(maskArray.shape[0]):
            slice = maskArray[i, :, :].astype(bool)
            slice = ndimage.binary_fill_holes(slice)
            maskArray[i, :, :] = slice

    imageFile = os.path.join(patientFolder, "HeadNeck.nii")
    imageArray = nib.load(imageFile).get_fdata()
    imageArray = np.transpose(imageArray, axes=(2, 1, 0))

    folder = os.path.join(patientFolder, "examineMask")
    if not os.path.isdir(folder):
        os.mkdir(folder)

    numSlices = maskArray.shape[0]
    for i in range(numSlices):
        plt.imshow(imageArray[i, :, :], cmap="gray")
        maskSlice = maskArray[i, :, :]
        plt.imshow(maskSlice, alpha=0.4)
        file = os.path.join(folder, "{:03d}.png".format(i))
        plt.savefig(file)
        plt.clf()
        print(file)


def brainMaskProcessing(input: np.ndarray) -> np.ndarray:
    dtype_org = input.dtype
    input = input > 0
    connected_components, num_components = ndimage.label(input)
    component_sizes = np.bincount(connected_components.ravel())
    component_sizes = component_sizes[1:]
    if len(component_sizes > 0):
        largest_component_label = np.argmax(component_sizes) + 1
        largest_component_size = component_sizes[largest_component_label - 1]
    else:
        return None
    input = input == largest_component_label
    input = ndimage.binary_fill_holes(input)
    input = input.astype(dtype_org)
    return input


def generateFullMask():
    patientFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/479"
    sourceMaskFile = os.path.join(patientFolder, "Prediction_HeadNeck.nii")
    maskArray = nib.load(sourceMaskFile).get_fdata()
    maskArray = np.transpose(maskArray, axes=(2, 1, 0))
    maskArray = brainMaskProcessing(maskArray)

    numSlices = maskArray.shape[0]
    shape_org = (numSlices, 232, 232)
    maskArray = transform.resize(maskArray, shape_org)
    dtype_org = maskArray.dtype
    maskArray = maskArray > 0
    maskArray = maskArray.astype(dtype_org)

    shape_full = (numSlices, 512, 512)
    fullArray = np.zeros(shape_full, dtype=maskArray.dtype)
    fullArray[:, 130: -150, 140: -140] = maskArray
    fullArray = np.transpose(fullArray, axes=(2, 1, 0))
    fullArray[:, :, 64:] = 0
    target = nib.Nifti1Image(fullArray, affine=np.eye(4))
    targetFile = os.path.join(patientFolder, "HeadNeckFull.nii")
    nib.save(target, targetFile)
    print(targetFile)


def preMaskCreation():
    """
    This function outputs the masks to files
    """
    patientFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/479"
    # Get CT files and relevant data
    dataFolder = os.path.join(patientFolder, "data")
    RTFile = os.path.join(dataFolder, "RTStruct.dcm")
    resolution_homo = 2.5
    CTData = []
    files = os.listdir(dataFolder)
    sliceThickness = None
    for file in files:
        path = os.path.join(dataFolder, file)
        dataset = pydicom.dcmread(path)
        if dataset.Modality == "CT":
            InstanceNumber = int(dataset.InstanceNumber)
            CTData.append((InstanceNumber, dataset.pixel_array, float(dataset.SliceLocation)))
            if sliceThickness is None:
                sliceThickness = dataset.SliceThickness
                PixelSpacing = dataset.PixelSpacing
                Rows = dataset.Rows
                Columns = dataset.Columns
                datatype = dataset.pixel_array.dtype
    CTData.sort(key = lambda a: a[0])
    nSlices = len(CTData)
    sliceThickness = abs(CTData[0][2] - CTData[1][2])
    CT_array = np.zeros((nSlices, Rows, Columns), dtype=datatype)
    for i, entry in enumerate(CTData):
        InstanceNumber, slice, _ = entry
        CT_array[i, :, :] = slice

    # get anatomy masks
    RTStruct = RTStructBuilder.create_from(dicom_series_path=dataFolder, rt_struct_path=RTFile)
    names = RTStruct.get_roi_names()
    exclude = ["CTV56", "CTV63", "CTV70", "GTV", "BRAIN_STEM", "SPINAL_CORD"]
    names = [a for a in names if a not in exclude]
    masks = {}
    for name in names:
        mk = RTStruct.get_roi_mask_by_name(name)
        mk = np.transpose(mk, (2, 0, 1))
        mk = np.flip(mk, axis=0)
        masks[name] = mk
        print(name)
    
    # get another structure, brain
    brainMaskFile = os.path.join(patientFolder, "HeadNeckFull.nii")
    brainMask = nib.load(brainMaskFile).get_fdata()
    brainMask = brainMask > 0
    brainMask = np.transpose(brainMask, axes=(2, 1, 0))
    masks["brain"] = brainMask
    names.append("brain")

    # process the masks so that the structures are mutually exclusive
    orders = ["PTV70", "PTV56"]
    # masks_ptv = [(a, masks[a]) for a in orders]
    masks_ptv = []
    for name in orders:
        mask_org = masks[name]
        # mask_org[74:, :, :] = 0
        masks_ptv.append((name, mask_org))
    mask_skin = masks["SKIN"]

    if True:
        # extend mask_skin
        template_slice = mask_skin[37, :, :]
        template_slice = np.expand_dims(template_slice, axis=0)
        mask_skin[:37, :, :] = template_slice

    set_aside = orders + ["SKIN"]
    masks_oar = [(a, masks[a]) for a in names if a not in set_aside]

    masks_ptv_updated = []
    union = np.zeros_like(mask_skin)
    for name, mask in masks_ptv:
        mask_new = np.logical_and(mask, np.logical_not(union))
        union = np.logical_or(mask, union)
        masks_ptv_updated.append((name, mask_new))
    masks_oar_updated = []
    for name, mask in masks_oar:
        mask = np.logical_and(mask, np.logical_not(union))
        masks_oar_updated.append((name, mask))
    masks = masks_ptv_updated + masks_oar_updated + [("SKIN", mask_skin)]

    # resize to the new resolution
    dim_org = np.array((nSlices, Rows, Columns))
    res_org = np.array((float(sliceThickness), PixelSpacing[0], PixelSpacing[1]))
    shape_org = dim_org * res_org
    dim_new = shape_org / resolution_homo
    dim_new = dim_new.astype(int)
    print(dim_new)
    CT_array = transform.resize(CT_array, dim_new)
    CT_array = CT_array * 65535
    CT_array = CT_array.astype(np.uint16)
    masks = {a: transform.resize(b, dim_new).astype(np.uint8) for a, b in masks}

    # generate mask
    densityFile = os.path.join(patientFolder, "density_raw.bin")
    CT_array_write = np.flip(CT_array, axis=0)
    CT_array_write.tofile(densityFile)
    for name, array in masks.items():
        array_write = np.flip(array, axis=0)
        file = os.path.join(patientFolder, "InputMask", "{}.bin".format(name))
        if not os.path.isdir(os.path.dirname(file)):
            os.mkdir(os.path.dirname(file))
        array_write.tofile(file)
        print(file)

    if True:
        colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
        outputFolder = os.path.join(patientFolder, "inputMaskView")
        if not os.path.isdir(outputFolder):
            os.mkdir(outputFolder)
        for i in range(nSlices):
            slice = CT_array[i, :, :]
            plt.imshow(slice, cmap="gray")
            for j, entry in enumerate(masks.items()):
                color = colors[j]
                name, mask = entry
                maskSlice = mask[i, :, :]
                contours = measure.find_contours(maskSlice)
                initial = True
                for contour in contours:
                    if initial:
                        plt.plot(contour[:, 1], contour[:, 0], color=color, label=name, linestyle="--")
                        initial = False
                    else:
                        plt.plot(contour[:, 1], contour[:, 0], color=color, linestyle="--")
            plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
            file = os.path.join(outputFolder, "{:03d}.png".format(i))
            plt.savefig(file)
            plt.clf()
            print(file)


def PTVMerge():
    patientFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/479"
    maskFolder = os.path.join(patientFolder, "InputMask")
    PTV1File = os.path.join(maskFolder, "PTV70.bin")
    PTV2File = os.path.join(maskFolder, "PTV63.bin")
    PTV3File = os.path.join(maskFolder, "PTV56.bin")
    PTV1 = np.fromfile(PTV1File, dtype=np.uint8)
    PTV2 = np.fromfile(PTV2File, dtype=np.uint8)
    PTV3 = np.fromfile(PTV3File, dtype=np.uint8)
    PTV_merge = np.logical_or(PTV1, PTV2)
    PTV_merge = np.logical_or(PTV_merge, PTV3)
    PTV_merge_file = os.path.join(maskFolder, "PTVMerge.bin")
    PTV_merge.tofile(PTV_merge_file)


def structuresFileGen():
    """
    This function generates the structures file
    """
    patientFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/479"
    maskFolder = os.path.join(patientFolder, "InputMask")
    structures = [a.split(".")[0] for a in os.listdir(maskFolder)]
    BODY = "SKIN"
    PTV = "PTVMerge"
    structures.remove(BODY)
    structures.remove(PTV)
    structures.insert(0, BODY)
    
    content = {
        "prescription": 70,
        "ptv": PTV,
        "oar": structures
    }
    content = json.dumps(content, indent=4)
    file = os.path.join(patientFolder, "structures.json")
    with open(file, "w") as f:
        f.write(content)


def StructInfoGen():
    """
    This function generates the StructureInfo.csv file for a new plan
    """
    if True:
        # Specify the variables below
        expFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/479/FastDose"
        dimensionFile = os.path.join(expFolder, "prep_output", "dimension.txt")

    with open(dimensionFile, "r") as f:
        lines = f.readlines()
    organs = lines[2]
    organs = organs.split(" ")
    organs.remove("")

    PTVs = [("PTV70", 70), ("PTV63", 63), ("PTV56", 56)]
    irrelevant = ["SKIN", "PTVMerge", "RingStructure"]
    PTV_names = [a[0] for a in PTVs]
    exclude = irrelevant + PTV_names
    OARs = [a for a in organs if a not in exclude]
    print(OARs)

    # prepare the StructureInfo content
    content = "Name,maxWeights,maxDose,minDoseTargetWeights,minDoseTarget,OARWeights,IdealDose"
    for name, dose in PTVs:
        newline = "{},100,{},100,{},NaN,{}".format(name, dose, dose, dose)
        content = content + "\n" + newline
    for name in OARs:
        newline = newline = "{},0,18,NaN,NaN,5,0".format(name)
        content = content + "\n" + newline
    print(content)
    
    outputFile = os.path.join(expFolder, "StructureInfo.csv")
    with open(outputFile, 'w') as f:
        f.write(content)


def doseMatMerge():
    """
    This function merges different matrices into one
    """
    NonZeroElements_collection = []
    numRowsPerMat_collection = []
    offsetsBuffer_collection = []
    columnsBuffer_collection = []
    valuesBuffer_collection = []
    fluenceMap_collection = []

    expFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/479/FastDose"
    numMatrices = 4
    for i in range(1, numMatrices+1):
        doseMatFolder = os.path.join(expFolder, "doseMat{}/doseMatFolder".format(i))
        NonZeroElementsFile = os.path.join(doseMatFolder, "NonZeroElements.bin")
        numRowsPerMatFile = os.path.join(doseMatFolder, "numRowsPerMat.bin")
        offsetsBufferFile = os.path.join(doseMatFolder, "offsetsBuffer.bin")
        columnsBufferFile = os.path.join(doseMatFolder, "columnsBuffer.bin")
        valuesBufferFile = os.path.join(doseMatFolder, "valuesBuffer.bin")
        fluenceMapFile = os.path.join(doseMatFolder, "fluenceMap.bin")

        NonZeroElements = np.fromfile(NonZeroElementsFile, dtype=np.uint64)
        numRowsPerMat = np.fromfile(numRowsPerMatFile, dtype=np.uint64)
        offsetsBuffer = np.fromfile(offsetsBufferFile, dtype=np.uint64)
        columnsBuffer = np.fromfile(columnsBufferFile, dtype=np.uint64)
        valuesBuffer = np.fromfile(valuesBufferFile, dtype=np.float32)
        fluenceMap = np.fromfile(fluenceMapFile, dtype=np.uint8)

        NonZeroElements_collection.append(NonZeroElements)
        numRowsPerMat_collection.append(numRowsPerMat)
        offsetsBuffer_collection.append(offsetsBuffer)
        columnsBuffer_collection.append(columnsBuffer)
        valuesBuffer_collection.append(valuesBuffer)
        fluenceMap_collection.append(fluenceMap)
        print(doseMatFolder)
    
    NonZeroElements = np.concatenate(NonZeroElements_collection)
    numRowsPerMat = np.concatenate(numRowsPerMat_collection, axis=0)
    offsetsBuffer = np.concatenate(offsetsBuffer_collection, axis=0)
    columnsBuffer = np.concatenate(columnsBuffer_collection, axis=0)
    valuesBuffer = np.concatenate(valuesBuffer_collection, axis=0)
    fluenceMap = np.concatenate(fluenceMap_collection, axis=0)
    print("concatenation")

    targetFolder = os.path.join(expFolder, "doseMatMerge/doseMatFolder")
    if not os.path.isdir(targetFolder):
        os.makedirs(targetFolder)
    NonZeroElements.tofile(os.path.join(targetFolder, "NonZeroElements.bin"))
    numRowsPerMat.tofile(os.path.join(targetFolder, "numRowsPerMat.bin"))
    offsetsBuffer.tofile(os.path.join(targetFolder, "offsetsBuffer.bin"))
    columnsBuffer.tofile(os.path.join(targetFolder, "columnsBuffer.bin"))
    valuesBuffer.tofile(os.path.join(targetFolder, "valuesBuffer.bin"))
    fluenceMap.tofile(os.path.join(targetFolder, "fluenceMap.bin"))
    print(targetFolder)


def drawDose_opt():
    """
    This function draws the dose map generated by the optimization algorithm
    """
    patientFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/479"
    expFolder = os.path.join(patientFolder, "FastDose")
    planFolder = os.path.join(expFolder, "plan1")
    doseFile = os.path.join(planFolder, "dose.bin")
    doseShape = (176, 200, 200)
    doseArray = np.fromfile(doseFile, dtype=np.float32)
    doseArray = np.reshape(doseArray, doseShape)
    # doseArray = np.flip(doseArray, axis=0)
    
    densityFile = os.path.join(patientFolder, "density_raw.bin")
    densityArray = np.fromfile(densityFile, dtype=np.uint16)
    densityArray = np.reshape(densityArray, doseShape)

    maskFolder = os.path.join(patientFolder, "InputMask")
    files = os.listdir(maskFolder)
    masks = []
    exclude = ["PTVMerge"]
    for file in files:
        name = file.split(".")[0]
        if name in exclude:
            continue
        path = os.path.join(maskFolder, file)
        maskArray = np.fromfile(path, dtype=np.uint8)
        maskArray = np.reshape(maskArray, doseShape)
        masks.append((name, maskArray))
    
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
    if True:
        nSlices = densityArray.shape[0]
        doseViewFolder = os.path.join(planFolder, "doseView")
        if not os.path.isdir(doseViewFolder):
            os.mkdir(doseViewFolder)
        maxDose = 80
        for i in range(nSlices):
            densitySlice = densityArray[i, :, :]
            doseSlice = doseArray[i, :, :]
            fig, ax = plt.subplots(figsize=(8, 5))
            plt.imshow(densitySlice, cmap="gray")
            plt.imshow(doseSlice, cmap="jet", vmin=0, vmax=maxDose, alpha=0.3)
            for j, entry in enumerate(masks):
                color = colors[j]
                name, maskArray = entry
                maskSlice = maskArray[i, :, :]
                contours = measure.find_contours(maskSlice)
                initial = True
                for contour in contours:
                    if initial:
                        plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                        initial = False
                    else:
                        plt.plot(contour[:, 1], contour[:, 0], color=color)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            file = os.path.join(doseViewFolder, "{:03d}.png".format(i))
            plt.savefig(file)
            plt.clf()
            print(file)


def drawDVH_opt():
    """
    Draw DVH for the optimized plan
    """
    patientFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/479"
    planFolder = os.path.join(patientFolder, "FastDose", "plan1")
    maskFolder = os.path.join(patientFolder, "InputMask")
    doseFile = os.path.join(planFolder, "dose.bin")
    refDoseFile = os.path.join(patientFolder, "doseExp1.npy")
    doseShape = (176, 200, 200)

    doseArray = np.fromfile(doseFile, dtype=np.float32)
    doseArray = np.reshape(doseArray, doseShape)
    masks = []
    files = os.listdir(maskFolder)
    exclude = ["SKIN", "PTVMerge"]
    PrimaryPTV = "PTV70"
    primaryMask = None
    for file in files:
        name = file.split(".")[0]
        if name in exclude:
            continue
        path = os.path.join(maskFolder, file)
        maskArray = np.fromfile(path, dtype=np.uint8)
        maskArray = np.reshape(maskArray, doseShape)
        masks.append((name, maskArray))
        if name == PrimaryPTV:
            primaryMask = maskArray
    
    primaryMask = primaryMask > 0
    primaryDose = doseArray[primaryMask]
    thresh = np.percentile(primaryDose, 5)
    doseArray = doseArray / thresh * 70

    doseRef = np.load(refDoseFile)
    doseRef = np.transpose(doseRef, axes=(2, 0, 1))
    doseRef = np.flip(doseRef, axis=0)
    doseRef = transform.resize(doseRef, doseShape)
    primaryDoseRef = doseRef[primaryMask]
    threshRef = np.percentile(primaryDoseRef, 5)
    doseRef = doseRef / threshRef * 70
    
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, entry in enumerate(masks):
        color = colors[i]
        name, mask = entry
        mask = mask > 0
        struct_dose = doseArray[mask]
        struct_dose = np.sort(struct_dose)
        struct_dose = np.insert(struct_dose, 0, 0.0)
        numPoints = struct_dose.size
        percentile = np.linspace(100, 0, numPoints)
        ax.plot(struct_dose, percentile, color=color, label=name)

        struct_ref_dose = doseRef[mask]
        struct_ref_dose = np.sort(struct_ref_dose)
        struct_ref_dose = np.insert(struct_ref_dose, 0, 0)
        ax.plot(struct_ref_dose, percentile, color=color, linestyle="--")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    DVH_opt_file = os.path.join(patientFolder, "DVH_opt.png")
    plt.savefig(DVH_opt_file)
    plt.clf()


if __name__ == "__main__":
    # examineDose_new()
    # dcm2nii()
    # examine_nifti()
    # examineMask()
    # generateFullMask()
    # preMaskCreation()
    # PTVMerge()
    # structuresFileGen()
    # StructInfoGen()
    # doseMatMerge()
    # drawDose_opt()
    drawDVH_opt()