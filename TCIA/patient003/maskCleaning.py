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

def dcm2png():
    PatFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003"
    dcmFolder = os.path.join(PatFolder, "data")
    viewFolder = os.path.join(PatFolder, "view")
    if not os.path.isdir(viewFolder):
        os.mkdir(viewFolder)

    CTData = []
    SourceFiles = os.listdir(dcmFolder)
    for file in SourceFiles:
        path = os.path.join(dcmFolder, file)
        dataset = pydicom.dcmread(path)
        if dataset.Modality == "CT":
            InstanceNumber = int(dataset.InstanceNumber)
            CTData.append((InstanceNumber, dataset.pixel_array))
    CTData.sort(key=lambda a: a[0])
    
    for number, image in CTData:
        file = os.path.join(viewFolder, "{:03d}.png".format(number))
        plt.imsave(file, image, cmap="gray")
        print(file)


def viewExample():
    """
    This function views the example image provided by the repository
    """
    folder = "/data/qifan/projects/AAPM2024/HeadCTSegmentation/image_data_predict_sample"
    sourceFile = os.path.join(folder, "1.nii.gz")
    img = nib.load(sourceFile)
    image = img.get_fdata()
    
    image = np.transpose(image, axes=(2, 1, 0))
    numSlices = image.shape[0]
    image += 1024

    imageFolder = os.path.join(folder, "image")
    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)
    for i in range(numSlices):
        slice = image[i, :, :]
        file = os.path.join(imageFolder, "{:03d}.png".format(i))
        plt.imsave(file, slice, cmap="gray")
        print(file)


def dcm2nii():
    """
    This function converts the dicom format into nii
    """
    dcmFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/data"
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
    DataArray = DataArray[:, 140: -160, 150: -150]
    newShape = (numSlices, 512, 512)
    DataArray = transform.resize(DataArray, newShape)

    DataArray = np.transpose(DataArray, axes=(2, 1, 0))
    nifti_img = nib.Nifti1Image(DataArray, affine=np.eye(4))
    targetFile = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/HeadNeck.nii"
    nib.save(nifti_img, targetFile)
    print(nifti_img)


def examine_nifti():
    sourceFile = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/HeadNeck.nii"
    image = nib.load(sourceFile).get_fdata()
    image = np.transpose(image, axes=(2, 1, 0))
    targetFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/niiView"
    if not os.path.isdir(targetFolder):
        os.mkdir(targetFolder)
    numSlice = image.shape[0]
    for i in range(numSlice):
        slice = image[i, :, :]
        file = os.path.join(targetFolder, "{:03d}.png".format(i))
        plt.imsave(file, slice, cmap="gray")
        print(file)


def examineMask():
    maskFile = "/data/qifan/projects/AAPM2024/HeadCTSegmentation/" \
        "results_folder/04_08_18_12_33_MODEL_PREDICT/Prediction_HeadNeck.nii"
    maskArray = nib.load(maskFile).get_fdata()
    maskArray = np.transpose(maskArray, axes=(2, 1, 0))
    maskArray = brainMaskProcessing(maskArray)

    imageFile = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/HeadNeck.nii"
    imageArray = nib.load(imageFile).get_fdata()
    imageArray = np.transpose(imageArray, axes=(2, 1, 0))

    folder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/examineMask"
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


def generateFullMask():
    sourceMaskFile = "/data/qifan/projects/AAPM2024/HeadCTSegmentation/" \
        "results_folder/04_08_18_12_33_MODEL_PREDICT/Prediction_HeadNeck.nii"
    maskArray = nib.load(sourceMaskFile).get_fdata()
    maskArray = np.transpose(maskArray, axes=(2, 1, 0))
    maskArray = brainMaskProcessing(maskArray)
    numSlices = maskArray.shape[0]
    shape_org = (numSlices, 212, 212)
    maskArray = transform.resize(maskArray, shape_org)
    dtype_org = maskArray.dtype
    maskArray = maskArray > 0
    maskArray = maskArray.astype(dtype_org)

    shape_full = (numSlices, 512, 512)
    fullArray = np.zeros(shape_full, dtype=maskArray.dtype)
    fullArray[:, 140: -160, 150: -150] = maskArray
    fullArray = np.transpose(fullArray, axes=(2, 1, 0))
    fullArray[:, :, 50:] = 0
    target = nib.Nifti1Image(fullArray, affine=np.eye(4))
    targetFile = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/HeadNeckFull.nii"
    nib.save(target, targetFile)
    print(targetFile)


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


def preMaskCreation():
    """
    This function outputs the masks to files
    """
    # Get CT files and relevant data
    dataFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/data"
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
            CTData.append((InstanceNumber, dataset.pixel_array))
            if sliceThickness is None:
                sliceThickness = dataset.SliceThickness
                PixelSpacing = dataset.PixelSpacing
                Rows = dataset.Rows
                Columns = dataset.Columns
                datatype = dataset.pixel_array.dtype
    CTData.sort(key = lambda a: a[0])
    nSlices = len(CTData)
    CT_array = np.zeros((nSlices, Rows, Columns), dtype=datatype)
    for i, entry in enumerate(CTData):
        InstanceNumber, slice = entry
        CT_array[i, :, :] = slice

    # get anatomy masks
    RTStruct = RTStructBuilder.create_from(dicom_series_path=dataFolder, rt_struct_path=RTFile)
    names = RTStruct.get_roi_names()
    exclude = ["CTV56", "GTV", "ptv54combo", "transvol70"]
    names = [a for a in names if a not in exclude]
    masks = {}
    for name in names:
        mk = RTStruct.get_roi_mask_by_name(name)
        mk = np.transpose(mk, (2, 0, 1))
        mk = np.flip(mk, axis=0)
        masks[name] = mk
        print(name)
    
    # get another structure, brain
    brainMaskFile = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/HeadNeckFull.nii"
    brainMask = nib.load(brainMaskFile).get_fdata()
    brainMask = brainMask > 0
    brainMask = np.transpose(brainMask, axes=(2, 1, 0))
    masks["brain"] = brainMask
    names.append("brain")

    # process the masks so that the structures are mutually exclusive
    orders = ["PTV70", "PTV56", "leftptv56"]
    masks_ptv = [(a, masks[a]) for a in orders]
    mask_skin = masks["SKIN"]
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
    CT_array = transform.resize(CT_array, dim_new)
    CT_array = CT_array * 65535
    CT_array = CT_array.astype(np.uint16)
    masks = {a: transform.resize(b, dim_new).astype(np.uint8) for a, b in masks}

    # generate mask
    patientFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003"
    densityFile = os.path.join(patientFolder, "density_raw.bin")
    CT_array_write = np.flip(CT_array, axis=0)
    CT_array_write.tofile(densityFile)
    for name, array in masks.items():
        array_write = np.flip(array, axis=0)
        file = os.path.join(patientFolder, "InputMask", "{}.bin".format(name))
        array_write.tofile(file)
        print(file)

    if False:
        colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
        outputFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/inputMaskView"
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


def examineDose():
    doseFile = "/data/qifan/projects/FastDoseWorkplace/TCIA/0522c0003/dose.dcm"
    dataset = pydicom.dcmread(doseFile)
    DoseArray = dataset.pixel_array
    ImagePositionPatientDose = np.array(dataset.ImagePositionPatient)
    PixelSpacingDose = dataset.PixelSpacing
    GridFrameOffsetVector = dataset.GridFrameOffsetVector
    shape_dose = np.array(DoseArray.shape)  # slice, hight, width

    SliceThicknessDose = abs(GridFrameOffsetVector[1] - GridFrameOffsetVector[0])
    res_dose = np.array((SliceThicknessDose, PixelSpacingDose[0], PixelSpacingDose[1]))

    if False:
        doseFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/doseView"
        if not os.path.isdir(doseFolder):
            os.mkdir(doseFolder)
        maxDose = np.max(DoseArray)
        nSlices = DoseArray.shape[0]
        for i in range(nSlices):
            slice = DoseArray[i, :, :]
            file = os.path.join(doseFolder, "{:03d}.png".format(i))
            plt.imsave(file, slice, cmap="jet", vmin=0, vmax=maxDose)
            print(file)
        return

    CTFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/data"
    CTData = []
    shape = None
    ImagePositionPatientCT = None
    InstanceNumberGlobal = 10000
    SourceFiles = os.listdir(CTFolder)
    for file in SourceFiles:
        path = os.path.join(CTFolder, file)
        dataset = pydicom.dcmread(path)
        if dataset.Modality == "CT":
            InstanceNumber = int(dataset.InstanceNumber)
            CTData.append((InstanceNumber, dataset.pixel_array))
            if InstanceNumber < InstanceNumberGlobal:
                InstanceNumberGlobal = InstanceNumber
                ImagePositionPatientCT = dataset.ImagePositionPatient
            if shape is None:
                shape = CTData[0][1].shape
                SliceThickness = dataset.SliceThickness
                PixelSpacing = dataset.PixelSpacing
    print(ImagePositionPatientCT, ImagePositionPatientDose)
    return
    res_CT = np.array((SliceThickness, PixelSpacing[0], PixelSpacing[1]))
    shape_CT = np.array((len(CTData), shape[0], shape[1]))

    # print(res_CT, shape_CT, res_dose, shape_dose)
    size_dose = res_dose * shape_dose
    shape_dose_new = size_dose / res_CT
    shape_dose_new = shape_dose_new.astype(int)
    DoseArray_resized = transform.resize(DoseArray, shape_dose_new)

    displacement = ImagePositionPatientDose - ImagePositionPatientCT
    displacement_voxel = displacement / res_CT
    displacement_voxel = displacement_voxel.astype(int)
    print(shape_CT, shape_dose_new, displacement_voxel)

    # calculate padding size
    paddingSize = shape_dose_new + displacement_voxel
    paddingSize = np.expand_dims(paddingSize, axis=0)
    paddingSize = np.repeat(paddingSize, 2, axis=0)
    paddingSize[1, :] = shape_CT
    paddingSize = np.max(paddingSize, axis=0)
    DoseArray_embedded = np.zeros(paddingSize, dtype=DoseArray_resized.dtype)
    displacement_end = displacement_voxel + shape_dose_new
    DoseArray_embedded[displacement_voxel[0]: displacement_end[0],
                       displacement_voxel[1]: displacement_end[1],
                       displacement_voxel[2]: displacement_end[2]] = DoseArray_resized
    # truncate unnecessary part
    DoseArray_embedded = DoseArray_embedded[:shape_CT[0], :, :]

    if False:
        doseFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/doseView"
        if not os.path.isdir(doseFolder):
            os.mkdir(doseFolder)
        doseMax = np.max(DoseArray_embedded)
        for i in range(shape_CT[0]):
            slice = DoseArray_embedded[i, :, :]
            file = os.path.join(doseFolder, "{:03d}.png".format(i))
            plt.imsave(file, slice, cmap="jet", vmin=0, vmax=doseMax)
            print(file)
    
    shape_homo = [160, 227, 227]
    DoseArray_homo = transform.resize(DoseArray_embedded, shape_homo)
    doseFileOutput = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/doseRef.npy"
    np.save(doseFileOutput, DoseArray_homo)


def examineDose_new():
    """The interpretation above was not correct"""
    doseFile = "/data/qifan/projects/FastDoseWorkplace/TCIA/0522c0003/dose.dcm"
    CTFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/data"
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
    doseInterpPath = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/doseExp1"
    np.save(doseInterpPath, doseValues)


def viewDoseExp1():
    patientFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003"
    CTFolder = os.path.join(patientFolder, "data")
    doseFile = os.path.join(patientFolder, "doseExp1.npy")
    
    CTData = []
    files = os.listdir(CTFolder)
    shape = None
    for file in files:
        path = os.path.join(CTFolder, file)
        dataset = pydicom.dcmread(path)
        if dataset.Modality != "CT":
            continue
        pixel_array = dataset.pixel_array
        InstanceNumber = int(dataset.InstanceNumber)
        CTData.append((InstanceNumber, pixel_array))
        if shape is None:
            shape = pixel_array.shape
            datatype = pixel_array.dtype
    CTData.sort(key = lambda a: a[0])
    CTShape = (shape[0], shape[1], len(CTData))
    CT_Array = np.zeros(CTShape, dtype=datatype)
    for i in range(CTShape[2]):
        CT_Array[:, :, i] = CTData[i][1]

    CTShape = (len(CTData), shape[0], shape[1])
    MaskFolder = os.path.join(patientFolder, "InputMask")
    maskFiles = os.listdir(MaskFolder)
    maskShape = (160, 227, 227)
    masks = {}
    for file in maskFiles:
        name = file.split(".")[0]
        path = os.path.join(MaskFolder, file)
        array = np.fromfile(path, dtype=np.uint8)
        array = np.reshape(array, maskShape)
        array = np.flip(array, axis=0)
        array = transform.resize(array, CTShape)
        masks[name] = array

    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
    Dose_Array = np.load(doseFile)
    outputFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASelect/003/doseExp1View"
    if not os.path.isdir(outputFolder):
        os.mkdir(outputFolder)
    maxDose = np.max(Dose_Array)
    for i in range(CTShape[2]):
        CT_slice = CT_Array[:, :, i]
        dose_slice = Dose_Array[:, :, i]
        plt.imshow(CT_slice, cmap="gray")
        plt.imshow(dose_slice, cmap="jet", vmin=0, vmax=maxDose, alpha=0.3)
        for j, entry in enumerate(masks.items()):
            color = colors[j]
            name, array = entry
            mask_slice = array[i, :, :]
            contours = measure.find_contours(mask_slice)
            initial = True
            for contour in contours:
                if initial:
                    plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                    initial = False
                else:
                    plt.plot(contour[:, 1], contour[:, 0], color=color)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        imageFile = os.path.join(outputFolder, "{:03d}.png".format(i))
        plt.savefig(imageFile)
        plt.clf()
        print(imageFile)


if __name__ == '__main__':
    # dcm2png()
    # viewExample()
    # dcm2nii()
    # examine_nifti()
    # examineMask()
    # generateFullMask()
    # preMaskCreation()
    # examineDose()
    # examineDose_new()
    viewDoseExp1()