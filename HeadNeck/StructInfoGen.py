import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from skimage import measure, transform
import pydicom
from rt_utils import RTStructBuilder

def StructInfoGen():
    """
    This function generates the StructureInfo.csv file for a new plan
    """
    if True:
        # Specify the variables below
        globalFolder = "/data/qifan/projects/FastDoseWorkplace/CORTTune/HeadNeck"
        dimensionFile = os.path.join(globalFolder, "prep_output", "dimension.txt")
        PTV_name = "PTV_crop"
        BODY_name = "External"

    with open(dimensionFile, "r") as f:
        lines = f.readlines()
    organs = lines[2]
    organs = organs.split(" ")
    organs.remove("")
    
    # bring the PTV and BODY to front
    assert PTV_name in organs and BODY_name in organs, "PTV_name and BODY_name provided is not included."
    organs.remove(PTV_name)
    organs.remove(BODY_name)
    organs = [PTV_name, BODY_name] + organs

    # prepare the StructureInfo content
    content = "Name,maxWeights,maxDose,minDoseTargetWeights,minDoseTarget,OARWeights,IdealDose"
    for name in organs:
        if name == PTV_name:
            newline = "{},100,20,100,20,NaN,20".format(name)
        elif name == BODY_name:
            newline = "{},0,18,NaN,NaN,0,0".format(name)
        else:
            newline = "{},0,18,NaN,NaN,5,0".format(name)
        content = content + "\n" + newline
    
    outputFile = os.path.join(globalFolder, "StructureInfo.csv")
    with open(outputFile, 'w') as f:
        f.writelines(content)


def drawDoseWash():
    prep_output = "/data/qifan/projects/FastDoseWorkplace/CORTTune/HeadNeck/prep_output"
    prep_result = "/data/qifan/projects/FastDoseWorkplace/CORTTune/HeadNeck/plan"
    dimension = (180, 193, 193)
    VOI_exclude = ["SKIN"]

    roi_listFile = os.path.join(prep_output, "roi_list.h5")
    densityFile = os.path.join(prep_output, "density.raw")
    doseFile = os.path.join(prep_result, "dose.bin")

    density = np.fromfile(densityFile, dtype=np.float32)
    dose = np.fromfile(doseFile, dtype=np.float32)
    density = np.reshape(density, dimension)
    dose = np.reshape(dose, dimension)
    dose /= np.max(dose)  # normalize

    file = h5py.File(roi_listFile, 'r')
    structures_filtered = list(file.keys())
    for a in VOI_exclude:
        structures_filtered.remove(a)
    structures_filtered.sort()
    print("Structures to show: ", structures_filtered)

    masks = {}
    for struct_name in structures_filtered:
        struct = file[struct_name]
        structProps = struct["ArrayProps"]
        structMask = struct["mask"]

        structSize = structProps.attrs["size"]
        structCropSize = structProps.attrs["crop_size"]
        structCropStart = structProps.attrs["crop_start"]

        structSize = np.flip(structSize, axis=0)
        structCropSize = np.flip(structCropSize, axis=0)
        structCropStart = np.flip(structCropStart, axis=0)
        # print(structSize, structCropSize, structCropStart)

        structMask = np.array(structMask)
        structMask = np.reshape(structMask, structCropSize)
        struct_mask = np.zeros(structSize, dtype=bool)
        struct_mask[structCropStart[0]: structCropStart[0] + structCropSize[0],
            structCropStart[1]: structCropStart[1] + structCropSize[1],
            structCropStart[2]: structCropStart[2] + structCropSize[2]] = structMask
        masks[struct_name] = struct_mask
        print(struct_name, np.sum(struct_mask))
    
    numStructs = len(masks)
    color_values = np.linspace(0, 1, numStructs)
    color_map = plt.get_cmap('viridis')
    colors = [color_map(value) for value in color_values]

    imageFolder = os.path.join(prep_result, "doseWash")
    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)

    for i in range(dimension[0]):
        densitySlice = density[i, :, :]
        plt.imshow(densitySlice, cmap='gray')
        for j in range(numStructs):
            color = colors[j]
            structure_name = structures_filtered[j]
            mask = masks[structure_name]
            mask_slice = mask[i, :, :]
            contours = measure.find_contours(mask_slice, 0.5)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color=color)
        plt.imshow(dose[i, :, :], cmap="viridis", alpha=0.5)
        imageFile = os.path.join(imageFolder, "{:03d}.png".format(i))
        plt.savefig(imageFile)
        plt.clf()
        print(imageFile)


def doseAnalyze():
    prep_output = "/data/qifan/projects/FastDoseWorkplace/CORTTune/HeadNeck/prep_output"
    prep_result = "/data/qifan/projects/FastDoseWorkplace/CORTTune/HeadNeck/plan"
    dimension = (180, 193, 193)
    VOI_exclude = ["SKIN", "RingStructure"]

    roi_listFile = os.path.join(prep_output, "roi_list.h5")
    densityFile = os.path.join(prep_output, "density.raw")
    doseFile = os.path.join(prep_result, "dose.bin")

    density = np.fromfile(densityFile, dtype=np.float32)
    dose = np.fromfile(doseFile, dtype=np.float32)
    density = np.reshape(density, dimension)
    dose = np.reshape(dose, dimension)
    file = h5py.File(roi_listFile, 'r')
    structures_filtered = list(file.keys())
    for a in VOI_exclude:
        structures_filtered.remove(a)
    structures_filtered.sort()
    print("Structures to show: ", structures_filtered)

    numStructs = len(structures_filtered)
    color_values = np.linspace(0, 1, numStructs)
    color_map = plt.get_cmap('viridis')
    colors = [color_map(value) for value in color_values]

    for struct_name, color in zip(structures_filtered, colors):
        struct = file[struct_name]
        structProps = struct["ArrayProps"]
        structMask = struct["mask"]

        structSize = structProps.attrs["size"]
        structCropSize = structProps.attrs["crop_size"]
        structCropStart = structProps.attrs["crop_start"]

        structSize = np.flip(structSize, axis=0)
        structCropSize = np.flip(structCropSize, axis=0)
        structCropStart = np.flip(structCropStart, axis=0)
        print(structSize, structCropSize, structCropStart)

        structMask = np.array(structMask)
        structMask = np.reshape(structMask, structCropSize)
        struct_mask = np.zeros(structSize, dtype=bool)
        struct_mask[structCropStart[0]: structCropStart[0] + structCropSize[0],
            structCropStart[1]: structCropStart[1] + structCropSize[1],
            structCropStart[2]: structCropStart[2] + structCropSize[2]] = structMask
        struct_dose = dose[struct_mask].copy()
        print(struct_dose.shape)
        DrawDVHLine(struct_dose, color)
    
    plt.xlabel("Dose (Gy)")
    plt.ylabel("Relative volume (%)")
    plt.legend(structures_filtered)
    plotFile = os.path.join(prep_result, "DVH.png")
    plt.savefig(plotFile)


def DrawDVHLine(struct_dose, color, linestyle='-'):
    """
    This function draws the DVH curve for one structure
    """
    struct_dose = np.sort(struct_dose)
    struct_dose = np.insert(struct_dose, 0, 0.0)
    num_voxels = struct_dose.size
    percentile = (num_voxels - np.arange(num_voxels)) / num_voxels * 100

    plt.plot(struct_dose, percentile, color=color, linestyle=linestyle)


def verify_roi_list():
    prep_output = "/data/qifan/projects/FastDoseWorkplace/CORTTune/HeadNeck/prep_output"
    prep_result = "/data/qifan/projects/FastDoseWorkplace/CORTTune/HeadNeck/plan"
    roi_listFile = os.path.join(prep_output, "roi_list.h5")
    file = h5py.File(roi_listFile, 'r')
    structures_filtered = list(file.keys())
    numStructs = len(structures_filtered)
    color_values = np.linspace(0, 1, numStructs)
    color_map = plt.get_cmap('viridis')
    colors = [color_map(value) for value in color_values]
    colors = [(a[0], a[1], a[2], 0.5) for a in colors]

    dimension = (180, 193, 193)
    imageFolder = os.path.join(prep_result, "doseWash")
    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)

    masks = {}
    for struct_name in structures_filtered:
        struct = file[struct_name]
        structProps = struct["ArrayProps"]
        structMask = struct["mask"]

        structSize = structProps.attrs["size"]
        structCropSize = structProps.attrs["crop_size"]
        structCropStart = structProps.attrs["crop_start"]

        structSize = np.flip(structSize, axis=0)
        structCropSize = np.flip(structCropSize, axis=0)
        structCropStart = np.flip(structCropStart, axis=0)
        # print(structSize, structCropSize, structCropStart)

        structMask = np.array(structMask)
        structMask = np.reshape(structMask, structCropSize)
        struct_mask = np.zeros(structSize, dtype=bool)
        struct_mask[structCropStart[0]: structCropStart[0] + structCropSize[0],
            structCropStart[1]: structCropStart[1] + structCropSize[1],
            structCropStart[2]: structCropStart[2] + structCropSize[2]] = structMask
        masks[struct_name] = struct_mask
        print(struct_name, np.sum(struct_mask))

    for i in range(dimension[0]):
        for j in range(numStructs):
            color = colors[j]
            structure_name = structures_filtered[j]
            mask = masks[structure_name]
            mask_slice = mask[i, :, :]
            mask_slice = np.expand_dims(mask_slice, axis=2)
            mask_slice = np.concatenate([mask_slice] * 4, axis=2)
            mask_slice = mask_slice.astype(float)
            mask_slice[:, :, 0] *= color[0]
            mask_slice[:, :, 1] *= color[1]
            mask_slice[:, :, 2] *= color[2]
            mask_slice[:, :, 3] *= color[3]
            plt.imshow(mask_slice)
        file = os.path.join(imageFolder, "{:03d}.png".format(i))
        plt.savefig(file)
        plt.clf()
        print(file)


def HN_resize():
    """
    Seems something wrong. Construct the HN phantom again.
    """
    sourceFolder = "/data/qifan/projects/FastDoseWorkplace/CORTclean/HeadNeck"
    rtFile = os.path.join(sourceFolder, "RTstruct_HN.dcm")
    resolutionNew = np.array((2.5, 2.5, 2.5)) # mm
    nSlices = 90

    densityArray = None
    resolutionOrg = None
    for i in range(nSlices):
        dicomFile = os.path.join(sourceFolder, "{:03d}.dcm".format(i))
        dataset = pydicom.dcmread(dicomFile)
        shape = dataset.pixel_array.shape
        if densityArray is None:
            dimension = [nSlices, shape[0], shape[1]]
            densityArray = np.zeros(dimension, dtype=dataset.pixel_array.dtype)
            PixelSpacing = dataset.PixelSpacing
            SliceThickness = dataset.SliceThickness
            resolutionOrg = np.array((float(SliceThickness), PixelSpacing[0], PixelSpacing[1]))
        densityArray[i, :, :] = dataset.pixel_array
    
    # resize the density array
    dimensionOrg = np.array(densityArray.shape)
    size = dimensionOrg * resolutionOrg
    dimensionNew = size / resolutionNew
    dimensionNew = np.floor(dimensionNew).astype(int)
    dtype_org = densityArray.dtype
    densityArray = transform.resize(densityArray, dimensionNew)
    densityArray *= 65536
    densityArray = densityArray.astype(dtype_org)
    print(np.max(densityArray))

    globalFolder = "/data/qifan/projects/FastDoseWorkplace/CORTclean/HeadNeckResize"
    if not os.path.isdir(globalFolder):
        os.mkdir(globalFolder)
    if False:
        # take a look
        imageFolder = os.path.join(globalFolder, "view")
        if not os.path.isdir(imageFolder):
            os.mkdir(imageFolder)
        for i in range(dimensionNew[0]):
            imageFile = os.path.join(imageFolder, "{:03d}.png".format(i))
            plt.imsave(imageFile, densityArray[i, : ,:], cmap="gray")
            print(imageFile)
        return
    
    # Then construct the new dicom files
    dicomFolder = os.path.join(globalFolder, "dicom")
    if not os.path.isdir(dicomFolder):
        os.mkdir(dicomFolder)
    baseDataset = os.listdir(sourceFolder)
    baseDataset.sort()
    baseDataset = baseDataset[0]
    baseDataset = os.path.join(sourceFolder, baseDataset)
    baseDataset = pydicom.dcmread(baseDataset)
    baseDataset.PixelSpacing = [resolutionNew[1], resolutionNew[2]]
    baseDataset.SliceThickness = str(resolutionNew[0])
    baseDataset.Rows = dimensionNew[1]
    baseDataset.Columns = dimensionNew[2]
    BaseImagePositionPatient = baseDataset.ImagePositionPatient

    # anotherIPP = os.path.join(sourceFolder, "003.dcm")
    # anotherDataset = pydicom.dcmread(anotherIPP)
    # anotherImagePositionPatient = anotherDataset.ImagePositionPatient
    for i in range(densityArray.shape[0]):
        slice = densityArray[i, :, :]
        baseDataset.PixelData = slice
        baseDataset.ImagePositionPatient = BaseImagePositionPatient
        baseDataset.ImagePositionPatient[2] = BaseImagePositionPatient[2] - i * resolutionNew[0]
        baseDataset.SliceLocation = str(baseDataset.ImagePositionPatient[2])
        file = os.path.join(dicomFolder, "{:03d}.dcm".format(i))
        baseDataset.save_as(file)
        print(file)


def examineResize():
    numSlices = 180
    globalFolder = "/data/qifan/projects/FastDoseWorkplace/CORTclean/HeadNeckResize"
    dicomFolder = os.path.join(globalFolder, "dicom")
    imageFolder = os.path.join(globalFolder, "view")
    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)
    for i in range(numSlices):
        sourceFile = os.path.join(dicomFolder, "{:03d}.dcm".format(i))
        targetFile = os.path.join(imageFolder, "{:03d}.png".format(i))
        pixel_array = pydicom.dcmread(sourceFile).pixel_array
        plt.imsave(targetFile, pixel_array, cmap="gray")
        print(targetFile)


def examineSKIN():
    externalShape = [67, 160, 160]
    externalPath = "/data/qifan/projects/FastDoseWorkplace/PlanTune/HN_binary/External.bin"
    externalMask = np.fromfile(externalPath, dtype=np.uint8)
    externalMask = np.reshape(externalMask, externalShape)
    imageFolder = "/data/qifan/projects/FastDoseWorkplace/CORTclean/HeadNeckResize/view"
    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)
    for i in range(externalShape[0]):
        slice = externalMask[i, :, :]
        file = os.path.join(imageFolder, "{:03d}.png".format(i))
        plt.imsave(file, slice)
        print(file)


def maskGen():
    dicomFolder = "/data/qifan/projects/FastDoseWorkplace/CORTclean/HeadNeckResize/dicom"
    binaryFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/HN_binary"
    files = os.listdir(binaryFolder)
    names = [a.split(".")[0] for a in files]
    shape_org = [67, 160, 160]
    shape_target = [180, 193, 193]

    RTStruct = RTStructBuilder.create_new(dicom_series_path=dicomFolder)
    for name in names:
        MaskFile = os.path.join(binaryFolder, name + ".bin")
        Mask = np.fromfile(MaskFile, np.uint8)
        Mask = np.reshape(Mask, shape_org)
        Mask = Mask.astype(float)
        Mask = transform.resize(Mask, shape_target)
        Mask = Mask > 0.5
        Mask = Mask.astype(bool)
        Mask = np.transpose(Mask, (2, 1, 0))

        RTStruct.add_roi(name=name, mask=Mask)
        try:
            Mask_test = RTStruct.get_roi_mask_by_name(name)
            print(name)
        except:
            print("Error loading structure {}".format(name))


def view_HN_PTV70():
    binaryFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/HN_binary"
    name = "PTV70"
    MaskFile = os.path.join(binaryFolder, name + ".bin")
    Mask = np.fromfile(MaskFile, np.uint8)
    shape = [67, 160, 160]
    Mask = np.reshape(Mask, shape)

    shape_new = [180, 193, 193]
    Mask = Mask.astype(float)
    Mask = transform.resize(Mask, shape_new)
    Mask = Mask > 0.5
    Mask = Mask.astype(bool)
    Mask = np.transpose(Mask, (2, 1, 0))

    if False:
        imageFolder = "/data/qifan/projects/FastDoseWorkplace/CORTclean/HeadNeckResize/{}view".format(name)
        if not os.path.isdir(imageFolder):
            os.mkdir(imageFolder)
        for i in range(Mask.shape[2]):
            slice = Mask[:, :, i]
            file = os.path.join(imageFolder, "{:03d}.png".format(i))
            plt.imsave(file, slice)
            print(file)
    
    dicomFolder = "/data/qifan/projects/FastDoseWorkplace/CORTclean/HeadNeckResize/dicom"
    RTStruct = RTStructBuilder.create_new(dicom_series_path=dicomFolder)
    RTStruct.add_roi(name=name, mask=Mask)
    try:
        Mask_test = RTStruct.get_roi_mask_by_name(name)
        print(name)
    except:
        print("Error loading structure {}".format(name))


def HNMaskResize():
    """
    This folder resizes the masks of the head and neck phantom and
    generates a new dicom file
    """
    dicomSource = "/data/qifan/projects/FastDoseWorkplace/CORTclean/HeadNeck"
    dicomTarget = "/data/qifan/projects/FastDoseWorkplace/CORTclean/HeadNeckResize/dicom"

    sourceSlices = 90
    dicomSourceExample = os.path.join(dicomSource, "000.dcm")

    targetSlices = 180
    dicomTargetExample = os.path.join(dicomTarget, "000.dcm")
    targetShape = pydicom.dcmread(dicomTargetExample).pixel_array.shape
    targetShape = (targetShape[0], targetShape[1], targetSlices)

    rtSourceFile = os.path.join(dicomSource, "RTstruct_HN.dcm")
    RTSource = RTStructBuilder.create_from(dicom_series_path=dicomSource, rt_struct_path=rtSourceFile)
    names = RTSource.get_roi_names()

    if False:
        mask = RTSource.get_roi_mask_by_name("SKIN")
        imageFolder = "/data/qifan/projects/FastDoseWorkplace/CORTclean/HeadNeckResize/view"
        if not os.path.isdir(imageFolder):
            os.mkdir(imageFolder)
        for i in range(mask.shape[2]):
            path = os.path.join(imageFolder, "{:03d}.png".format(i))
            slice = mask[:, :, i]
            plt.imsave(path, slice)
            print(path)
        return

    masks = {}
    print("reading mask")
    for name in names:
        mask = RTSource.get_roi_mask_by_name(name)
        mask = mask.astype(float)
        mask = transform.resize(mask, targetShape)
        mask = mask > 0.2
        mask = mask.astype(bool)
        print(name, np.max(mask), np.sum(mask))
        masks[name] = mask

    if False:
        # for debug purposes, to check the mask of SKIN
        skin_mask = masks["SKIN"]
        imageFolder = "/data/qifan/projects/FastDoseWorkplace/CORTclean/HeadNeckResize/view"
        if not os.path.isdir(imageFolder):
            os.mkdir(imageFolder)
        for i in range(targetSlices):
            slice = skin_mask[:, :, i]
            file = os.path.join(imageFolder, "{:03d}.png".format(i))
            plt.imsave(file, slice)
            print(file)
        return
    
    dicomError = ["GTV", "PTV70", "SKIN"]
    names_filtered = [a for a in names if a not in dicomError]
    
    RTTarget = RTStructBuilder.create_new(dicom_series_path=dicomTarget)
    print("\nGenerating new RTstruct")
    for name in names_filtered:
        mask = masks[name]
        RTTarget.add_roi(name=name, mask=mask)
        try:
            mask_test = RTTarget.get_roi_mask_by_name(name)
            print("Structure {}, number of voxels: {}".format(name, np.sum(mask_test)))
        except:
            print("Error loading structure: {}".format(name))
    RTTargetFile = os.path.join(dicomTarget, "RTstruct_HN.dcm")
    RTTarget.save(RTTargetFile)


def examineHNMaskResize():
    dicomFolder = "/data/qifan/projects/FastDoseWorkplace/CORTclean/HeadNeckResize/dicom"
    imageFolder = "/data/qifan/projects/FastDoseWorkplace/CORTclean/HeadNeckResize/view"
    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)
    rtFile = os.path.join(dicomFolder, "RTstruct_HN.dcm")
    rtStruct = RTStructBuilder.create_from(dicom_series_path=dicomFolder, rt_struct_path=rtFile)
    names = rtStruct.get_roi_names()
    masks = {}
    for name in names:
        print("reading structure: {}".format(name))
        mask = rtStruct.get_roi_mask_by_name(name)
        masks[name] = mask
    numStructs = len(names)
    color_values = np.linspace(0, 1, numStructs)
    color_map = plt.get_cmap()
    colors = [color_map(a) for a in color_values]

    numSlices = 180
    for i in range(numSlices):
        file = os.path.join(dicomFolder, "{:03d}.dcm".format(i))
        pixel_array = pydicom.dcmread(file).pixel_array
        plt.imshow(pixel_array)
        for j, name in enumerate(names):
            mask = masks[name]
            maskSlice = mask[:, :, i]
            color = colors[j]
            contours = measure.find_contours(maskSlice)
            initial = True
            for contour in contours:
                if initial:
                    plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color=color, label=name)
                    initial = False
                else:
                    plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color=color)
            plt.legend()
        imageFile = os.path.join(imageFolder, "{:03d}.png".format(i))
        plt.savefig(imageFile)
        plt.clf()
        print(imageFile)


if __name__ == "__main__":
    StructInfoGen()
    # drawDoseWash()
    # doseAnalyze()
    # verify_roi_list()
    # HN_resize()
    # examineResize()
    # examineSKIN()
    # maskGen()
    # view_HN_PTV70()
    # HNMaskResize()
    # examineHNMaskResize()