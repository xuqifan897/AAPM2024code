import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib
from skimage import transform, measure
import json
import h5py

rootFolder = "/data/qifan/projects/FastDoseWorkplace/VMAT_forLu"
patientName = "HN_002"
groupName = "our_model"
voxelSizeTarget = np.array((2.5, 2.5, 2.5))  # mm
colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
BODYname = "BODY"

def dataView():
    patientFolder = os.path.join(rootFolder, patientName)
    CTFolder = os.path.join(patientFolder, "CT")

    CTFile = os.listdir(CTFolder)
    CTFile = os.path.join(CTFolder, CTFile[0])
    CT_img = nib.load(CTFile)
    CT_header = CT_img.header
    voxelSize = CT_header.get_zooms()
    CT_array = CT_img.get_fdata()
    CT_array = CT_array + 1000
    CT_array[CT_array<0] = 0
    CT_array = np.transpose(CT_array, axes=(2, 1, 0))

    # get the new dimension
    voxelSize = np.flip(voxelSize)
    shape_org = np.array(CT_array.shape)
    size_org  = voxelSize * shape_org
    shape_new = size_org / voxelSizeTarget
    shape_new = shape_new.astype(int)
    CT_array = transform.resize(CT_array, shape_new)
    CT_array = CT_array.astype(np.uint16)

    maskFolder = os.path.join(patientFolder, groupName)
    fullFolder = os.path.join(patientFolder, maskFolder)
    pattern = os.path.join(fullFolder, "*.nii.gz")
    maskFiles = glob.glob(pattern)
    structureNames = []
    for maskFile in maskFiles:
        filename = os.path.basename(maskFile)
        structureName = filename.split(".")[0]
        structureNames.append(structureName)
    Masks = {}
    for name, file in zip(structureNames, maskFiles):
        maskArray = nib.load(file).get_fdata()
        maskArray = np.transpose(maskArray, axes=(2, 1, 0))
        maskArray = transform.resize(maskArray, shape_new)
        maskArray = maskArray > 0
        maskArray = maskArray.astype(np.uint8)
        Masks[name] = maskArray

    # PTV processing
    PTVnames = []
    for name in Masks.keys():
        if "PTV" in name:
            PTVnames.append(name)
    PTVnames.sort(reverse=True)
    Others = [name for name in Masks.keys() if name not in PTVnames]
    MasksProcessed = {}
    baseMask = np.zeros_like(maskArray)
    for PTVname in PTVnames:
        local_mask = Masks[PTVname]
        local_mask_processed = np.logical_and(local_mask, np.logical_not(baseMask))
        baseMask = np.logical_or(baseMask, local_mask)
        MasksProcessed[PTVname] = local_mask_processed
    MasksProcessed["PTVMerge"] = baseMask
    for other in Others:
        if other == BODYname:
            MasksProcessed[other] = Masks[other]
            continue
        mask = Masks[other]
        mask = np.logical_and(mask, np.logical_not(baseMask))
        MasksProcessed[other] = mask
    
    imageFolder = os.path.join(patientFolder, groupName + "_view")
    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)
    for j in range(shape_new[0]):
        CT_slice = CT_array[j, :, :]
        plt.imshow(CT_slice, cmap="gray")
        for idx, entry in enumerate(MasksProcessed.items()):
            name, array = entry
            color = colors[idx]
            maskSlice = array[j, :, :]
            if np.max(maskSlice) == 0:
                continue
            contours = measure.find_contours(maskSlice)
            initial = True
            for contour in contours:
                if initial:
                    plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                    initial = False
                else:
                    plt.plot(contour[:, 1], contour[:, 0], color=color)
        figureFile = os.path.join(imageFolder, "{:03d}.png".format(j))
        plt.legend()
        plt.savefig(figureFile)
        plt.clf()
        print(figureFile)


def writeArray():
    """This function writes the nii.gz files"""
    patientFolder = os.path.join(rootFolder, patientName)
    CTFolder = os.path.join(patientFolder, "CT")

    CTFile = os.listdir(CTFolder)
    CTFile = os.path.join(CTFolder, CTFile[0])
    CT_img = nib.load(CTFile)
    CT_header = CT_img.header
    voxelSize = CT_header.get_zooms()
    CT_array = CT_img.get_fdata()
    CT_array = CT_array + 1000
    CT_array[CT_array<0] = 0
    CT_array = np.transpose(CT_array, axes=(2, 1, 0))

    # get the new dimension
    voxelSize = np.flip(voxelSize)
    shape_org = np.array(CT_array.shape)
    size_org  = voxelSize * shape_org
    shape_new = size_org / voxelSizeTarget
    shape_new = shape_new.astype(int)
    print(shape_new)
    CT_array = transform.resize(CT_array, shape_new)
    CT_array = CT_array.astype(np.uint16)
    CT_file = os.path.join(patientFolder, "planFolder", "density_raw.bin")
    CT_array.tofile(CT_file)

    maskFolder = os.path.join(patientFolder, groupName)
    fullFolder = os.path.join(patientFolder, maskFolder)
    pattern = os.path.join(fullFolder, "*.nii.gz")
    maskFiles = glob.glob(pattern)
    structureNames = []
    for maskFile in maskFiles:
        filename = os.path.basename(maskFile)
        structureName = filename.split(".")[0]
        structureNames.append(structureName)
    Masks = {}
    for name, file in zip(structureNames, maskFiles):
        maskArray = nib.load(file).get_fdata()
        maskArray = np.transpose(maskArray, axes=(2, 1, 0))
        maskArray = transform.resize(maskArray, shape_new)
        maskArray = maskArray > 0
        maskArray = maskArray.astype(np.uint8)
        Masks[name] = maskArray

    # PTV processing
    PTVnames = []
    for name in Masks.keys():
        if "PTV" in name:
            PTVnames.append(name)
    PTVnames.sort(reverse=True)
    Others = [name for name in Masks.keys() if name not in PTVnames]
    MasksProcessed = {}
    baseMask = np.zeros_like(maskArray)
    for PTVname in PTVnames:
        local_mask = Masks[PTVname]
        local_mask_processed = np.logical_and(local_mask, np.logical_not(baseMask))
        baseMask = np.logical_or(baseMask, local_mask)
        MasksProcessed[PTVname] = local_mask_processed
    MasksProcessed["PTVMerge"] = baseMask
    for other in Others:
        if other == BODYname:
            MasksProcessed[other] = Masks[other]
            continue
        mask = Masks[other]
        mask = np.logical_and(mask, np.logical_not(baseMask))
        MasksProcessed[other] = mask

    InputMaskFolder = os.path.join(patientFolder, "planFolder", "InputMask_" + groupName)
    if not os.path.isdir(InputMaskFolder):
        os.makedirs(InputMaskFolder)
    for name, mask in MasksProcessed.items():
        fileName = os.path.join(InputMaskFolder, "{}.bin".format(name))
        mask.tofile(fileName)
        print(fileName)


def structureGen():
    """
    This file generates the structures.json file
    """
    patientFolder = os.path.join(rootFolder, patientName)
    CTFolder = os.path.join(patientFolder, "CT")
    maskFolder = os.path.join(patientFolder, groupName)
    pattern = os.path.join(patientFolder, "planFolder", "InputMask_" + groupName, "*.bin")
    maskFiles = glob.glob(pattern)
    maskFiles = [os.path.basename(a) for a in maskFiles]
    maskNames = [a.split(".")[0] for a in maskFiles]

    PTVName = "PTVMerge"
    BODYName = "BODY"
    maskNames.remove(PTVName)
    maskNames.remove(BODYName)
    maskNames.insert(0, BODYName)
    content = {
        "prescription": 70,
        "ptv": PTVName,
        "oar": maskNames
    }
    content = json.dumps(content, indent=4)
    targetFile = os.path.join(patientFolder, "planFolder", "structures_" + groupName + ".json")
    with open(targetFile, "w") as f:
        f.write(content)


def StructureInfoGen():
    expFolder = os.path.join(rootFolder, patientName, "planFolder")
    inputFolder = os.path.join(expFolder, "prep_output_" + groupName)
    dimensionFile = os.path.join(inputFolder, "dimension.txt")
    with open(dimensionFile, "r") as f:
        lines = f.readlines()
    organs = lines[2]
    organs = organs.split(" ")
    organs.remove("")

    if patientName == "HGJ_001":
        PTVs = [("PTV70", 70), ("PTV60", 60)]
    elif patientName == "HN_002":
        PTVs = [("PTV70", 70), ("PTV60", 60), ("PTV54", 54)]
    irrelevant = ["BODY", "PTVMerge"]

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
    
    outputFile = os.path.join(expFolder, "StructureInfo_" + groupName + ".csv")
    with open(outputFile, 'w') as f:
        f.write(content)


def doseMatMerge():
    NonZeroElements_collection = []
    numRowsPerMat_collection = []
    offsetsBuffer_collection = []
    columnsBuffer_collection = []
    valuesBuffer_collection = []
    fluenceMap_collection = []
    planFolder = os.path.join(rootFolder, patientName, "planFolder")
    numMatrices = 4
    for i in range(1, numMatrices+1):
        doseMatFolder = os.path.join(planFolder, "{}_doseMat{}/doseMatFolder"
                                    .format(groupName, i))
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

    targetFolder = os.path.join(planFolder, "{}_doseMatMerge/doseMatFolder".format(groupName))
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
    planFolder = os.path.join(rootFolder, patientName, "planFolder")
    optResultFolder = os.path.join(planFolder, "plan1_{}".format(groupName))
    prep_output_folder = os.path.join(planFolder, "prep_output_{}".format(groupName))
    dimensionFile = os.path.join(prep_output_folder, "dimension.txt")
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

    exclude = ["BODY", "RingStructure", "PTVMerge"]
    PrimaryPTV = "PTV70"
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
    doseArray = doseArray / thresh * 70

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
    plt.xlabel("Dose (Gy)")
    plt.ylabel("Percentage (%)")
    plt.title("DVH {} {}".format(patientName, groupName))
    DVH_opt_file = os.path.join(optResultFolder, "DVH_opt.png")
    plt.savefig(DVH_opt_file)
    plt.clf()
    print(DVH_opt_file)


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


def patient_HGJ_001_DVH_comp():
    planFolder = os.path.join(rootFolder, patientName, "planFolder")
    dimensionFile = os.path.join(planFolder, "prep_output_benchmark/dimension.txt")
    with open(dimensionFile, "r") as f:
        lines = f.readlines()
    doseShape = lines[0]
    doseShape = doseShape.split(" ")
    doseShape = [int(a) for a in doseShape]
    doseShape.reverse()
    doseShape = tuple(doseShape)

    result_benchmark = os.path.join(planFolder, "plan1_benchmark", "dose.bin")
    result_our_model = os.path.join(planFolder, "plan1_our_model", "dose.bin")
    dose_benchmark = np.fromfile(result_benchmark, dtype=np.float32)
    dose_our_model = np.fromfile(result_our_model, dtype=np.float32)
    dose_benchmark = np.reshape(dose_benchmark, doseShape)
    dose_our_model = np.reshape(dose_our_model, doseShape)

    roi_list_benchmark = os.path.join(planFolder, "prep_output_benchmark", "roi_list.h5")
    roi_list_our_model = os.path.join(planFolder, "prep_output_our_model", "roi_list.h5")
    exclude = ["BODY", "RingStructure", "PTVMerge"]
    PrimaryPTV = "PTV70"
    Masks_benchmark = getStructures(roi_list_benchmark)
    Masks_our_model = getStructures(roi_list_our_model)
    Masks_benchmark = {a[0]: a[1] for a in Masks_benchmark if a[0] not in exclude}
    Masks_our_model = {a[0]: a[1] for a in Masks_our_model if a[0] not in exclude}
    for name in Masks_benchmark.keys():
        if name == PrimaryPTV:
            primaryMask_benchmark = Masks_benchmark[name]
            primaryMask_our_model = Masks_our_model[name]
            break

    primaryMask_benchmark = primaryMask_benchmark > 0
    primaryDose_benchmark = dose_benchmark[primaryMask_benchmark]
    thresh_benchmark = np.percentile(primaryDose_benchmark, 5)
    dose_benchmark = dose_benchmark / thresh_benchmark * 70

    primaryMask_our_model = primaryMask_our_model > 0
    primaryDose_our_model = dose_our_model[primaryMask_our_model]
    thresh_our_model = np.percentile(primaryDose_our_model, 5)
    dose_our_model = dose_our_model / thresh_our_model * 70

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(Masks_benchmark.keys()):
        color = colors[i]
        mask_benchmark = Masks_benchmark[name]
        struct_benchmark = dose_benchmark[mask_benchmark]
        struct_benchmark = np.sort(struct_benchmark)
        struct_benchmark = np.insert(struct_benchmark, 0, 0.0)
        numPoints = struct_benchmark.size
        percentile = np.linspace(100, 0, numPoints)
        plt.plot(struct_benchmark, percentile, color=color, linestyle="--")

        mask_our_model = Masks_our_model[name]
        struct_our_model = dose_our_model[mask_our_model]
        struct_our_model = np.sort(struct_our_model)
        struct_our_model = np.insert(struct_our_model, 0, 0.0)
        numPoints = struct_our_model.size
        percentile = np.linspace(100, 0, numPoints)
        plt.plot(struct_our_model, percentile, color=color, label=name, linestyle="-")
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.xlabel("Dose (Gy)")
    plt.ylabel("Percentage (%)")
    plt.title("DVH comparison")
    DVH_file = os.path.join(planFolder, "DVH_comp.png")
    plt.savefig(DVH_file)
    print(DVH_file)


def drawDoseWash():
    patientFolder = os.path.join(rootFolder, patientName)
    planFolder = os.path.join(patientFolder, "planFolder")
    prep_output = os.path.join(planFolder, "prep_output_{}".format(groupName))

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

    doseFile = os.path.join(planFolder, "plan1_{}".format(groupName), "dose.bin")
    doseArray = np.fromfile(doseFile, dtype=np.float32)
    doseArray = np.reshape(doseArray, doseShape)

    structuresFile = os.path.join(prep_output, "roi_list.h5")
    structures = getStructures(structuresFile)
    structures = [a for a in structures if a[0] != "PTVMerge"]

    dose_vmax = 90
    imageFolder = os.path.join(planFolder, "doseWash_{}".format(groupName))
    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)
    nSlices = doseShape[0]
    for i in range(nSlices):
        plt.imshow(density[i, :, :], cmap="gray")
        plt.imshow(doseArray[i, :, :], cmap="jet", vmin=0, vmax=dose_vmax, alpha=0.3)
        for j, entry in enumerate(structures):
            name, array = entry
            color = colors[j]
            array_slice = array[i]
            contours = measure.find_contours(array_slice)
            initial = True
            for contour in contours:
                if initial:
                    plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                    initial = False
                else:
                    plt.plot(contour[:, 1], contour[:, 0], color=color)
        plt.legend()
        figureFile = os.path.join(imageFolder, "{:03d}.png".format(i))
        plt.savefig(figureFile)
        plt.clf()
        print(figureFile)


def summarize():
    sumFolder = os.path.join(rootFolder, "summary")
    if not os.path.isdir(sumFolder):
        os.mkdir(sumFolder)
    inputInfo = [
        ("HGJ_001", "benchmark", "plan1_benchmark"),
        ("HGJ_001", "our_model", "plan2_our_model"),
        ("HN_002", "benchmark", "plan1_benchmark"),
        ("HN_002", "our_model", "plan1_our_model")
    ]
    for patient, group, folder in inputInfo:
        CTFile = os.path.join(rootFolder, patient, "CT", "*")
        CTFile = glob.glob(CTFile)[0]
        CTimg = nib.load(CTFile)
        CTArray = CTimg.get_fdata()
        shape_org = CTArray.shape
        shape_org = np.flip(shape_org)

        voxelSize = CTimg.header.get_zooms()
        voxelSize = np.flip(voxelSize)
        shape_new = shape_org * voxelSize / voxelSizeTarget
        shape_new = shape_new.astype(int)

        doseFile = os.path.join(rootFolder, patient, "planFolder", folder, "dose.bin")
        doseArray = np.fromfile(doseFile, dtype=np.float32)
        doseArray = np.reshape(doseArray, shape_new)

        doseArray = transform.resize(doseArray, shape_org)
        # transpose back
        doseArray = np.transpose(doseArray, axes=(2, 1, 0))
        doseImg = nib.Nifti1Image(doseArray, CTimg.affine, header=CTimg.header)
        doseFile = os.path.join(sumFolder, "dose_{}_{}.nii.gz".format(patient, group))
        nib.save(doseImg, doseFile)
        print(doseFile)


if __name__ == "__main__":
    # dataView()
    # writeArray()
    # structureGen()
    # StructureInfoGen()
    # doseMatMerge()
    # drawDose_opt()
    # drawDoseWash()
    # patient_HGJ_001_DVH_comp()
    summarize()