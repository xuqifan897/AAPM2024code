import os
import numpy as np
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator
import h5py
from skimage import measure

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
RootFolder = "/data/qifan/FastDoseWorkplace/TCIAAdd"
PatientName = "190"
planNo = 1

def drawDVH_opt():
    """
    Draw DVH for the optimized plan
    """
    patientFolder = os.path.join(RootFolder, PatientName)
    planFolder = os.path.join(patientFolder, "FastDose", "plan{}".format(planNo))
    maskFile = os.path.join(patientFolder, "FastDose", "prep_output", "roi_list.h5")
    doseFile = os.path.join(planFolder, "dose.bin")
    refDoseFile = os.path.join(patientFolder, "dose.bin")
    dimensionFile = os.path.join(patientFolder, "FastDose", "prep_output", "dimension.txt")

    with open(dimensionFile, "r") as f:
        lines = f.readlines()
    dimension = lines[0]
    dimension = dimension.replace(" ", ", ")
    dimension = eval(dimension)
    dimension = np.flip(dimension)

    voxelSize = lines[1]
    voxelSize = voxelSize.replace(" ", ", ")
    voxelSize = eval(voxelSize)
    voxelSize = np.flip(voxelSize)
    
    organs = lines[2]
    organs = organs.split(" ")
    organs.remove("")

    doseArray = np.fromfile(doseFile, dtype=np.float32)
    doseArray = np.reshape(doseArray, dimension)
    refDoseArray = np.fromfile(refDoseFile, dtype=np.float32)
    refDoseArray = np.reshape(refDoseArray, dimension)
    
    masks = getStructures(maskFile)
    masks = {a: b for a, b in masks}
    structures = list(masks.keys())
    # exclude = ["PTVMerge", "PTVSeg0", "PTVSeg1", "PTVSeg2", "PTVSeg3", "SKIN", "Trachea", "RingStructure"]
    exclude = ["PTVMerge", "PTVSeg0", "PTVSeg1", "PTVSeg2", "PTVSeg3", "SKIN", "RingStructures"]
    if True:
        exclude.append("RingStructure")
    structures = [a for a in structures if a not in exclude]
    masks = {a: b for a, b in masks.items() if a in structures}

    percentile_value = 10
    # if PatientName in ["002" , "003", "009", "125"]:
    #     percentile_value = 10

    # Normalize
    PTVList = [(a, eval(a[3:])) for a in structures if "PTV" in a]
    PTVList.sort(key=lambda a: a[1], reverse=True)
    PrimaryPTVName, PrimaryDose = PTVList[0]
    print(PrimaryDose, PrimaryPTVName)
    PrimaryMask = masks[PrimaryPTVName]
    PrimaryMask = PrimaryMask > 0
    doseArrayThresh = doseArray[PrimaryMask]
    doseArrayThresh = np.percentile(doseArrayThresh, percentile_value)
    factor = PrimaryDose / doseArrayThresh
    print(factor)
    doseArray *= factor

    checkPoint = 80
    fraction = np.sum(doseArray[PrimaryMask] > checkPoint) / np.sum(PrimaryMask)
    print("The fraction of dose greater than {}: {}".format(checkPoint, fraction))

    refDoseArrayThresh = refDoseArray[PrimaryMask]
    refDoseArrayThresh = np.percentile(refDoseArrayThresh, percentile_value)
    refDoseArray *= PrimaryDose / refDoseArrayThresh

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

        struct_ref_dose = refDoseArray[mask]
        struct_ref_dose = np.sort(struct_ref_dose)
        struct_ref_dose = np.insert(struct_ref_dose, 0, 0.0)
        plt.plot(struct_ref_dose, percentile,  color=color, linestyle="--")
    plt.xlim(0, 100)
    plt.xlabel("Dose (Gy)")
    plt.ylabel("Percent Volume (%)")
    plt.title("Dose Volume Histogram of patient {}".format(PatientName))
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    figureFile = os.path.join(planFolder, "DVH_comp.png")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()


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


def drawDoseWash():
    patientFolder = os.path.join(RootFolder, PatientName)
    FastDoseFolder = os.path.join(patientFolder, "FastDose")
    MaskFile = os.path.join(FastDoseFolder, "prep_output", "roi_list.h5")
    PlanFolder = os.path.join(FastDoseFolder, "plan{}".format(planNo))
    doseFile = os.path.join(PlanFolder, "dose.bin")
    densityFile = os.path.join(patientFolder, "density_raw.bin")
    dimensionFile = os.path.join(FastDoseFolder, "prep_output", "dimension.txt")

    with open(dimensionFile, "r") as f:
        lines = f.readlines()
    dimension = lines[0]
    dimension = dimension.replace(" ", ", ")
    dimension = eval(dimension)
    dimension = np.flip(dimension)

    doseArray = np.fromfile(doseFile, dtype=np.float32)
    doseArray = np.reshape(doseArray, dimension)

    densityArray = np.fromfile(densityFile, dtype=np.uint16)
    densityArray = np.reshape(densityArray, dimension)

    masks = getStructures(MaskFile)

    SKINMask = [mask for name, mask in masks if name == "SKIN"]
    assert len(SKINMask) == 1
    SKINMask = SKINMask[0]
    doseArray[SKINMask == 0] = 0

    excludeList = ["PTVMerge", "PTVSeg0", "PTVSeg1", "PTVSeg2", "PTVSeg3", "SKIN"]
    masks = [(a, b) for a, b in masks if a not in excludeList]

    FiguresFolder = os.path.join(PlanFolder, "DoseWash")
    if not os.path.isdir(FiguresFolder):
        os.mkdir(FiguresFolder)
    
    for i in range(dimension[0]):
        densitySlice = densityArray[i, :, :]
        doseSlice = doseArray[i, :, :]
        maskSlice = [(name, array[i, :, :]) for name, array in masks]
        file = os.path.join(FiguresFolder, "{:03d}.png".format(i))
        drawSlice(densitySlice, doseSlice, maskSlice,file)


def drawAxialSagittalCoronal():
    patientFolder = os.path.join(RootFolder, PatientName)
    FastDoseFolder = os.path.join(patientFolder, "FastDose")
    MaskFile = os.path.join(FastDoseFolder, "prep_output", "roi_list.h5")
    PlanFolder = os.path.join(FastDoseFolder, "plan{}".format(planNo))
    doseFile = os.path.join(PlanFolder, "dose.bin")
    densityFile = os.path.join(patientFolder, "density_raw.bin")
    dimensionFile = os.path.join(FastDoseFolder, "prep_output", "dimension.txt")

    with open(dimensionFile, "r") as f:
        lines = f.readlines()
    dimension = lines[0]
    dimension = dimension.replace(" ", ", ")
    dimension = eval(dimension)
    dimension = np.flip(dimension)

    doseArray = np.fromfile(doseFile, dtype=np.float32)
    doseArray = np.reshape(doseArray, dimension)

    densityArray = np.fromfile(densityFile, dtype=np.uint16)
    densityArray = np.reshape(densityArray, dimension)

    masks = getStructures(MaskFile)

    SKINMask = [mask for name, mask in masks if name == "SKIN"]
    assert len(SKINMask) == 1
    SKINMask = SKINMask[0]
    doseArray[SKINMask == 0] = 0

    PTVMask = [mask for name, mask in masks if name == "PTV70"]
    assert len(PTVMask) == 1
    PTVMask = PTVMask[0]
    PTVDose = doseArray[PTVMask > 0]
    percentile_value = 10
    thresh = np.percentile(PTVDose, percentile_value)
    doseArray *= 70 / thresh

    # excludeList = ["PTVMerge", "PTVSeg0", "PTVSeg1", "PTVSeg2", "PTVSeg3", "SKIN", "RingStructure"]
    # excludeList = ["PTVMerge", "PTVSeg0", "PTVSeg1", "PTVSeg2", "PTVSeg3", "SKIN", "Trachea"]
    excludeList = ["PTVMerge", "PTVSeg0", "PTVSeg1", "PTVSeg2", "PTVSeg3", "SKIN"]
    masks = [(a, b) for a, b in masks if a not in excludeList]

    AxialIdx = int(dimension[0] / 2)
    densityAxialSlice = densityArray[AxialIdx, : :]
    doseAxialSlice = doseArray[AxialIdx, :, :]
    maskAxialSlice = [(name, array[AxialIdx, :, :]) for name, array in masks]
    figureFile = os.path.join(PlanFolder, "Axial.png")
    drawSlice(densityAxialSlice, doseAxialSlice, maskAxialSlice, figureFile)

    CoronalIdx = int(dimension[1] / 2)
    densityCoronalSlice = np.flip(densityArray[:, CoronalIdx, :], axis=0)
    doseCoronalSlice = np.flip(doseArray[:, CoronalIdx, :], axis=0)
    maskCoronalSlice = [(name, np.flip(array[:, CoronalIdx, :], axis=0)) for name, array in masks]
    figureFile = os.path.join(PlanFolder, "Coronal.png")
    drawSlice(densityCoronalSlice, doseCoronalSlice, maskCoronalSlice, figureFile)

    SagittalIdx = int(dimension[2] / 2)
    densitySagittalSlice = np.flip(densityArray[:, :, SagittalIdx], axis=0)
    doseSagittalSlice = np.flip(doseArray[:, :, SagittalIdx], axis=0)
    maskSagittalSlice = [(name, np.flip(array[:, :, SagittalIdx], axis=0)) for name, array in masks]
    figureFile = os.path.join(PlanFolder, "Sagittal.png")
    drawSlice(densitySagittalSlice, doseSagittalSlice, maskSagittalSlice, figureFile)

    SagittalIdx = int(dimension[2] / 2)


def drawSlice(densitySlice, doseSlice, maskSlice, outputFile):
    fig, ax = plt.subplots()
    ax.imshow(densitySlice, cmap="gray", vmin=500, vmax=1500)
    for j, entry in enumerate(maskSlice):
        name, maskArray = entry
        color = colors[j]
        if np.sum(maskArray) == 0:
            continue
        contours = measure.find_contours(maskArray)
        initial = True
        for contour in contours:
            if initial:
                plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                initial = False
            else:
                plt.plot(contour[:, 1], contour[:, 0], color=color)

    alpha = (doseSlice > 5) * 0.3
    vmax = 80

    cax = ax.imshow(doseSlice, cmap="jet", vmin=0, vmax=vmax, alpha=alpha)
    cbar = fig.colorbar(cax, ax=ax)
    ax.axis("off")
    ax.legend(loc="upper right", bbox_to_anchor=(-0.02, 1.0))
    plt.tight_layout()
    plt.savefig(outputFile)
    plt.close(fig)
    plt.clf()
    print(outputFile)


def drawDoseWashRef():
    patientFolder = os.path.join(RootFolder, PatientName)
    FastDoseFolder = os.path.join(patientFolder, "FastDose")
    MaskFile = os.path.join(FastDoseFolder, "prep_output", "roi_list.h5")
    doseFile = os.path.join(patientFolder, "dose.bin")
    densityFile = os.path.join(patientFolder, "density_raw.bin")
    dimensionFile = os.path.join(FastDoseFolder, "prep_output", "dimension.txt")

    with open(dimensionFile, "r") as f:
        lines = f.readlines()
    dimension = lines[0]
    dimension = dimension.replace(" ", ", ")
    dimension = eval(dimension)
    dimension = np.flip(dimension)

    doseArray = np.fromfile(doseFile, dtype=np.float32)
    doseArray = np.reshape(doseArray, dimension)

    densityArray = np.fromfile(densityFile, dtype=np.uint16)
    densityArray = np.reshape(densityArray, dimension)

    masks = getStructures(MaskFile)

    SKINMask = [mask for name, mask in masks if name == "SKIN"]
    assert len(SKINMask) == 1
    SKINMask = SKINMask[0]
    doseArray[SKINMask == 0] = 0

    PTVMask = [mask for name, mask in masks if name == "PTV70"]
    assert len(PTVMask) == 1
    PTVMask = PTVMask[0]
    PTVDose = doseArray[PTVMask > 0]
    thresh = np.percentile(PTVDose, 5)
    doseArray *= 70 / thresh

    excludeList = ["PTVMerge", "PTVSeg0", "PTVSeg1", "PTVSeg2", "PTVSeg3", "SKIN"]
    masks = [(a, b) for a, b in masks if a not in excludeList]

    FiguresFolder = os.path.join(FastDoseFolder, "DoseWashRef")
    if not os.path.isdir(FiguresFolder):
        os.mkdir(FiguresFolder)
    
    for i in range(dimension[0]):
        densitySlice = densityArray[i, :, :]
        doseSlice = doseArray[i, :, :]
        maskSlice = [(name, array[i, :, :]) for name, array in masks]
        file = os.path.join(FiguresFolder, "{:03d}.png".format(i))
        drawSlice(densitySlice, doseSlice, maskSlice,file)


def drawAllViews():
    patientFolder = os.path.join(RootFolder, PatientName)
    FastDoseFolder = os.path.join(patientFolder, "FastDose")
    MaskFile = os.path.join(FastDoseFolder, "prep_output", "roi_list.h5")
    PlanFolder = os.path.join(FastDoseFolder, "plan{}".format(planNo))
    doseFile = os.path.join(PlanFolder, "dose.bin")
    densityFile = os.path.join(patientFolder, "density_raw.bin")
    dimensionFile = os.path.join(FastDoseFolder, "prep_output", "dimension.txt")

    with open(dimensionFile, "r") as f:
        lines = f.readlines()
    dimension = lines[0]
    dimension = dimension.replace(" ", ", ")
    dimension = eval(dimension)
    dimension = np.flip(dimension)

    doseArray = np.fromfile(doseFile, dtype=np.float32)
    doseArray = np.reshape(doseArray, dimension)

    densityArray = np.fromfile(densityFile, dtype=np.uint16)
    densityArray = np.reshape(densityArray, dimension)

    masks = getStructures(MaskFile)

    PrimaryName = "PTV70"
    PrimaryMask = [mask for name, mask in masks if name == PrimaryName]
    assert len(PrimaryMask) == 1
    PrimaryMask = PrimaryMask[0]
    percentile_value = 10
    doseArrayThresh = doseArray[PrimaryMask]
    doseArrayThresh = np.percentile(doseArrayThresh, percentile_value)
    doseArray *= 70 / doseArrayThresh

    SKINMask = [mask for name, mask in masks if name == "SKIN"]
    assert len(SKINMask) == 1
    SKINMask = SKINMask[0]
    doseArray[SKINMask == 0] = 0

    excludeList = ["PTVMerge", "PTVSeg0", "PTVSeg1", "PTVSeg2", "PTVSeg3", "SKIN"]
    masks = [(a, b) for a, b in masks if a not in excludeList]

    # axial
    axialFolder = os.path.join(PlanFolder, "AxialDoseWash")
    if not os.path.isdir(axialFolder):
        os.mkdir(axialFolder)
    for i in range(dimension[0]):
        densitySlice = densityArray[i, :, :]
        doseSlice = doseArray[i, :, :]
        maskSlice = [(name, maskArray[i, :, :]) for name, maskArray in masks]
        outputFile = os.path.join(axialFolder, "{:03d}.png".format(i))
        drawSlice(densitySlice, doseSlice, maskSlice, outputFile)
    
    # coronal
    coronalFolder = os.path.join(PlanFolder, "CoronalDoseWash")
    if not os.path.isdir(coronalFolder):
        os.mkdir(coronalFolder)
    for i in range(dimension[1]):
        densitySlice = densityArray[:, i, :]
        doseSlice = doseArray[:, i, :]
        maskSlice = [(name, maskArray[:, i, :]) for name, maskArray in masks]
        outputFile = os.path.join(coronalFolder, "{:03d}.png".format(i))
        drawSlice(densitySlice, doseSlice, maskSlice, outputFile)
    
    # sagittal
    sagittalFolder = os.path.join(PlanFolder, "SagittalDoseWash")
    if not os.path.isdir(sagittalFolder):
        os.mkdir(sagittalFolder)
    for i in range(dimension[2]):
        densitySlice = densityArray[:, :, i]
        doseSlice = doseArray[:, :, i]
        maskSlice = [(name, maskArray[:, :, i]) for name, maskArray in masks]
        outputFile = os.path.join(sagittalFolder, "{:03d}.png".format(i))
        drawSlice(densitySlice, doseSlice, maskSlice, outputFile)
    

if __name__ == "__main__":
    drawDVH_opt()
    # drawAxialSagittalCoronal()
    # drawDoseWash()
    # drawAllViews()
    # drawDoseWashRef()