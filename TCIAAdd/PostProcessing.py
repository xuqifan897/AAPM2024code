import os
import numpy as np
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator
import h5py
from skimage import measure

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
RootFolder = "/data/qifan/projects/FastDoseWorkplace/TCIAAdd"
PatientName = "190"

def drawDVH_opt():
    """
    Draw DVH for the optimized plan
    """
    patientFolder = os.path.join(RootFolder, PatientName)
    planFolder = os.path.join(patientFolder, "FastDose", "plan1")
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
    exclude = ["PTVMerge", "PTVSeg0", "PTVSeg1", "PTVSeg2", "PTVSeg3", "SKIN"]
    if True:
        exclude.append("RingStructure")
    structures = [a for a in structures if a not in exclude]
    masks = {a: b for a, b in masks.items() if a in structures}

    percentile_value = 5
    if PatientName in ["003", "125"]:
        percentile_value = 10

    # Normalize
    PTVList = [(a, eval(a[3:])) for a in structures if "PTV" in a]
    PTVList.sort(key=lambda a: a[1], reverse=True)
    PrimaryPTVName, PrimaryDose = PTVList[0]
    PrimaryMask = masks[PrimaryPTVName]
    PrimaryMask = PrimaryMask > 0
    doseArrayThresh = doseArray[PrimaryMask]
    doseArrayThresh = np.percentile(doseArrayThresh, percentile_value)
    factor = PrimaryDose / doseArrayThresh
    print(factor)
    doseArray *= factor
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
    PlanFolder = os.path.join(FastDoseFolder, "plan1")
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
    excludeList = ["PTVMerge", "PTVSeg0", "PTVSeg1", "PTVSeg2", "PTVSeg3", "SKIN", "RingStructure"]
    masks = [(a, b) for a, b in masks if a not in excludeList]

    FiguresFolder = os.path.join(PlanFolder, "DoseWash")
    if not os.path.isdir(FiguresFolder):
        os.mkdir(FiguresFolder)
    
    for i in range(dimension[0]):
        densitySlice = densityArray[i, :, :]
        doseSlice = doseArray[i, :, :]
        plt.imshow(densitySlice, cmap="gray", vmin=500, vmax=1500)
        for j, entry in enumerate(masks):
            name, maskArray = entry
            color = colors[j]
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
        plt.imshow(doseSlice, cmap="jet", vmin=0, vmax=80, alpha=0.3)
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        plt.tight_layout()
        FigureFile = os.path.join(FiguresFolder, "{:03d}.png".format(i))
        plt.savefig(FigureFile)
        plt.clf()
        print(FigureFile)


if __name__ == "__main__":
    drawDVH_opt()
    # drawDoseWash()