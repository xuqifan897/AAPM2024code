import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from skimage import io, measure


def doseAnalyze():
    prep_output = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_output"
    prep_result = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_result"
    dimension = (149, 220, 220)
    VOI_exclude = "Skin"

    roi_listFile = os.path.join(prep_output, "roi_list.h5")
    densityFile = os.path.join(prep_output, "density.raw")
    doseFile = os.path.join(prep_result, "dose.bin")

    density = np.fromfile(densityFile, dtype=np.float32)
    dose = np.fromfile(doseFile, dtype=np.float32)
    density = np.reshape(density, dimension)
    dose = np.reshape(dose, dimension)
    file = h5py.File(roi_listFile, 'r')
    structures_filtered = list(file.keys())
    structures_filtered.remove(VOI_exclude)
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


def drawDoseWash():
    prep_output = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_output"
    prep_result = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_result"
    dimension = (149, 220, 220)
    VOI_exclude = "Skin"

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
    structures_filtered.remove(VOI_exclude)
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
        print(structSize, structCropSize, structCropStart)

        structMask = np.array(structMask)
        structMask = np.reshape(structMask, structCropSize)
        struct_mask = np.zeros(structSize, dtype=bool)
        struct_mask[structCropStart[0]: structCropStart[0] + structCropSize[0],
            structCropStart[1]: structCropStart[1] + structCropSize[1],
            structCropStart[2]: structCropStart[2] + structCropSize[2]] = structMask
        masks[struct_name] = struct_mask
    
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


def compare_dose():
    shape = [149, 220, 220]
    doseOldFile = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_bench/doseOld.bin"
    doseOld = np.fromfile(doseOldFile, dtype=np.float64)
    doseOld = np.reshape(doseOld, shape)
    doseOld = np.transpose(doseOld, [0, 2, 1])

    doseNewFile = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_result_v1/dose.bin"
    doseNew = np.fromfile(doseNewFile, dtype=np.float32)
    doseNew = np.reshape(doseNew, shape)

    VOI_exclude = ["Skin", "RingStructure"]
    roi_listFile = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_output/roi_list.h5"
    densityFile = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_output/density.raw"

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
        print(structSize, structCropSize, structCropStart)

        structMask = np.array(structMask)
        structMask = np.reshape(structMask, structCropSize)
        struct_mask = np.zeros(structSize, dtype=bool)
        struct_mask[structCropStart[0]: structCropStart[0] + structCropSize[0],
            structCropStart[1]: structCropStart[1] + structCropSize[1],
            structCropStart[2]: structCropStart[2] + structCropSize[2]] = structMask
        masks[struct_name] = struct_mask
    
    # draw DVH
    numStructs = len(structures_filtered)
    color_values = np.linspace(0, 1, numStructs)
    color_map = plt.get_cmap('viridis')
    colors = [color_map(value) for value in color_values]

    # normalize against D95
    PTV_mask = masks["PTV_ENLARGED"]
    PTVDoseOld = doseOld[PTV_mask]
    doseOldThresh = np.percentile(PTVDoseOld, 5)
    doseOld /= doseOldThresh

    PTVDoseNew = doseNew[PTV_mask]
    doseNewThresh = np.percentile(PTVDoseNew, 5)
    doseNew /= doseNewThresh

    # draw DVH for doseOld
    for i in range(numStructs):
        color = colors[i]
        struct_name = structures_filtered[i]
        mask = masks[struct_name]
        maskedDose = doseOld[mask]
        DrawDVHLine(maskedDose, color)

    # draw DVH for doseNew
    for i in range(numStructs):
        color = colors[i]
        struct_name = structures_filtered[i]
        mask = masks[struct_name]
        maskedDose = doseNew[mask]
        DrawDVHLine(maskedDose, color, '--')

    plt.legend(structures_filtered)
    plt.title("DVH comparison between old (solid) and new (dashed) methods")
    plt.xlabel('Dose (Gy)')
    plt.ylabel('Fractional volume (%)')
    figureFile = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_bench/DVH_comp.png"
    plt.savefig(figureFile)

def drawDoseWash_old():
    prep_output = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_output"
    prep_result = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_result"
    dimension = (149, 220, 220)
    VOI_exclude = "Skin"

    roi_listFile = os.path.join(prep_output, "roi_list.h5")
    densityFile = os.path.join(prep_output, "density.raw")
    doseFile = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_bench/dose.bin"

    density = np.fromfile(densityFile, dtype=np.float32)
    dose = np.fromfile(doseFile, dtype=np.float64)
    density = np.reshape(density, dimension)
    dose = np.reshape(dose, dimension)
    dose = np.transpose(dose, [0, 2, 1])
    dose /= np.max(dose)  # normalize

    file = h5py.File(roi_listFile, 'r')
    structures_filtered = list(file.keys())
    structures_filtered.remove(VOI_exclude)
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
        print(structSize, structCropSize, structCropStart)

        structMask = np.array(structMask)
        structMask = np.reshape(structMask, structCropSize)
        struct_mask = np.zeros(structSize, dtype=bool)
        struct_mask[structCropStart[0]: structCropStart[0] + structCropSize[0],
            structCropStart[1]: structCropStart[1] + structCropSize[1],
            structCropStart[2]: structCropStart[2] + structCropSize[2]] = structMask
        masks[struct_name] = struct_mask
    
    numStructs = len(masks)
    color_values = np.linspace(0, 1, numStructs)
    color_map = plt.get_cmap('viridis')
    colors = [color_map(value) for value in color_values]

    imageFolder = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_bench/doseOldShow"
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


def pick_lines():
    """
    This function copies the lines of the beamlist_full to
    get a new beamlist
    """
    beamIdx = [19, 5, 6, 18, 14, 4, 38, 61, 86, 28, 10, 124, 429, 269, 444, 88, 289, 143, 117, 447, 417]
    sourceFile = "/data/qifan/projects/AAPM2024/beamlist_full.txt"
    targetFile = "/data/qifan/projects/AAPM2024/beamlist.txt"
    with open(sourceFile, 'r') as f:
        lines = f.readlines()
    lines_selected = [lines[i] for i in beamIdx]
    with open(targetFile, 'w') as f:
        f.writelines(lines_selected)

if __name__ == '__main__':
    # doseAnalyze()
    # drawDoseWash()
    compare_dose()
    # drawDoseWash_old()
    # pick_lines()