import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
from skimage import measure

patientFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/Patient003"
doseShape = None

def getDoseShape():
    densityFile = os.path.join(patientFolder, "FastDose", 'prep_output', "dimension.txt")
    with open(densityFile, "r") as f:
        lines = f.readlines()
    line0 = lines[0]
    line0 = line0.split(" ")
    line0 = [eval(a) for a in line0]
    line0.reverse()
    global doseShape
    doseShape = tuple(line0)
    print(doseShape)


def showDose():
    optFolder = os.path.join(patientFolder, "QihuiRyan")
    binaryDoseFile = os.path.join(optFolder, "binaryDose2000.bin")
    doseArray = np.fromfile(binaryDoseFile, dtype=np.float32)
    doseArray = np.reshape(doseArray, doseShape)
    doseArray = np.transpose(doseArray, axes=(0, 2, 1))
    maxDose = np.max(doseArray)

    densityFile = os.path.join(patientFolder, "FastDose", "prep_output", 'density.raw')
    densityArray = np.fromfile(densityFile, dtype=np.float32)
    densityArray = np.reshape(densityArray, doseShape)

    structFile = os.path.join(patientFolder, "FastDose", "prep_output", "roi_list.h5")
    Masks = getStructures(structFile)
    VOI_exclude = ["RingStructure"]
    Masks = [a for a in Masks if a[0] not in VOI_exclude]
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())

    if False:
        PTVmask = Masks[0][1]
        PTVmask = PTVmask > 0
        DosePTV = doseArray[PTVmask]
        thresh = np.percentile(DosePTV, 5)
        print(maxDose, thresh)

    visFolder = os.path.join(optFolder, "doseView")
    if not os.path.isdir(visFolder):
        os.mkdir(visFolder)
    numSlices = doseArray.shape[0]
    for i in range(numSlices):
        fig, ax = plt.subplots(figsize=(8, 5))
        phantomSlice = densityArray[i, :, :]
        ax.imshow(phantomSlice, cmap="gray")
        doseSlice = doseArray[i, :, :]
        ax.imshow(doseSlice, cmap="jet", vmin=0, vmax=maxDose, alpha=0.3)
        for j, entry in enumerate(Masks):
            color = colors[j]
            name, array = entry
            maskSlice = array[i, :, :]
            contours = measure.find_contours(maskSlice)
            initial = True
            for contour in contours:
                if initial:
                    ax.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                    initial = False
                else:
                    ax.plot(contour[:, 1], contour[:, 0], color=color)
        ax.legend(loc="upper right", bbox_to_anchor=(1.05, 1))
        imageFile = os.path.join(visFolder, "{:03d}.png".format(i))
        plt.savefig(imageFile)
        plt.clf()
        print(imageFile)


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


def DVH_comp():
    """
    This function draws the DVH plots of the FastDose method and
    Qihui/Ryan method side-by-side
    """
    QihuiRyanFolder = os.path.join(patientFolder, "QihuiRyan")
    FastDoseFolder = os.path.join(patientFolder, "FastDose")
    prescription_dose = 20

    QihuiDoseFile = os.path.join(QihuiRyanFolder, "binaryDose2000.bin")
    QihuiDoseArray = np.fromfile(QihuiDoseFile, dtype=np.float32)
    QihuiDoseArray = np.reshape(QihuiDoseArray, doseShape)
    QihuiDoseArray = np.transpose(QihuiDoseArray, axes=(0, 2, 1))

    FastDoseFile = os.path.join(FastDoseFolder, "plan1", "dose.bin")
    FastDoseArray = np.fromfile(FastDoseFile, dtype=np.float32)
    FastDoseArray = np.reshape(FastDoseArray, doseShape)

    structFile = os.path.join(FastDoseFolder, "prep_output", "roi_list.h5")
    Masks = getStructures(structFile)
    VOI_exclude = ["RingStructure", "SKIN"]
    Masks = [a for a in Masks if a[0] not in VOI_exclude]
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())

    # normalize    
    PTVname = "PTV"
    for name, mask in Masks:
        if name == PTVname:
            PTVmask = mask
            break
    PTVmask = PTVmask > 0
    QihuiPTVDose = QihuiDoseArray[PTVmask]
    QihuiDoseThresh = np.percentile(QihuiPTVDose, 5)
    QihuiDoseArray *= prescription_dose / QihuiDoseThresh

    FastDosePTV = FastDoseArray[PTVmask]
    FastDoseThresh = np.percentile(FastDosePTV, 5)
    FastDoseArray *= prescription_dose / FastDoseThresh

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, entry in enumerate(Masks):
        color = colors[i]
        name, mask = entry
        mask = mask > 0
        QihuiROIDose = QihuiDoseArray[mask]
        QihuiROIDose = np.sort(QihuiROIDose)
        QihuiROIDose = np.insert(QihuiROIDose, 0, 0)

        FastDoseROI = FastDoseArray[mask]
        FastDoseROI = np.sort(FastDoseROI)
        FastDoseROI = np.insert(FastDoseROI, 0, 0)

        numEntries = np.sum(mask) + 1
        y_axis = np.linspace(100, 0, numEntries)

        ax.plot(FastDoseROI, y_axis, color=color, linestyle="-", label=name)
        ax.plot(QihuiROIDose, y_axis, color=color, linestyle="--")
    fig.legend(loc="upper right", bbox_to_anchor=(1, 0.9))
    ax.set_xlim(0, 30)
    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Percentile")
    patientName = patientFolder.split("/")[-1]
    ax.set_title(patientName + " DVH")
    imageFile = os.path.join(QihuiRyanFolder, "DVHcomp.png")
    plt.savefig(imageFile)
    plt.clf()


if __name__ == "__main__":
    getDoseShape()
    # showDose()
    DVH_comp()