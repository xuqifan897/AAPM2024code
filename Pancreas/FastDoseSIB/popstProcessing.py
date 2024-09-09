import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import measure

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
sourceFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB"
numPatients = 5

colorMap = {}
def colorMapInit():
    global colorMap
    structsList = []
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        maskFolder = os.path.join(sourceFolder, patientName, "InputMask")
        structs_local = [a.split(".")[0] for a in os.listdir(maskFolder)]
        for struct in structs_local:
            if struct not in structsList:
                structsList.append(struct)
    ptv = "ROI"
    body = "SKIN"
    assert ptv in structsList and  body in structsList
    structsList.remove(ptv)
    structsList.remove(body)
    structsList.sort()
    structsList.insert(0, ptv)
    structsList.append(body)
    for i in range(len(structsList)):
        colorMap[structsList[i]] = colors[i]


def initDVHComp():
    """
    This function compares the initial DVH between the clinical dose and our dose
    """
    dvhInitFolder = os.path.join(sourceFolder, "dvhInit")
    if not os.path.isdir(dvhInitFolder):
        os.mkdir(dvhInitFolder)
    ptv = "ROI"
    body = "SKIN"
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)
        FastDoseFolder = os.path.join(patientFolder, "FastDose")

        doseClinic = os.path.join(patientFolder, "doseNorm.bin")
        doseClinic = np.fromfile(doseClinic, dtype=np.float32)
        
        doseOurs = os.path.join(FastDoseFolder, "plan1", "dose.bin")
        doseOurs = np.fromfile(doseOurs, dtype=np.float32)
        assert doseClinic.size == (phantomSize := doseOurs.size)

        maskFolder = os.path.join(patientFolder, "InputMask")
        structures = [a.split(".")[0] for a in os.listdir(maskFolder)]
        structures.remove(body)

        maskDict = {}
        for struct in structures:
            mask = os.path.join(maskFolder, struct+".bin")
            mask = np.fromfile(mask, dtype=np.uint8).astype(bool)
            maskDict[struct] = mask

        # normalize against average dose
        ptvMask = maskDict[ptv]
        ptvDoseOurs = np.mean(doseOurs[ptvMask])
        ptvDoseClinic = np.mean(doseClinic[ptvMask])
        doseOurs *= ptvDoseClinic / ptvDoseOurs
        
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, mask in maskDict.items():
            color = colorMap[name]
            doseOursMasked = doseOurs[mask]
            doseOursMasked = np.sort(doseOursMasked)
            doseOursMasked = np.insert(doseOursMasked, 0, 0.0)

            doseClinicMasked = doseClinic[mask]
            doseClinicMasked = np.sort(doseClinicMasked)
            doseClinicMasked = np.insert(doseClinicMasked, 0, 0.0)

            assert doseOursMasked.size == (nPoints := doseClinicMasked.size)
            yAxis = (1 - np.arange(nPoints) / (nPoints - 1)) * 100
            ax.plot(doseOursMasked, yAxis, color=color, linestyle="-", label=name)
            ax.plot(doseClinicMasked, yAxis, color=color, linestyle="--")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.set_xlabel("Dose (Gy)")
        ax.set_ylabel("Percentile (%)")
        ax.set_title(patientName)
        plt.tight_layout()
        file = os.path.join(dvhInitFolder, patientName + ".png")
        plt.savefig(file)
        plt.close(fig)
        plt.clf()
        print(file)


def QihuiRyanDVH():
    """
    This function compares the initial DVH between the clinical dose and our dose
    """
    dvhInitFolder = os.path.join(sourceFolder, "dvhQihuiRyan")
    if not os.path.isdir(dvhInitFolder):
        os.mkdir(dvhInitFolder)
    ptv = "ROI"
    body = "SKIN"
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)
        FastDoseFolder = os.path.join(patientFolder, "QihuiRyan")

        doseClinic = os.path.join(patientFolder, "doseNorm.bin")
        doseClinic = np.fromfile(doseClinic, dtype=np.float32)
        
        doseOurs = os.path.join(FastDoseFolder, "doseQihuiRyan.bin")
        doseOurs = np.fromfile(doseOurs, dtype=np.float32)
        assert doseClinic.size == (phantomSize := doseOurs.size)

        maskFolder = os.path.join(patientFolder, "InputMask")
        structures = [a.split(".")[0] for a in os.listdir(maskFolder)]
        structures.remove(body)

        maskDict = {}
        for struct in structures:
            mask = os.path.join(maskFolder, struct+".bin")
            mask = np.fromfile(mask, dtype=np.uint8).astype(bool)
            maskDict[struct] = mask

        # normalize against average dose
        ptvMask = maskDict[ptv]
        ptvDoseOurs = np.mean(doseOurs[ptvMask])
        ptvDoseClinic = np.mean(doseClinic[ptvMask])
        doseOurs *= ptvDoseClinic / ptvDoseOurs
        
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, mask in maskDict.items():
            color = colorMap[name]
            doseOursMasked = doseOurs[mask]
            doseOursMasked = np.sort(doseOursMasked)
            doseOursMasked = np.insert(doseOursMasked, 0, 0.0)

            doseClinicMasked = doseClinic[mask]
            doseClinicMasked = np.sort(doseClinicMasked)
            doseClinicMasked = np.insert(doseClinicMasked, 0, 0.0)

            assert doseOursMasked.size == (nPoints := doseClinicMasked.size)
            yAxis = (1 - np.arange(nPoints) / (nPoints - 1)) * 100
            ax.plot(doseOursMasked, yAxis, color=color, linestyle="-", label=name)
            ax.plot(doseClinicMasked, yAxis, color=color, linestyle="--")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.set_xlabel("Dose (Gy)")
        ax.set_ylabel("Percentile (%)")
        ax.set_title(patientName)
        plt.tight_layout()
        file = os.path.join(dvhInitFolder, patientName + ".png")
        plt.savefig(file)
        plt.close(fig)
        plt.clf()
        print(file)


def doseWashPlot():
    "This function plots the dose wash of our generated plans"
    doseWashFolder = os.path.join(sourceFolder, "doseWash")
    if not os.path.isdir(doseWashFolder):
        os.mkdir(doseWashFolder)
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)
        dimension = os.path.join(patientFolder, "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)  # (x, y, z)
        dimension_flip = np.flip(dimension)  # (z, y, x)

        density = os.path.join(patientFolder, "density_raw.bin")
        density = np.fromfile(density, dtype=np.uint16)
        density = np.reshape(density, dimension_flip)  # (z, y, x)

        dose = os.path.join(patientFolder, "FastDose", "plan1", "dose.bin")
        dose = np.fromfile(dose, dtype=np.float32)
        dose = np.reshape(dose, dimension_flip)  # (z, y, x)
        maxDose = np.max(dose)

        maskFolder = os.path.join(patientFolder, "InputMask")
        structures = [a.split(".")[0] for a in os.listdir(maskFolder)]
        masks = {}
        for name in structures:
            struct = os.path.join(maskFolder, name + ".bin")
            struct = np.fromfile(struct, dtype=np.uint8)
            struct = np.reshape(struct, dimension_flip)
            masks[name] = struct
        
        nSlices = dimension_flip[0]
        patientDoseWashFolder = os.path.join(doseWashFolder, patientName)
        drawDoseWash(density, dose, masks, patientDoseWashFolder)


def drawDoseWash(density, dose, masks, figureFolder):
    # order: (z, y, x)
    if not os.path.isdir(figureFolder):
        os.mkdir(figureFolder)
    nSlices = density.shape[0]
    maxDose = np.max(dose)
    for i in range(nSlices):
        densitySlice = density[i, :, :]
        doseSlice = dose[i, :, :]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(densitySlice, cmap="gray", vmin=0, vmax=1600)
        doseMap = ax.imshow(doseSlice, cmap="jet", vmin=0, vmax=maxDose, alpha=0.3*(doseSlice>3))
        for name, maskArray in masks.items():
            color = colorMap[name]
            maskSlice = maskArray[i, :, :]
            if np.any(maskSlice) == 0:
                continue
            contours = measure.find_contours(maskSlice)
            initial = True
            for contour in contours:
                if initial:
                    ax.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                    initial = False
                else:
                    ax.plot(contour[:, 1], contour[:, 0], color=color)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        fig.colorbar(doseMap, ax=ax, location="left")
        fig.tight_layout()
        file = os.path.join(figureFolder, "{:03d}.png".format(i))
        plt.savefig(file)
        plt.close(fig)
        plt.clf()
        print(file)


if __name__ == "__main__":
    colorMapInit()
    # initDVHComp()
    QihuiRyanDVH()
    # doseWashPlot()