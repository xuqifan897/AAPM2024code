import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

sourceFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas"
targetFolder = os.path.join(sourceFolder, "plansAngleCorrect")
figureFolder = "/data/qifan/projects/AAPM2024/manufigures"
manuFiguresFolder = "/data/qifan/projects/AAPM2024/manufigures"
colorMap = None
isoRes = 2.5  # mm
numPatients = 5

structGroup1 = ['PTV', 'duodenum', 'stomach', 'gallbladder', 'liver']
structGroup2 = ['kidney_right', 'kidney_left', 'spine', 'colon', 'aorta']
structGroup3 = ['spinal_cord', 'spleen', 'small_bowel', 'lung_left', 'lung_right']
structGroup4 = ['esophagus', 'heart', 'trachea', 'urinary_bladder']
groups = [structGroup1, structGroup2, structGroup3, structGroup4]

def hex_to_rgb(hex_color):
    """Converts a color from hexadecimal format to RGB."""
    hex_color = hex_color.lstrip('#')
    result = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    result = np.array(result) / 255
    result = "{} {} {}".format(*result)
    return result


def StructsInit():
    # rank the structures according to average dose
    bbox = "SKIN"
    StructuresScore = {}
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i + 1)
        doseOurs = os.path.join(targetFolder, patientName, "FastDose", "plan1", "dose.bin")
        doseOurs = np.fromfile(doseOurs, dtype=np.float32)
        maskFolder = os.path.join(sourceFolder, patientName, "InputMask")
        structuresLocal = [b for a in os.listdir(maskFolder) if (b:=a.split(".")[0]) != bbox]
        
        for struct in structuresLocal:
            mask = os.path.join(maskFolder, struct + ".bin")
            mask = np.fromfile(mask, dtype=np.uint8).astype(bool)
            structMeanDose = np.mean(doseOurs[mask])
            if struct in StructuresScore:
                StructuresScore[struct].append(structMeanDose)
            else:
                StructuresScore[struct] = [structMeanDose]
    
    StructuresRank = []
    for struct in StructuresScore:
        structAvg = np.mean(StructuresScore[struct])
        StructuresRank.append((struct, structAvg))
    StructuresRank.sort(key=lambda a: a[1], reverse=True)
    StructuresRank = [a[0] for a in StructuresRank]
    print(len(StructuresRank))


def colorMapInit():
    global colorMap
    colorMap = {}
    structuresList = []
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i + 1)
        maskFolder = os.path.join(sourceFolder, patientName, "InputMask")
        for file in os.listdir(maskFolder):
            struct = file.split(".")[0]
            if struct not in structuresList:
                structuresList.append(struct)
    bbox_name = "SKIN"
    ptv_name = "PTV"
    assert bbox_name in structuresList and ptv_name in structuresList
    structuresList.remove(bbox_name)
    structuresList.remove(ptv_name)
    structuresList.sort()
    structuresList.insert(0, bbox_name)
    structuresList.insert(0, ptv_name)

    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
    for i in range(len(structuresList)):
        struct = structuresList[i]
        color = colors[i]
        colorMap[struct] = color


def GridFigure():
    normalizePercentile = 10  # %
    prescriptionDose = 20
    numGroups = 4
    nRows = numPatients + 1
    nCols = numGroups + 1
    linewidth = 2
    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(nRows, nCols, width_ratios=[0.2, 5, 5, 5, 5], height_ratios=[4, 4, 4, 4, 4, 0.2])

    # Create the common ylabel
    ylabel_block = fig.add_subplot(gs[:-1, 0])
    ylabel_block.text(0.9, 0.5, "Fractional Volume (%)", ha="center", va="center",
        rotation="vertical", fontsize=18)
    ylabel_block.axis("off")

    # Create the common xlabel
    xlabel_block = fig.add_subplot(gs[-1, 1:])
    xlabel_block.text(0.5, 0.9, "Dose (Gy)", ha="center", va="center", fontsize=18)
    xlabel_block.axis("off")

    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i + 1)
        dimension = os.path.join(sourceFolder, patientName, "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = eval(dimension.replace(" ", ", "))  # (x, y, z)
        dimension_flip = np.flip(dimension)  # (z, y, x)

        clinicalDose = os.path.join(sourceFolder, patientName, "doseRef.bin")
        clinicalDose = np.fromfile(clinicalDose, dtype=np.float32)
        clinicalDose = np.reshape(clinicalDose, dimension_flip)  # (z, y, x)

        ourDose = os.path.join(targetFolder, patientName, "FastDose", "plan1", "dose.bin")
        ourDose = np.fromfile(ourDose, dtype=np.float32)
        ourDose = np.reshape(ourDose, dimension_flip)  # (z, y, x)

        QihuiRyanDose = os.path.join(targetFolder, patientName, "QihuiRyan", "doseRef.bin")
        QihuiRyanDose = np.fromfile(QihuiRyanDose, dtype=np.float32)
        QihuiRyanDose = np.reshape(QihuiRyanDose, dimension_flip)

        maskFolder = os.path.join(sourceFolder, patientName, "InputMask")

        # normalize
        ptv = os.path.join(maskFolder, "PTV.bin")
        ptv = np.fromfile(ptv, dtype=np.uint8)
        ptv = np.reshape(ptv, dimension_flip).astype(bool)  # (z, y, x)
        thresh_clinicalDose = np.percentile(clinicalDose[ptv], normalizePercentile)
        clinicalDose *= prescriptionDose / thresh_clinicalDose

        thresh_ourDose = np.percentile(ourDose[ptv], normalizePercentile)
        ourDose *= prescriptionDose / thresh_ourDose

        thresh_QihuiRyanDOse = np.percentile(QihuiRyanDose[ptv], normalizePercentile)
        QihuiRyanDose *= prescriptionDose / thresh_QihuiRyanDOse

        for j in range(numGroups):
            block = fig.add_subplot(gs[i, j+1])
            currentGroup = groups[j]
            for struct in currentGroup:
                mask = os.path.join(maskFolder, struct + ".bin")
                if not os.path.isfile(mask):
                    continue
                mask = np.fromfile(mask, dtype=np.uint8)
                mask = np.reshape(mask, dimension_flip).astype(bool)  # (z, y, x)

                clinicalDoseMask = clinicalDose[mask]
                clinicalDoseMask = np.sort(clinicalDoseMask)
                clinicalDoseMask = np.insert(clinicalDoseMask, 0, 0)

                ourDoseMask = ourDose[mask]
                ourDoseMask = np.sort(ourDoseMask)
                ourDoseMask = np.insert(ourDoseMask, 0, 0)

                QihuiRyanDoseMask = QihuiRyanDose[mask]
                QihuiRyanDoseMask = np.sort(QihuiRyanDoseMask)
                QihuiRyanDoseMask = np.insert(QihuiRyanDoseMask, 0, 0)

                nPoints = np.sum(mask) + 1
                yAxis = (1 - np.arange(nPoints) / (nPoints - 1)) * 100
                color = colorMap[struct]

                block.plot(ourDoseMask, yAxis, color=color, linestyle="-", linewidth=linewidth)
                block.plot(QihuiRyanDoseMask, yAxis, color=color, linestyle="--", linewidth=linewidth)
                block.plot(clinicalDoseMask, yAxis, color=color, linestyle="-.", linewidth=linewidth)
            block.tick_params(axis="both", labelsize=15)
            print(patientName, j)
    plt.tight_layout()
    figureFile = os.path.join(manuFiguresFolder, "FastDosePancreasDVHFull.png")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()


if __name__ == "__main__":
    # StructsInit()
    colorMapInit()
    GridFigure()