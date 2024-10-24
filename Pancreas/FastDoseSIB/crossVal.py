import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from skimage import measure

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
sourceFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB"
figureFolder = os.path.join(sourceFolder, "R50SanityCheck")
if not os.path.isdir(figureFolder):
    os.mkdir(figureFolder)
numPatients = 5

def main():
    factor = 0.5
    prescriptionLevel = 10
    
    content = [
        "PTV dose: {}%".format(100 - prescriptionLevel),
        "R{} comparison".format(int(factor * 100)),
        "| Patient | UHPP | SOTA | Clinical |", "| - | - | - | - |"]
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)
        dimension = os.path.join(patientFolder, "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension_flip = np.flip(dimension)
        
        doseRef = os.path.join(patientFolder, "doseNorm.bin")
        doseRef = np.fromfile(doseRef, dtype=np.float32)
        doseRef = np.reshape(doseRef, dimension_flip)

        doseUHPP = os.path.join(patientFolder, "FastDose", "plan1", "dose.bin")
        doseUHPP = np.fromfile(doseUHPP, dtype=np.float32)
        doseUHPP = np.reshape(doseUHPP, dimension_flip)

        doseSOTA = os.path.join(patientFolder, "QihuiRyan", "doseQihuiRyan.bin")
        if i in [0, 1, 2]:
            doseSOTA = os.path.join(patientFolder, 'QihuiRyan', "doseQihuiRyan_UHPP.bin")
        doseSOTA = np.fromfile(doseSOTA, dtype=np.float32)
        doseSOTA = np.reshape(doseSOTA, dimension_flip)

        bodyMask = os.path.join(patientFolder, "InputMask", "SKIN.bin")
        bodyMask = np.fromfile(bodyMask, dtype=np.uint8)
        bodyMask = np.reshape(bodyMask, dimension_flip) > 0
        notBody = np.logical_not(bodyMask)

        PTVMask = os.path.join(patientFolder, "InputMask", "ROI.bin")
        PTVMask = np.fromfile(PTVMask, dtype=np.uint8)
        PTVMask = np.reshape(PTVMask, dimension_flip) > 0

        density = os.path.join(patientFolder, "density_raw.bin")
        density = np.fromfile(density, dtype=np.uint16)
        density = np.reshape(density, dimension_flip)

        doseList = [doseUHPP, doseSOTA, doseRef]
        prescriptionDose = 20
        R50Thresh = prescriptionDose * factor
        ptvVoxels = np.sum(PTVMask)
        RscoreList = []
        for doseArray in doseList:
            doseArray[notBody] = 0.0
            PTVDose = doseArray[PTVMask]
            threshLevel = np.percentile(PTVDose, prescriptionLevel)
            doseArray *= prescriptionDose / threshLevel
            doseGreaterThanDose50 = doseArray > R50Thresh
            Rscore = np.sum(doseGreaterThanDose50) / ptvVoxels
            RscoreList.append(Rscore)
        currentLine = "| {:03d} | {:.3f} | {:.3f} | {:.3f} |".format(i+1, *RscoreList)
        content.append(currentLine)

        if False:
            patientImageFolder = os.path.join(figureFolder, patientName)
            if not os.path.isdir(patientImageFolder):
                os.mkdir(patientImageFolder)
            for j in range(dimension_flip[0]):
                densitySlice = density[j, :, :]
                PTVSlice = PTVMask[j, :, :]
                fig = plt.figure(figsize=(12, 4))
                gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
                PTVcontours = measure.find_contours(PTVSlice)
                for k in range(3):
                    current_block = fig.add_subplot(gs[0, k])
                    current_block.imshow(densitySlice, vmin=0, vmax=1200, cmap="gray")
                    doseSlice = doseList[k][j, :, :]
                    current_block.imshow(doseSlice, vmin=0, vmax=50, alpha=0.3, cmap="jet")
                    doseR50 = doseSlice > R50Thresh
                    doseR50Contours = measure.find_contours(doseR50)
                    for contour in PTVcontours:
                        current_block.plot(contour[:, 1], contour[:, 0], color=colors[0], linewidth=1, linestyle="--")
                    for contour in doseR50Contours:
                        current_block.plot(contour[:, 1], contour[:, 0], color=colors[1], linewidth=1, linestyle="--")
                fig.tight_layout()
                figureFile = os.path.join(patientImageFolder, "{:03d}.png".format(j+1))
                plt.savefig(figureFile)
                plt.close(fig)
                plt.clf()
                print(figureFile)
    content = "\n".join(content)
    print(content)


def cross_validation_beamlist_gen():
    """
    Here we are gonna use the beams generated with the baseline method to optimize in the SOTA method
    """
    for i in range(1, numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)
        selectedAngles = os.path.join(patientFolder, "QihuiRyan", "selected_angles_UHPP.csv")
        with open(selectedAngles, "r") as f:
            lines = f.readlines()
        lines = lines[1:]  # remote the title line
        lines = [eval(a) for a in lines]
        newLines = []
        if False:
            for line in lines:
                newLines.append("{:.4f} {:.4f} {:.4f}".format(line[1], line[2], 0))
            newLines = "\n".join(newLines)
            crossValFolder = os.path.join(patientFolder, "crossVal")
            if not os.path.isdir(crossValFolder):
                os.mkdir(crossValFolder)
            
            beamListFile = os.path.join(crossValFolder, "beamlist_UHPP.txt")
            with open(beamListFile, "w") as f:
                f.write(newLines)
            print(patientName)
        else:
            beamListOrg = os.path.join(patientFolder, "FastDose", "beamlist.txt")
            with open(beamListOrg, "r") as f:
                beamListOrg = f.readlines()
            newBeamList = []
            for line in lines:
                beamIdx = int(line[0]) - 1
                newBeamList.append(beamListOrg[beamIdx])
            newBeamList = "".join(newBeamList)

            crossValFolder = os.path.join(patientFolder, "crossVal")
            if not os.path.isdir(crossValFolder):
                os.mkdir(crossValFolder)
            targetBeamList = os.path.join(crossValFolder, "beamlist_UHPP.txt")
            with open(targetBeamList, "w") as f:
                f.write(newBeamList)
            print(targetBeamList)


def crossValR50Calc():
    content = ["R50 comparison", "| Patient | UHPP | SOTA | SOTA angle, UHPP polishing | SOTA BOO, UHPP dose | SOTA BOO, UHPP dose, UHPP polishing | Clinical |", "| - | - | - | - |"]
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)
        dimension = os.path.join(patientFolder, "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension_flip = np.flip(dimension)
        
        doseRef = os.path.join(patientFolder, "doseNorm.bin")
        doseRef = np.fromfile(doseRef, dtype=np.float32)
        doseRef = np.reshape(doseRef, dimension_flip)

        doseUHPP = os.path.join(patientFolder, "FastDose", "plan1", "dose.bin")
        doseUHPP = np.fromfile(doseUHPP, dtype=np.float32)
        doseUHPP = np.reshape(doseUHPP, dimension_flip)

        doseSOTA = os.path.join(patientFolder, "QihuiRyan", "doseQihuiRyan.bin")
        doseSOTA = np.fromfile(doseSOTA, dtype=np.float32)
        doseSOTA = np.reshape(doseSOTA, dimension_flip)

        doseSOTA_crossVal = os.path.join(patientFolder, "crossVal", "plan1", "dose.bin")
        doseSOTA_crossVal = np.fromfile(doseSOTA_crossVal, dtype=np.float32)
        doseSOTA_crossVal = np.reshape(doseSOTA_crossVal, dimension_flip)

        doseSOTA_UHPP = os.path.join(patientFolder, "QihuiRyan", "doseQihuiRyan_UHPP.bin")
        doseSOTA_UHPP = np.fromfile(doseSOTA_UHPP, dtype=np.float32)
        doseSOTA_UHPP = np.reshape(doseSOTA_UHPP, dimension_flip)

        doseSOTA_UHPP_polishing = os.path.join(patientFolder, "crossVal", "plan1_UHPP", "dose.bin")
        doseSOTA_UHPP_polishing = np.fromfile(doseSOTA_UHPP_polishing, dtype=np.float32)
        doseSOTA_UHPP_polishing = np.reshape(doseSOTA_UHPP_polishing, dimension_flip)

        bodyMask = os.path.join(patientFolder, "InputMask", "SKIN.bin")
        bodyMask = np.fromfile(bodyMask, dtype=np.uint8)
        bodyMask = np.reshape(bodyMask, dimension_flip) > 0
        notBody = np.logical_not(bodyMask)

        PTVMask = os.path.join(patientFolder, "InputMask", "ROI.bin")
        PTVMask = np.fromfile(PTVMask, dtype=np.uint8)
        PTVMask = np.reshape(PTVMask, dimension_flip) > 0

        density = os.path.join(patientFolder, "density_raw.bin")
        density = np.fromfile(density, dtype=np.uint16)
        density = np.reshape(density, dimension_flip)

        doseList = [doseUHPP, doseSOTA, doseSOTA_crossVal, doseSOTA_UHPP, doseSOTA_UHPP_polishing, doseRef]
        prescriptionDose = 20
        prescriptionLevel = 10  # D90
        R50Thresh = prescriptionDose * 0.5
        ptvVoxels = np.sum(PTVMask)
        RscoreList = []
        for doseArray in doseList:
            doseArray[notBody] = 0.0
            PTVDose = doseArray[PTVMask]
            threshLevel = np.percentile(PTVDose, prescriptionLevel)
            doseArray *= prescriptionDose / threshLevel
            doseGreaterThanDose50 = doseArray > R50Thresh
            Rscore = np.sum(doseGreaterThanDose50) / ptvVoxels
            RscoreList.append(Rscore)
        currentLine = "| {:03d} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(i+1, *RscoreList)
        content.append(currentLine)

        if False:
            patientImageFolder = os.path.join(figureFolder, patientName)
            if not os.path.isdir(patientImageFolder):
                os.mkdir(patientImageFolder)
            for j in range(dimension_flip[0]):
                densitySlice = density[j, :, :]
                PTVSlice = PTVMask[j, :, :]
                fig = plt.figure(figsize=(12, 4))
                gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
                PTVcontours = measure.find_contours(PTVSlice)
                for k in range(3):
                    current_block = fig.add_subplot(gs[0, k])
                    current_block.imshow(densitySlice, vmin=0, vmax=1200, cmap="gray")
                    doseSlice = doseList[k][j, :, :]
                    current_block.imshow(doseSlice, vmin=0, vmax=50, alpha=0.3, cmap="jet")
                    doseR50 = doseSlice > R50Thresh
                    doseR50Contours = measure.find_contours(doseR50)
                    for contour in PTVcontours:
                        plt.plot(contour[:, 1], contour[:, 0], color=colors[0], linewidth=1, linestyle="--")
                    for contour in doseR50Contours:
                        plt.plot(contour[:, 1], contour[:, 0], color=colors[1], linewidth=1, linestyle="--")
                fig.tight_layout()
                figureFile = os.path.join(patientImageFolder, "{:03d}.png".format(j+1))
                plt.savefig(figureFile)
                plt.close(fig)
                plt.clf()
                print(figureFile)
    content = "\n".join(content)
    print(content)


def comp_SOTA_UHPP():
    """
    Here we compare two groups:
    1. UHPP dose-loading matrices, SOTA optimization, SOTA fluence map polishing
    2. UHPP dose-loading matrices, SOTA optimization, UHPP fluence map polishing
    """
    factor = 0.5
    prescriptionLevel = 10
    currentFigureFolder = os.path.join(sourceFolder, "comp_SOTA_UHPP_new")
    if not os.path.isdir(currentFigureFolder):
        os.mkdir(currentFigureFolder)
    # content = ["UHPP dose, SOTA BOO, SOTA polishing | UHPP dose, SOTA BOO, UHPP polishing | Clinical |",
    #     "| - | - | - |"]
    content = ["UHPP dose, SOTA BOO, SOTA polishing | Clinical |",
        "| - | - |"]
    # for i in range(numPatients):
    for i in range(1):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)
        dimension = os.path.join(patientFolder, "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension_flip = np.flip(dimension)

        doseRef = os.path.join(patientFolder, "doseNorm.bin")
        doseRef = np.fromfile(doseRef, dtype=np.float32)
        doseRef = np.reshape(doseRef, dimension_flip)

        dose_SOTA_UHPP = os.path.join(patientFolder, "QihuiRyan", "doseQihuiRyan_UHPP.bin")
        dose_SOTA_UHPP = np.fromfile(dose_SOTA_UHPP, dtype=np.float32)
        dose_SOTA_UHPP = np.reshape(dose_SOTA_UHPP, dimension_flip)

        if False:
            dose_SOTA_UHPP_polish = os.path.join(patientFolder, "crossVal", "plan1_UHPP", "dose.bin")
            dose_SOTA_UHPP_polish = np.fromfile(dose_SOTA_UHPP_polish, dtype=np.float32)
            dose_SOTA_UHPP_polish = np.reshape(dose_SOTA_UHPP_polish, dimension_flip)

        bodyMask = os.path.join(patientFolder, "InputMask", "SKIN.bin")
        bodyMask = np.fromfile(bodyMask, dtype=np.uint8)
        bodyMask = np.reshape(bodyMask, dimension_flip) > 0
        notBody = np.logical_not(bodyMask)

        PTVMask = os.path.join(patientFolder, "InputMask", "ROI.bin")
        PTVMask = np.fromfile(PTVMask, dtype=np.uint8)
        PTVMask = np.reshape(PTVMask, dimension_flip) > 0

        density = os.path.join(patientFolder, "density_raw.bin")
        density = np.fromfile(density, dtype=np.uint16)
        density = np.reshape(density, dimension_flip)

        # doseList = [dose_SOTA_UHPP, dose_SOTA_UHPP_polish, doseRef]
        doseList = [dose_SOTA_UHPP, doseRef]
        prescriptionDose = 20
        R50Thresh = prescriptionDose * factor
        ptvVoxels = np.sum(PTVMask)
        RscoreList = []
        for doseArray in doseList:
            doseArray[notBody] = 0.0
            PTVDose = doseArray[PTVMask]
            threshLevel = np.percentile(PTVDose, prescriptionLevel)
            doseArray *= prescriptionDose / threshLevel
            doseGreaterThanDose50 = doseArray > R50Thresh
            Rscore = np.sum(doseGreaterThanDose50) / ptvVoxels
            RscoreList.append(Rscore)

        if False:
            titles = ["UHPP dose, SOTA BOO, SOTA polishing", "UHPP dose, SOTA BOO, UHPP polishing", "Clinical"]
            patientImageFolder = os.path.join(currentFigureFolder, patientName)
            if not os.path.isdir(patientImageFolder):
                os.mkdir(patientImageFolder)
            for j in range(dimension_flip[0]):
                densitySlice = density[j, :, :]
                PTVSlice = PTVMask[j, :, :]
                fig = plt.figure(figsize=(12, 4))
                gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
                PTVcontours = measure.find_contours(PTVSlice)
                for k in range(3):
                    current_block = fig.add_subplot(gs[0, k])
                    current_block.imshow(densitySlice, vmin=0, vmax=1200, cmap="gray")
                    doseSlice = doseList[k][j, :, :]
                    current_block.imshow(doseSlice, vmin=0, vmax=50, alpha=0.3, cmap="jet")
                    doseR50 = doseSlice > R50Thresh
                    doseR50Contours = measure.find_contours(doseR50)
                    for contour in PTVcontours:
                        current_block.plot(contour[:, 1], contour[:, 0], color=colors[0], linewidth=1, linestyle="--")
                    for contour in doseR50Contours:
                        current_block.plot(contour[:, 1], contour[:, 0], color=colors[1], linewidth=1, linestyle="--")
                    current_block.set_title(titles[k])
                fig.tight_layout()
                figureFile = os.path.join(patientImageFolder, "{:03d}.png".format(j+1))
                plt.savefig(figureFile)
                plt.close(fig)
                plt.clf()
                print(figureFile)
        # currentLine = "| {:03d} | {:.3f} | {:.3f} | {:.3f} |".format(i+1, *RscoreList)
        currentLine = "| {:03d} | {:.3f} | {:.3f} |".format(i+1, *RscoreList)
        content.append(currentLine)
    content = "\n".join(content)
    print(content)


def viewDose():
    for i in range(1):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)
        dimension = os.path.join(patientFolder, "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension_flip = np.flip(dimension)

        QihuiRyanFolder = os.path.join(sourceFolder, patientName, "QihuiRyan")
        doseFile = os.path.join(QihuiRyanFolder, "doseQihuiRyan_UHPP.bin")
        doseArray = np.fromfile(doseFile, dtype=np.float32)
        doseArray = np.reshape(doseArray, dimension_flip)
        
        if True:
            doseArray = np.transpose(doseArray, axes=[0, 2, 1])

        densityFile = os.path.join(patientFolder, "density_raw.bin")
        density = np.fromfile(densityFile, dtype=np.uint16)
        density = np.reshape(density, dimension_flip)  # (z, y, x)

        figureFolder = os.path.join(QihuiRyanFolder, "doseWashUHPP")
        if not os.path.isdir(figureFolder):
            os.mkdir(figureFolder)
        
        for j in range(dimension_flip[0]):
            densitySlice = density[j, :, :]
            doseSlice = doseArray[j, :, :]
            plt.imshow(densitySlice, cmap="gray", vmin=0, vmax=1200)
            plt.imshow(doseSlice, cmap="jet", vmin=0, vmax=50, alpha=0.3)
            figureFile = os.path.join(figureFolder, "{:03d}.png".format(j))
            plt.savefig(figureFile)
            plt.clf()
            print(figureFile)


def totalVoxelCountPatient002():
    prescription_dose = 20
    prescription_level = 10  # D90
    R50_level = prescription_dose * 0.5

    patientFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB/Patient002"
    dimension = os.path.join(patientFolder, "FastDose", "prep_output", "dimension.txt")
    with open(dimension, "r") as f:
        dimension = f.readline()
    dimension = eval(dimension.replace(" ", ", "))
    dimension_flip = np.flip(dimension)

    density = os.path.join(patientFolder, "density_raw.bin")
    density = np.fromfile(density, dtype=np.uint16)
    density = np.reshape(density, dimension_flip)

    PTV = os.path.join(patientFolder, "InputMask", "ROI.bin")
    PTV = np.fromfile(PTV, dtype=np.uint8)
    PTV = np.reshape(PTV, dimension_flip) > 0

    body = os.path.join(patientFolder, "InputMask", "SKIN.bin")
    body = np.fromfile(body, dtype=np.uint8)
    body = np.reshape(body, dimension_flip) > 0
    not_body = np.logical_not(body)

    doseRef = os.path.join(patientFolder, "doseNorm.bin")
    doseRef = np.fromfile(doseRef, dtype=np.float32)
    doseRef = np.reshape(doseRef, dimension_flip)
    doseRef[not_body] = 0.0
    PTVdoseRef = doseRef[PTV]
    threshRef = np.percentile(PTVdoseRef, prescription_level)
    doseRef *= prescription_dose / threshRef

    doseQihuiRyan = os.path.join(patientFolder, "QihuiRyan", "doseQihuiRyan.bin")
    doseQihuiRyan = np.fromfile(doseQihuiRyan, dtype=np.float32)
    doseQihuiRyan = np.reshape(doseQihuiRyan, dimension_flip)
    doseQihuiRyan[not_body] = 0.0
    PTVdoseQihuiRyan = doseQihuiRyan[PTV]
    threshQihuiRyan = np.percentile(PTVdoseQihuiRyan, prescription_level)
    doseQihuiRyan *= prescription_dose / threshQihuiRyan

    doseFastDose = os.path.join(patientFolder, "FastDose", "plan1", "dose.bin")
    doseFastDose = np.fromfile(doseFastDose, dtype=np.float32)
    doseFastDose = np.reshape(doseFastDose, dimension_flip)
    doseFastDose[not_body] = 0.0
    PTVdoseFastDose = doseFastDose[PTV]
    threshFastDose = np.percentile(PTVdoseFastDose, prescription_level)
    doseFastDose *= prescription_dose / threshFastDose

    figureFolder = os.path.join(patientFolder, "countVoxels")
    if not os.path.isdir(figureFolder):
        os.mkdir(figureFolder)
    
    doseList = [doseRef, doseFastDose, doseQihuiRyan]
    titleList = ["Clinical", "UHPP", "SOTA"]
    if False:
        for i in range(dimension_flip[0]):
            fig = plt.figure(figsize=(12, 4))
            gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
            density_slice = density[i, :, :]
            PTV_slice = PTV[i, :, :]
            PTV_contours = measure.find_contours(PTV_slice)
            for j in range(3):
                currentBlock = fig.add_subplot(gs[0, j])
                currentBlock.imshow(density_slice, cmap="gray", vmin=0, vmax=1500)
                for contour in PTV_contours:
                    currentBlock.plot(contour[:, 1], contour[:, 0], linewidth=1,
                        linestyle="--", color=colors[0])

                currentDoseSlice = doseList[j][i, :, :]
                currentDoseR50 = currentDoseSlice > R50_level
                currentBlock.imshow(currentDoseSlice, cmap="jet", vmin=0, vmax=50, alpha=0.3*currentDoseR50)
                currentDoseR50Contours = measure.find_contours(currentDoseR50)
                for contour in currentDoseR50Contours:
                    currentBlock.plot(contour[:, 1], contour[:, 0], linewidth=1,
                        linestyle="--", color=colors[1])
                nPixels = np.sum(currentDoseR50)
                currentBlock.set_title("{} R50 voxels: {}".format(titleList[j], nPixels))
            file = os.path.join(figureFolder, "{:03d}.png".format(i + 1))
            fig.tight_layout()
            plt.savefig(file)
            plt.close(fig)
            plt.clf()
            print(file)
    if False:
        content = ["| # Slice \ Group | Clinical | UHPP | SOTA |",
            "| - | - | - | - |"]
        total_count = [0, 0, 0]
        for i in range(dimension_flip[0]):
            localVoxelList = []
            for j in range(3):
                currentDoseSlice = doseList[j][i, :, :]
                currentDoseR50 = currentDoseSlice > R50_level
                currentVoxels = np.sum(currentDoseR50)
                localVoxelList.append(currentVoxels)
                total_count[j] += currentVoxels
            localLine = "| {} | {} | {} | {} |".format(i+1, *localVoxelList)
            content.append(localLine)
        content.append("| Total | {} | {} | {} |".format(*total_count))
        content = "\n".join(content)
        print(content)

    if True:
        content = ["Slice \ Group,Clinical,UHPP,SOTA"]
        total_count = [0, 0, 0]
        for i in range(dimension_flip[0]):
            localVoxelList = []
            for j in range(3):
                currentDoseSlice = doseList[j][i, :, :]
                currentDoseR50 = currentDoseSlice > R50_level
                currentVoxels = np.sum(currentDoseR50)
                localVoxelList.append(currentVoxels)
                total_count[j] += currentVoxels
            localLine = "{},{},{},{}".format(i+1, *localVoxelList)
            content.append(localLine)
        content.append("Total,{},{},{}".format(*total_count))
        PTVVoxels = np.sum(PTV)
        content.append("PTV voxels,{},{},{}".format(PTVVoxels, PTVVoxels, PTVVoxels))
        R50 = np.array(total_count) / PTVVoxels
        content.append("R50,{},{},{}".format(*R50))
        content = "\n".join(content)
        csvFile = os.path.join(patientFolder, "countVoxels.csv")
        with open(csvFile, "w") as f:
            f.write(content)



if __name__ == "__main__":
    # main()
    # cross_validation_beamlist_gen()
    # crossValR50Calc()
    # comp_SOTA_UHPP()
    viewDose()
    # totalVoxelCountPatient002()