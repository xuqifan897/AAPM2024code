import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import scipy.io

resultFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas"
numPatients = 5
ManuFiguresFolder = "/data/qifan/projects/AAPM2024/manufigures"

StructureList = []
colorMap = {}
exclude = {"SKIN"}

def StructsInit():
    global StructureList
    global colorMap
    for i in range(numPatients):
        patientFolder = os.path.join(resultFolder, "Patient{:03d}".format(i+1))
        InputMaskFolder = os.path.join(patientFolder, "InputMask")
        structures = os.listdir(InputMaskFolder)
        structures = [a.split(".")[0] for a in structures]
        for a in structures:
            if a not in StructureList:
                StructureList.append(a)
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
    for i in range(len(StructureList)):
        colorMap[StructureList[i]] = colors[i]


def DVH_plot():
    rowSize = 3
    colSize = int(np.ceil(numPatients / rowSize))
    fig, axes = plt.subplots(colSize, rowSize, figsize=(15, 8))
    for i in range(numPatients):
        patientFolder = os.path.join(resultFolder, "Patient{:03d}".format(i + 1))
        FastDoseFolder = os.path.join(patientFolder, "FastDose")
        dimensionFile = os.path.join(FastDoseFolder, "prep_output", "dimension.txt")
        with open(dimensionFile, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension = np.flip(dimension)

        MaskFolder = os.path.join(patientFolder, "InputMask")
        StructuresLocal = os.listdir(MaskFolder)
        StructuresLocal = [a.split(".")[0] for a in StructuresLocal]
        StructuresLocal = [a for a in StructuresLocal if a not in exclude]
        maskDict = {}
        for struct in StructuresLocal:
            maskFile = os.path.join(MaskFolder, "{}.bin".format(struct))
            maskArray = np.fromfile(maskFile, dtype=np.uint8)
            maskArray = np.reshape(maskArray, dimension)
            maskDict[struct] = maskArray
        
        DoseMatOursFile = os.path.join(FastDoseFolder, "plan1", "dose.bin")
        DoseMatOurs = np.fromfile(DoseMatOursFile, dtype=np.float32)
        DoseMatOurs = np.reshape(DoseMatOurs, dimension)

        DoseMatRefFile = os.path.join(patientFolder, "QihuiRyan", "binaryDose*.bin")
        DoseMatRefFile = glob.glob(DoseMatRefFile)[0]
        DoseMatRef = np.fromfile(DoseMatRefFile, dtype=np.float32)
        DoseMatRef = np.reshape(DoseMatRef, dimension)
        DoseMatRef = np.transpose(DoseMatRef, axes=(0, 2, 1))

        rowIdx = i % rowSize
        colIdx = i // rowSize
        assert colIdx < colSize, "Figure index ({}, {}) error.".format(rowIdx, colIdx)
        ax = axes[colIdx, rowIdx]
        for name, mask in maskDict.items():
            color = colorMap[name]
            mask = mask.astype(bool)
            StructDoseOurs = DoseMatOurs[mask]
            StructDoseOurs = np.sort(StructDoseOurs)
            StructDoseOurs = np.insert(StructDoseOurs, 0, 0.0)

            StructDoseRef = DoseMatRef[mask]
            StructDoseRef = np.sort(StructDoseRef)
            StructDoseRef = np.insert(StructDoseRef, 0, 0.0)

            y_axis = np.linspace(100, 0, np.sum(mask)+1)
            ax.plot(StructDoseOurs, y_axis, color=color, linewidth=2.0)
            ax.plot(StructDoseRef, y_axis, color=color, linewidth=2.0, linestyle="--")
            ax.set_xlim(0, 23)
            print(name)
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_title("Patient {:03d}".format(i+1), fontsize=18)
        print()
    fig.delaxes(axes[1, 2])

    # prepare legend
    legend_ax = fig.add_subplot(2, 3, 6)
    legend_ax.axis("off")
    handles = []
    labels = []
    for name, color in colorMap.items():
        handleEntry = plt.Line2D([0], [0], color=color, lw=2)
        handles.append(handleEntry)
        labels.append(name)
    legend_ax.legend(handles, labels, loc="center", ncol=2, fontsize=16)

    if False:
        # prepare global xlabel and ylabel
        fig.subplots_adjust(left=0.2, right=1.0, top=1.0, bottom=0.5)
        fig.text(0.5, 0.04, "Dose (Gy)", ha='center', va='center', fontsize=16)
        fig.text(0.00, 0.5, "Fractional Volume (%)", ha='center', va='center',
            rotation='vertical', fontsize=16)
    
    plt.tight_layout()
    figureFile = os.path.join(ManuFiguresFolder, "FastDosePancreas.png")
    plt.savefig(figureFile)
    figureFile = os.path.join(ManuFiguresFolder, "FastDosePancreas.eps")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()
    print(figureFile, "\n")


def GridFigure():
    # This example is adapted from an example provided by ChatGPT
    rowSize = 4
    colSize = 3
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(colSize, rowSize, width_ratios=[0.2, 5, 5, 5], height_ratios=[4, 4, 0.2])
    
    # Create the common ylabel
    ylabel_block = fig.add_subplot(gs[:-1, 0])
    ylabel_block.text(0.9, 0.5, "Fractional Volume (%)", ha="center", va="center",
        rotation="vertical", fontsize=20)
    ylabel_block.axis("off")

    # Create the common xlabel
    xlabel_block = fig.add_subplot(gs[-1, 1:])
    xlabel_block.text(0.5, 0.9, "Dose (Gy)", ha="center", va="center", fontsize=20)
    xlabel_block.axis("off")

    # Create the DVH plots
    localRowSize = rowSize - 1
    localColSize = colSize - 1
    for i in range(numPatients):
        patientFolder = os.path.join(resultFolder, "Patient{:03d}".format(i+1))
        FastDoseFolder = os.path.join(patientFolder, "FastDose")
        dimensionFile = os.path.join(FastDoseFolder, "prep_output", "dimension.txt")
        with open(dimensionFile, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension = np.flip(dimension)

        MaskFolder = os.path.join(patientFolder, "InputMask")
        StructuresLocal = os.listdir(MaskFolder)
        StructuresLocal = [a.split(".")[0] for a in StructuresLocal]
        StructuresLocal = [a for a in StructuresLocal if a not in exclude]
        maskDict = {}
        for struct in StructuresLocal:
            maskFile = os.path.join(MaskFolder, "{}.bin".format(struct))
            maskArray = np.fromfile(maskFile, dtype=np.uint8)
            maskArray = np.reshape(maskArray, dimension)
            maskDict[struct] = maskArray
        
        DoseMatOursFile = os.path.join(FastDoseFolder, "plan1", "dose.bin")
        DoseMatOurs = np.fromfile(DoseMatOursFile, dtype=np.float32)
        DoseMatOurs = np.reshape(DoseMatOurs, dimension)

        DoseMatRefFile = os.path.join(patientFolder, "QihuiRyan", "binaryDose*.bin")
        DoseMatRefFile = glob.glob(DoseMatRefFile)[0]
        DoseMatRef = np.fromfile(DoseMatRefFile, dtype=np.float32)
        DoseMatRef = np.reshape(DoseMatRef, dimension)
        DoseMatRef = np.transpose(DoseMatRef, axes=(0, 2, 1))

        rowIdx = i % localRowSize + 1
        colIdx = i // localRowSize
        assert colIdx < colSize, "Figure index ({}, {}) error.".format(rowIdx, colIdx)
        block = fig.add_subplot(gs[colIdx, rowIdx])

        for name, mask in maskDict.items():
            color = colorMap[name]
            mask = mask.astype(bool)
            StructDoseOurs = DoseMatOurs[mask]
            StructDoseOurs = np.sort(StructDoseOurs)
            StructDoseOurs = np.insert(StructDoseOurs, 0, 0.0)

            StructDoseRef = DoseMatRef[mask]
            StructDoseRef = np.sort(StructDoseRef)
            StructDoseRef = np.insert(StructDoseRef, 0, 0.0)

            y_axis = np.linspace(100, 0, np.sum(mask)+1)
            block.plot(StructDoseOurs, y_axis, color=color, linewidth=2.0)
            block.plot(StructDoseRef, y_axis, color=color, linewidth=2.0, linestyle="--")
            block.set_xlim(0, 23)
            print(name)
        block.tick_params(axis="x", labelsize=16)
        block.tick_params(axis="y", labelsize=16)
        block.set_title("Patient {:03d}".format(i+1), fontsize=20)
        print()

    # prepare legend
    legendBlock = fig.add_subplot(gs[colSize-2, rowSize-1])
    legendBlock.axis("off")
    handles = []
    labels = []
    for name, color in colorMap.items():
        handleEntry = plt.Line2D([0], [0], color=color, lw=2)
        handles.append(handleEntry)
        labels.append(name)
    legendBlock.legend(handles, labels, loc="center", ncol=2, fontsize=16)
    plt.tight_layout()

    figureFile = os.path.join(ManuFiguresFolder, "FastDosePancreas.png")
    plt.savefig(figureFile)
    figureFile = os.path.join(ManuFiguresFolder, "FastDosePancreas.eps")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()


def DoseCalcTimeTableGen():
    """
    This function generates the time consumption comparison between the baseline method and ours
    """
    def ReadTimeFromFile(file: str):
        assert os.path.isfile(file), "The file {} doesn't exist.".format(file)
        with open(file, "r") as f:
            lines = f.readlines()
        pattern = "Dose calculation time:"
        selected  = [a for a in lines if pattern in a]
        assert len(selected) == 1
        selected = selected[0]
        selected = selected.split(" ")[-2]
        result = eval(selected)
        return result
    Content = "| Case \ Group | Ours | Baseline | Speedup |\n|-|-|-|-|"
    speedupAvg = 0
    for i in range(numPatients):
        PatientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(resultFolder, PatientName)
        FastDoseFolder = os.path.join(patientFolder, "FastDose")
        logFile1 = os.path.join(FastDoseFolder, "dosecalc1.log")
        logFile2 = os.path.join(FastDoseFolder, "dosecalc2.log")
        time1 = ReadTimeFromFile(logFile1)
        time2 = ReadTimeFromFile(logFile2)
        TimeFastDose = time1 + time2
        minutesFastDose = int(TimeFastDose / 60)
        secondsFastDose = TimeFastDose - minutesFastDose * 60
        if minutesFastDose > 0:
            entryFastDose = "{}:{:.2f}".format(minutesFastDose, secondsFastDose)
        else:
            entryFastDose = "{:.2f}".format(secondsFastDose)
        
        BaselineLogFile = os.path.join(patientFolder, "QihuiRyan", "dosecalc-beamlet.log")
        with open(BaselineLogFile, "r") as f:
            lines = f.readlines()
        useful = [a for a in lines if "real" in a]
        assert len(useful) == 1, "More than one line or no line contains the pattern 'real'"
        useful = useful[0]
        useful = useful.split("\t")[1]
        useful = useful.split("m")
        minutesBaseline = eval(useful[0])
        secondsBaseline = useful[1]
        secondsBaseline = eval(secondsBaseline.split("s")[0])
        timeBaseline = minutesBaseline * 60 + secondsBaseline
        speedup = timeBaseline / TimeFastDose
        speedupAvg += speedup
        entryBaseline = "{}:{:.2f}".format(minutesBaseline, secondsBaseline)
        
        newLine = "| {} | {} | {} | {:.2f} |".format(PatientName, entryFastDose, entryBaseline, speedup)
        Content = Content + "\n" + newLine
    print(Content)
    speedupAvg /= numPatients
    print("Average speedup: {}".format(speedupAvg))


def OptimizeTimeTableGen():
    timeFastDose = []
    timeBaseline = []
    BeamWeightList = [1000, 2000, 2000, 2000, 2000]
    for i in range(numPatients):
        PatientName = "Patient{:03d}".format(i+1)
        OptLogFile = os.path.join(resultFolder, PatientName, "FastDose", "optimize.log")
        with open(OptLogFile, "r") as f:
            lines = f.readlines()
        pattern = "Optimization iterations"
        lines = [a for a in lines if pattern in a]
        assert len(lines) == 2, "The number of lines is not 2"
        line = lines[0]
        line = line.split(" ")
        entry = line[-2]
        entry = eval(entry)
        timeFastDose.append(entry)

        BeamWeight = BeamWeightList[i]
        BOOResultFile = os.path.join(resultFolder, PatientName,
            "QihuiRyan", "BOOResultBW{}.mat".format(BeamWeight))
        assert os.path.isfile(BOOResultFile), "The file '{}' doesn't exist.".format(BOOResultFile)
        mat_content = scipy.io.loadmat(BOOResultFile)
        BOOResult = mat_content["BOOresult"]
        BOOResult = BOOResult[0][0]
        BeamSelectionTime = BOOResult[7][0][0]
        timeBaseline.append(BeamSelectionTime)
    
    def seconds2minutes(seconds: float):
        minutes = int(seconds / 60)
        seconds -= 60 * minutes
        result = "{}:{:.2f}".format(minutes, seconds)
        return result

    speedupAvg = 0
    content = "| Case \ Group | Ours | Baseline | Speedup |\n|-|-|-|-|"
    for i in range(numPatients):
        PatientName = "Patient{:03d}".format(i + 1)
        localTimeFastDose = timeFastDose[i]
        entryFastDose = seconds2minutes(localTimeFastDose)

        localTimeBaseline = timeBaseline[i]
        entryBaseline = seconds2minutes(localTimeBaseline)
        speedup = localTimeBaseline / localTimeFastDose
        speedupAvg += speedup
        newLine = "| {} | {} | {} | {:.2f} |".format(PatientName,
            entryFastDose, entryBaseline, speedup)
        content = content + "\n" + newLine
    print(content)
    speedupAvg /= numPatients
    print("Average speedup: {:.2f}".format(speedupAvg))


if __name__ == "__main__":
    StructsInit()
    # DVH_plot()
    # GridFigure()
    # DoseCalcTimeTableGen()
    OptimizeTimeTableGen()