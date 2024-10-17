import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from skimage import measure
from scipy.sparse import csr_array

sourceFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB"
beamSanityFolder = os.path.join(sourceFolder, "BeamSanityCheckView")
if not os.path.isdir(beamSanityFolder):
    os.mkdir(beamSanityFolder)
numPatients = 5

def showBeamlet():
    """
    To reflect the trajectory of a beam, in this script, we show the central beamlet
    of each beam in the dose colorwash
    """
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)
        beamIdxFastDose = os.path.join(patientFolder, "FastDose", "plan1", "metadata.txt")
        beamIdxFastDose = extractBeamIdxFastDose(beamIdxFastDose)
        beamIdxQihuiRyan = os.path.join(patientFolder, "QihuiRyan", "selected_angles.csv")
        beamIdxQihuiRyan = extractBeamIdxQihuiRyan(beamIdxQihuiRyan)

        dimension = os.path.join(patientFolder, "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = eval(dimension.replace(" ", ", "))
        dimension_flip = np.flip(dimension)
        # nVoxels = dimension[0] * dimension[1] * dimension[2]
        nVoxels = np.prod(dimension)

        # get the union of the two indices
        beamIdxSet = []
        for a in beamIdxFastDose:
            if a not in beamIdxSet:
                beamIdxSet.append(a)
        for a in beamIdxQihuiRyan:
            if a not in beamIdxSet:
                beamIdxSet.append(a)
        beamIdxSet.sort()

        fluenceDim = (20, 20)
        fluenceMapFile1 = os.path.join(patientFolder, "FastDose", "doseMat1", "doseMatFolder", "fluenceMap.bin")
        fluenceMapFile2 = os.path.join(patientFolder, "FastDose", "doseMat2", "doseMatFolder", "fluenceMap.bin")
        centralIndexMap, nBeams1 = getFluenceMap(fluenceMapFile1, fluenceDim, beamIdxSet, 0)
        centralIdxMap2, nBeams2 = getFluenceMap(fluenceMapFile2, fluenceDim, beamIdxSet, nBeams1)
        centralIndexMap.update(centralIdxMap2)

        M1_folder = os.path.join(patientFolder, "FastDose", "doseMat1", "doseMatFolder")
        M2_folder = os.path.join(patientFolder, "FastDose", "doseMat2", "doseMatFolder")
        M_list, M1_number = loadDoseMat(M1_folder, nVoxels, beamIdxSet, 0)
        M2_list, M2_number = loadDoseMat(M2_folder, nVoxels, beamIdxSet, M1_number)
        M_list.update(M2_list)

        FastDoseBeamList = [M_list[a] for a in beamIdxFastDose]
        FastDoseFluenceCenterIdx = [centralIndexMap[a] for a in beamIdxFastDose]
        FastDoseCentralBeamletDose = sumCentralBeamlet(FastDoseBeamList, FastDoseFluenceCenterIdx, dimension_flip)
        QihuiRyanBeamList = [M_list[b] for b in beamIdxQihuiRyan]
        QihuiRYanFluenceCenterIdx = [centralIndexMap[a] for a in beamIdxQihuiRyan]
        QihuiRyanCentralBeamletDose = sumCentralBeamlet(QihuiRyanBeamList, QihuiRYanFluenceCenterIdx, dimension_flip)
        
        resultFolder = os.path.join(patientFolder, "centralBeamletView")
        if not os.path.isdir(resultFolder):
            os.mkdir(resultFolder)
        file = os.path.join(resultFolder, "FastDoseCentralBeamlet.npy")
        np.save(file, FastDoseCentralBeamletDose)
        print(file)
        file = os.path.join(resultFolder, "QihuiRyanCentralBeamlet.npy")
        np.save(file, QihuiRyanCentralBeamletDose)
        print(file, "\n")

    
def sumCentralBeamlet(beamList, centralIdx, dimension):
    assert (nBeams:=len(beamList)) == len(centralIdx)
    nVoxels = np.prod(dimension)
    doseArrays = np.zeros(nVoxels)
    for i in range(nBeams):
        current_dose_loading_matrix = beamList[i]
        current_central_fluence_idx = centralIdx[i]
        nBeamletsLocal = current_dose_loading_matrix.shape[0]
        selectionArray = np.zeros(nBeamletsLocal, dtype=int)
        selectionArray[current_central_fluence_idx] = 1
        current_mat = selectionArray @ current_dose_loading_matrix
        doseArrays += current_mat
    doseArrays = np.reshape(doseArrays, dimension)
    return doseArrays


def extractBeamIdxFastDose(file):
    with open(file, "r") as f:
        lines = f.readlines()
    relevant = lines[3]
    relevant = relevant.replace("  ", ", ")
    relevant = sorted(eval(relevant))
    return relevant


def extractBeamIdxQihuiRyan(file):
    with open(file, "r") as f:
        lines = f.readlines()
    lines = lines[1:-1]  # remove the title and the last line
    result = []
    for line in lines:
        line = eval(line.split(",")[0])
        result.append(line - 1)  # convert from 1-based index to 0-based index
    result.sort()
    return tuple(result)


def loadDoseMat(doseMatFolder, nVoxels, beamIdxList, offset):
    offsetsBufferFile = os.path.join(doseMatFolder, "offsetsBuffer.bin")
    columnsBufferFile = os.path.join(doseMatFolder, "columnsBuffer.bin")
    valuesBufferFile = os.path.join(doseMatFolder, "valuesBuffer.bin")
    numRowsPerMatFile = os.path.join(doseMatFolder, "numRowsPerMat.bin")
    offsetsBuffer = np.fromfile(offsetsBufferFile, dtype=np.uint64).astype(int)
    print("Finished loading the offsets buffer")
    columnsBuffer = np.fromfile(columnsBufferFile, dtype=np.uint64).astype(int)
    print("Finished loading the columns buffer")
    valuesBuffer = np.fromfile(valuesBufferFile, dtype=np.float32)
    print("Finished loading the values buffer")
    numRowsPerMat = np.fromfile(numRowsPerMatFile, dtype=np.uint64).astype(int)
    print("Finished loading the number of rows per mat buffer")

    offsetsBufferIdx = 0
    columnsBufferIdx = 0
    nMatrices = numRowsPerMat.size
    matrices = {}
    for i in range(nMatrices):
        currentNumRowsPerMat = numRowsPerMat[i]
        currentOffsetsSize = currentNumRowsPerMat + 1
        currentOffsetsBuffer = offsetsBuffer[offsetsBufferIdx: offsetsBufferIdx + currentOffsetsSize]

        currentNumElements = currentOffsetsBuffer[-1]
        currentColumnsBuffer = columnsBuffer[columnsBufferIdx: columnsBufferIdx + currentNumElements]
        currentValuesBuffer = valuesBuffer[columnsBufferIdx: columnsBufferIdx + currentNumElements]

        offsetsBufferIdx += currentOffsetsSize
        offsetsBufferIdx = offsetsBufferIdx
        columnsBufferIdx += currentNumElements
        columnsBufferIdx = int(columnsBufferIdx)

        if i + offset not in beamIdxList:
            continue

        # then convert the csr offsetsBuffer into individual row indices
        rowIdx = np.zeros(currentNumElements)
        for j in range(currentNumRowsPerMat):
            beginIdx = currentOffsetsBuffer[j]
            endIdx = currentOffsetsBuffer[j+1]
            rowIdx[beginIdx: endIdx] = j
        
        current_csr_array = csr_array((currentValuesBuffer, (rowIdx, currentColumnsBuffer)),
            shape=(currentNumRowsPerMat, nVoxels))
        matrices[i + offset] = current_csr_array
        print("Matrix idx: {}".format(i + offset))
    return matrices, nMatrices


def getFluenceMap(fluenceMapFile, fluenceDim, beamIdxSet, offset):
    fluenceCenterIdx_0, fluenceCenterIdx_1 = fluenceDim
    fluenceCenterIdx_0 = int(np.floor(fluenceCenterIdx_0 / 2))
    fluenceCenterIdx_1 = int(np.floor(fluenceCenterIdx_1 / 2))
    fluenceCenterIdx = fluenceCenterIdx_1 + fluenceCenterIdx_0 * fluenceDim[0]

    fluenceMapElements = np.prod(fluenceDim)
    fluenceMapArray = np.fromfile(fluenceMapFile, dtype=np.uint8)
    assert fluenceMapArray.size % fluenceMapElements == 0
    nBeams = int(fluenceMapArray.size / fluenceMapElements)

    result = {}
    for i in range(nBeams):
        if i + offset not in beamIdxSet:
            continue
        currentSlice_begin = i * fluenceMapElements
        currentSlice_end = currentSlice_begin + fluenceMapElements
        currentSlice = fluenceMapArray[currentSlice_begin: currentSlice_end]
        assert currentSlice[fluenceCenterIdx] > 1e-4  # ensure that the center beamlet is active
        nBeamlets_prior = currentSlice[:fluenceCenterIdx]
        currentIdx = np.sum(nBeamlets_prior)
        result[i + offset] = currentIdx
    return result, nBeams


def drawBeamTrajectory():
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        patientFolder = os.path.join(sourceFolder, patientName)
        dimension = os.path.join(patientFolder, "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = eval(dimension.replace(" ", ", "))
        dimension_flip = np.flip(dimension)

        densityFile = os.path.join(patientFolder, "density_raw.bin")
        density = np.fromfile(densityFile, dtype=np.uint16)
        density = np.reshape(density, dimension_flip)

        centralBeamViewFolder = os.path.join(patientFolder, "centralBeamletView")
        FastDoseProfile = os.path.join(centralBeamViewFolder, "FastDoseCentralBeamlet.npy")
        FastDoseProfile = np.load(FastDoseProfile)
        QihuiRyanProfile = os.path.join(centralBeamViewFolder, "QihuiRyanCentralBeamlet.npy")
        QihuiRyanProfile = np.load(QihuiRyanProfile)
        assert np.all(FastDoseProfile.shape == dimension_flip) \
            and np.all(QihuiRyanProfile.shape == dimension_flip)
        
        currentFigureFolder = os.path.join(patientFolder, "BeamAngleView")
        if not os.path.isdir(currentFigureFolder):
            os.mkdir(currentFigureFolder)
        
        doseScale = max(np.max(FastDoseProfile), np.max(QihuiRyanProfile)) * 0.3
        for j in range(dimension_flip[0]):
            densitySlice = density[j, :, :]
            FastDoseProfileSlice = FastDoseProfile[j, :, :]
            QihuiRyanProfileSlice = QihuiRyanProfile[j, :, :]
            fig  = plt.figure(figsize=(8, 4))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

            sliceList = [FastDoseProfileSlice, QihuiRyanProfileSlice]
            titleList = ["UHPP", "SOTA"]
            for k in range(2):
                current_block = fig.add_subplot(gs[0, k])
                current_block.imshow(densitySlice, cmap="gray", vmin=0, vmax=1200)
                current_block.imshow(sliceList[k], cmap="jet", vmin=0, vmax=doseScale, alpha=0.8 * (sliceList[k] > 1e-4))
                current_block.set_title(titleList[k])
            fig.tight_layout()
            figureFile = os.path.join(currentFigureFolder, "{:03d}.png".format(j+1))
            plt.savefig(figureFile)
            plt.close(fig)
            plt.clf()
            print(figureFile)



if __name__ == "__main__":
    # showBeamlet()
    drawBeamTrajectory()