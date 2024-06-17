import os
import numpy as np
import h5py
from scipy.signal import convolve

def cropPatient2RingStructure():
    """
    This function crops out the PTV structure for patient 2
    """
    prepFolder = "/data/qifan/projects/FastDoseWorkplace/TCIAAdd/002/FastDose/prep_output"
    roiFile = os.path.join(prepFolder, "roi_list.h5")
    dimension = os.path.join(prepFolder, "dimension.txt")
    with open(dimension, 'r') as f:
        dimension = f.readline()
    dimension = dimension.replace(" ", ", ")
    dimension = eval(dimension)

    dataset = h5py.File(roiFile, "r")
    names = list(dataset.keys())
    assert "RingStructure" in names and "Trachea" in names
    RingStructMask = getMask("RingStructure", dataset)
    TracheaMask = getMask("Trachea", dataset)

    convolveKernel = np.ones((2, 2, 2))
    TracheaMask = convolve(TracheaMask, convolveKernel, mode="same")
    TracheaMask = TracheaMask > 4
    print(np.sum(RingStructMask), np.sum(TracheaMask))
    RingStructMask = np.logical_and(RingStructMask, np.logical_not(TracheaMask))
    RingStructMask = RingStructMask.astype(np.uint8)
    print(np.sum(RingStructMask))

    file = "/data/qifan/projects/FastDoseWorkplace/TCIAAdd/002/PlanMask/RingStructModify.bin"
    RingStructMask.tofile(file)


def getMask(name, dataset):
    struct = dataset[name]
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
    return struct_mask


def cropPatient9RingStructure():
    """
    We found out that the ring structure significantly compromises the PTV56 region
    So we decide to split the ring structure into upper and lower components, each assigned different weights
    """
    patient = "009"
    prepFolder = "/data/qifan/projects/FastDoseWorkplace/TCIAAdd/{}/FastDose/prep_output".format(patient)
    roiFile = os.path.join(prepFolder, "roi_list.h5")
    dimension = os.path.join(prepFolder, "dimension.txt")
    with open(dimension, 'r') as f:
        dimension = f.readline()
    dimension = dimension.replace(" ", ", ")
    dimension = eval(dimension)

    dataset = h5py.File(roiFile, "r")
    names = list(dataset.keys())
    assert "RingStructure" in names
    RingStructMask = getMask("RingStructure", dataset).astype(np.uint8)

    septa1 = 68
    septa2 = 106

    MaskFolder = "/data/qifan/projects/FastDoseWorkplace/TCIAAdd/{}/PlanMask".format(patient)
    RingStructUpper = RingStructMask.copy()
    RingStructUpper[:septa2, :, :] = 0
    upperFile = os.path.join(MaskFolder, "RingStructUpper.bin")
    RingStructUpper.tofile(upperFile)

    RingStructMiddle = RingStructMask.copy()
    RingStructMiddle[:septa1, :, :] = 0
    RingStructMiddle[septa2:, :, :] = 0
    middleFile = os.path.join(MaskFolder, "RingStructMiddle.bin")
    RingStructMiddle.tofile(middleFile)

    RingStructLower = RingStructMask.copy()
    RingStructLower[septa1:, :, :] = 0
    lowerFile = os.path.join(MaskFolder, "RingStructLower.bin")
    RingStructLower.tofile(lowerFile)


if __name__ == "__main__":
    # cropPatient2RingStructure()
    cropPatient9RingStructure()