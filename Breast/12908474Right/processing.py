import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rt_utils import RTStructBuilder
from skimage import measure, transform
import json
import h5py
from scipy.interpolate import RegularGridInterpolator

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
colors[7] = colors[-1]
rootFolder = "/data/qifan/projects/FastDoseWorkplace/Breast"
patientName = "12908474Right"
targetResolution = np.array((2.5, 2.5, 2.5))  # mm


def initialProcessing():
    patientFolder = os.path.join(rootFolder, patientName)
    dicomFolder = os.path.join(patientFolder, "dicom")
    dicomFiles = os.listdir(dicomFolder)
    rtFile = None
    ctData = []
    RescaleSlope = None
    RescaleIntercept = None
    base = None
    HUmin = -1000
    for file in dicomFiles:
        file = os.path.join(dicomFolder, file)
        dataset = pydicom.dcmread(file)
        if dataset.Modality == "RTSTRUCT":
            rtFile = file
            continue
        if dataset.Modality == "CT":
            InstanceNumber = dataset.InstanceNumber
            InstanceNumber = int(InstanceNumber)
            if RescaleSlope is None or  RescaleIntercept is None:
                RescaleSlope = dataset.RescaleSlope
                RescaleIntercept = dataset.RescaleIntercept
            pixel_array = dataset.pixel_array
            pixel_array = pixel_array * RescaleSlope + RescaleIntercept
            pixel_array -= HUmin
            pixel_array[pixel_array < 0] = 0
            ctData.append((InstanceNumber, pixel_array))
    ctData.sort(key=lambda a: a[0])
    
    assert rtFile is not None, "rtFile not found"
    RTStruct = RTStructBuilder.create_from(
        dicom_series_path=dicomFolder, rt_struct_path=rtFile)
    structureNames = RTStruct.get_roi_names()
    # for a in names_useful:
    #     assert a in structureNames, "The structure {} not included in the structure list".format(a)
    # structureNames = names_useful
    print(structureNames)
    return

    # There are so many structures, I'd like to take a look, to see which are relevant.
    structures = {}
    count = 0
    limit = 10
    for name in structureNames:
        try:
            mask = RTStruct.get_roi_mask_by_name(name)
        except:
            print("Failed to load structure {}".format(name))
            continue
        mask = np.transpose(mask, axes=(2, 0, 1))
        mask = np.flip(mask, axis=0)
        structures[name] = mask
        print("Loading structure {}".format(name))
        count += 1
        # if  count == limit:
        #     break

    viewFolder = os.path.join(patientFolder, "dicomView")
    if not os.path.isdir(viewFolder):
        os.mkdir(viewFolder)
    for i in range(len(ctData)):
        slice = ctData[i][1]
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.imshow(slice, cmap="gray", vmin=0, vmax=2000)
        for j, entry in enumerate(structures.items()):
            name, array = entry
            color = colors[j]
            slice = array[i, :, :]
            contours = measure.find_contours(slice)
            initial = True
            for contour in contours:
                if initial:
                    plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                    initial = False
                else:
                    plt.plot(contour[:, 1], contour[:, 0], color=color)
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
        plt.tight_layout()
        file = os.path.join(viewFolder, "{:03d}.png".format(i))
        plt.savefig(file, bbox_inches="tight")
        plt.clf()
        plt.close(fig)
        print(file)


if __name__ == "__main__":
    initialProcessing()