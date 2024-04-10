import os
import nibabel as nib
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors  as mcolors
from skimage import measure


def test_nib():
    """
    This script tests the functionality of nibabel
    """
    CTFile = "/data/qifan/projects/FastDoseWorkplace/HN_Lu/CT/HN_ct_001.nii.gz"
    DoseFile = "/data/qifan/projects/FastDoseWorkplace/HN_Lu/DOSE/HN_dose_001.nii.gz"
    PTVFile = "/data/qifan/projects/FastDoseWorkplace/HN_Lu/PTV/HN_ptv_001.nii.gz"
    RTFile = "/data/qifan/projects/FastDoseWorkplace/HN_Lu/RTSTRUCT/HN_rt_001.nii.gz"
    img = nib.load(RTFile)
    header = img.header
    data = img.get_fdata()
    # print(np.max(data))
    print(header)


def dataView():
    PTVStructs = [
        ["PTV54", "PTV60"],
        ["PTV54", "PTV60", "PTV70"],
        ["PTV54", "PTV60"],
        ["PTV54", "PTV60", "PTV70"],
        ["PTV retroph 52.5", "PTV 52.5 L neck", "PTV 52.5 R neck",
            "PTV 56 L neck", "PTV 70 primary", "PTV 70 L neck", "PTV 70 L NK Low"],
        ["PTV54", "PTV60", "PTV70"]
    ]
    OAR_LUT = {1: "BODY", 2: "Cord", 3: "Rparotid", 4: "Lparotid", 5: "Optic nerve R",
               6: "Optic nerve L", 7: "Cochlea R", 8: "Cochlea L", 9: "Mandible",
               10: "Brainstem", 11: "Chiasm", 12: "Larynx", 13: "Esophagus"}
    PatNames = ["001", "002", "003", "004", "005", "007"]
    CTFolder = "/data/qifan/projects/FastDoseWorkplace/HN_Lu/CT"
    DoseFolder = "/data/qifan/projects/FastDoseWorkplace/HN_Lu/DOSE"
    PTVFolder = "/data/qifan/projects/FastDoseWorkplace/HN_Lu/PTV"
    RTFolder = "/data/qifan/projects/FastDoseWorkplace/HN_Lu/RTSTRUCT"
    targetFolder = "/data/qifan/projects/FastDoseWorkplace/HN_Lu/view"
    numPatients = 6
    if not os.path.isdir(targetFolder):
        os.mkdir(targetFolder)

    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())

    for i in range(numPatients):
        patientName = PatNames[i]
        CTFile = os.path.join(CTFolder, "HN_ct_{}.nii.gz".format(patientName))
        DoseFile = os.path.join(DoseFolder, "HN_dose_{}.nii.gz".format(patientName))
        PTVFile = os.path.join(PTVFolder, "HN_ptv_{}.nii.gz".format(patientName))
        RTFile = os.path.join(RTFolder, "HN_rt_{}.nii.gz".format(patientName))

        CTMat = nib.load(CTFile).get_fdata()
        DoseMat = nib.load(DoseFile).get_fdata()
        PTVMat = nib.load(PTVFile).get_fdata()
        RTMat = nib.load(RTFile).get_fdata()
        CTMat = np.transpose(CTMat, axes=(2, 1, 0))
        DoseMat = np.transpose(DoseMat, axes=(2, 1, 0))
        PTVMat = np.transpose(PTVMat, axes=(2, 1, 0))
        RTMat = np.transpose(RTMat, axes=(2, 1, 0))
        CTMat += 1000

        ROI_list = {}
        PTV_names = PTVStructs[i]
        for i, name in enumerate(PTV_names):
            ROI_list[name] = PTVMat == i+1
        
        for number, name in OAR_LUT.items():
            mask = RTMat == number
            if np.sum(mask) == 0:
                continue
            ROI_list[name] = mask
        
        numSlices = PTVMat.shape[0]
        imageFolder = os.path.join(targetFolder, "patient_{}".format(patientName))
        if not os.path.isdir(imageFolder):
            os.mkdir(imageFolder)
        
        doseMax = np.max(DoseMat)
        for i in range(numSlices):
            CTSlice = CTMat[i, :, :]
            DoseSlice = DoseMat[i, :, :]
            plt.imshow(CTSlice, cmap="gray")
            plt.imshow(DoseSlice, cmap="jet", vmin=0, vmax=doseMax, alpha=0.3)

            for j, entry in enumerate(ROI_list.items()):
                color = colors[j]
                name, mask = entry
                maskSlice = mask[i, :, :]
                contours = measure.find_contours(maskSlice)
                initial = True
                for contour in contours:
                    if initial:
                        plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                        initial = False
                    else:
                        plt.plot(contour[:, 1], contour[:, 0], color=color)
            plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            file = os.path.join(imageFolder, "{:03d}.png".format(i))
            plt.savefig(file)
            plt.clf()
            print(file)
        print(patientName)


if __name__ == "__main__":
    # test_nib()
    dataView()