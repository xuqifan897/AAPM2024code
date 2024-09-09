import os
import numpy as np
import pydicom
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import measure
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_erosion

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
rootFolder = "/mnt/shengdata1/qifan/TCIAPancreas/" \
    "Pancreatic-CT-CBCT-SEG_v2_20220823/Pancreatic-CT-CBCT-SEG"
numPatients = 40
# Patients 010, 012, 025 have no dose

def viewDataFormat():
    for i in range(numPatients):
        patientName = "Pancreas-CT-CB_{:03d}".format(i+1)
        patientFolder = os.path.join(rootFolder, patientName)
        subFolders = os.listdir(patientFolder)
        if len(subFolders) == 2:
            subFolders = [a for a in subFolders if "CRANE" in a]
        patientFolder1 = os.path.join(patientFolder, subFolders[0])
        domains = os.listdir(patientFolder1)

        if False:
            # view the domains
            print(patientName)
            for domain in domains:
                domainFolder = os.path.join(patientFolder1, domain)
                domainFiles = os.listdir(domainFolder)
                domainFiles.sort()
                firstFile = os.path.join(domainFolder, domainFiles[0])
                dataset = pydicom.dcmread(firstFile)
                print(domain, dataset.Modality)
            print()
        
        if False:
            found = False
            for domain in domains:
                if "DI" in domain:
                    found = True
                    break
            if not found:
                print(patientName)

        if False:
            # the code to check the rtdose properties
            domainAttr = []
            for domain in domains:
                domainFolder = os.path.join(patientFolder1, domain)
                domainFiles = os.listdir(domainFolder)
                domainFiles.sort()
                firstFile = os.path.join(domainFolder, domainFiles[0])
                dataset = pydicom.dcmread(firstFile)
                modality = dataset.Modality
                nFiles = len(domainFiles)
                entry = (domain, modality, nFiles)
                domainAttr.append(entry)
            rtDoseFolders = [a[0] for a in domainAttr if a[1]=="RTDOSE"]
            if len(rtDoseFolders) != 1:
                print(patientName, rtDoseFolders)
        
        if False:
            # the code to find the planning CT
            domains_CT_align = []
            domains_CT_others = []
            for domain in domains:
                domainFolder = os.path.join(patientFolder1, domain)
                domainFiles = os.listdir(domainFolder)
                if len(domainFiles) > 1:
                    if "Aligned" in domain:
                        domains_CT_align.append((domain, len(domainFiles)))
                    else:
                        domains_CT_others.append((domain, len(domainFiles)))
            CT_align_slices = domains_CT_align[0][1]
            for entry in domains_CT_align:
                assert entry[1] == CT_align_slices
            planCT = [a[0] for a in domains_CT_others if a[1] == CT_align_slices]
            assert len(planCT) == 1


def rtMatch():
    fullPancreasImages = "/data/qifan/projects/FastDoseWorkplace/Pancreas/fullView"
    if not os.path.isdir(fullPancreasImages):
        os.mkdir(fullPancreasImages)
    doseFolder = "/mnt/shengdata1/qifan/Pancreas/DoseAligned"

    for i in range(numPatients):
        patientName = "Pancreas-CT-CB_{:03d}".format(i+1)
        patientFolder = os.path.join(rootFolder, patientName)
        subFolders = os.listdir(patientFolder)
        if len(subFolders) == 2:
            subFolders = [a for a in subFolders if "CRANE" in a]
        patientFolder1 = os.path.join(patientFolder, subFolders[0])
        domains = os.listdir(patientFolder1)

        # find the planning CT folder, and RTDOSE
        domains_CT_align = []
        domains_CT_others = []
        for domain in domains:
            domainFolder = os.path.join(patientFolder1, domain)
            domainFiles = os.listdir(domainFolder)
            if len(domainFiles) > 1:
                if "Aligned" in domain:
                    domains_CT_align.append((domain, len(domainFiles)))
                else:
                    domains_CT_others.append((domain, len(domainFiles)))

        CT_align_slices = domains_CT_align[0][1]
        for entry in domains_CT_align:
            assert entry[1] == CT_align_slices
        planCT = [a[0] for a in domains_CT_others if a[1] == CT_align_slices]
        assert len(planCT) == 1
        planCT = os.path.join(patientFolder1, planCT[0])
        
        # find the rt struct file
        rtDomain = None
        for domain in domains:
            if "BSP" in domain:
                rtDomain = domain
                break
        rtFile = os.path.join(patientFolder1, rtDomain, "1-1.dcm")
        rtStruct = RTStructBuilder.create_from(dicom_series_path=planCT, rt_struct_path=rtFile)
        names = rtStruct.get_roi_names()
        ptv = "ROI"
        assert ptv in names
        ptv = rtStruct.get_roi_mask_by_name(ptv)
        ptv = np.flip(ptv, axis=2)
        ctFiles = os.listdir(planCT)
        assert ptv.shape[2] == len(ctFiles)

        if False:
            # erode the ptv by 1cm to get the real PTV
            exampleCT = os.path.join(planCT, ctFiles[0])
            exampleCT = pydicom.dcmread(exampleCT)
            PixelSpacing = exampleCT.PixelSpacing
            SliceThicknesss = exampleCT.SliceThickness
            voxelRes = (PixelSpacing[0], PixelSpacing[1], SliceThicknesss)
            voxelRes = tuple(float(a) for a in voxelRes)
            voxelRes = np.array(voxelRes)
            marginDim = 10 / voxelRes
            fullMarginDim = 2 * marginDim + 1
            fullMarginDim = np.round(fullMarginDim).astype(int)
            fullMargin = np.ones(fullMarginDim)
            ptv = binary_erosion(ptv, structure=fullMargin).astype(np.uint8)

        lungL, lungR = "LUNG_L", "LUNG_R"
        assert lungL in names and lungR in names
        lungL = rtStruct.get_roi_mask_by_name(lungL)
        lungR = rtStruct.get_roi_mask_by_name(lungR)
        lungL = np.flip(lungL, axis=2)
        lungR = np.flip(lungR, axis=2)

        # prepare dose
        doseFile = os.path.join(doseFolder, patientName+".npy")
        dose = np.load(doseFile)
        # normalize
        ptvDose = dose[ptv.astype(bool)]
        thresh = np.percentile(ptvDose, 10)
        dose *= 20 / thresh
            
        patientViewFolder = os.path.join(fullPancreasImages, patientName)
        if not os.path.isdir(patientViewFolder):
            os.mkdir(patientViewFolder)
        nSlices = len(ctFiles)
        colorMap = [(ptv, colors[0]), (lungL, colors[1]), (lungR, colors[2])]
        for j in range(nSlices):
            ctSlice = os.path.join(planCT, "1-{:03d}.dcm".format(j+1))
            ctSlice = pydicom.dcmread(ctSlice).pixel_array
            fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
            ax.imshow(ctSlice, vmin=0, vmax=1800, cmap="gray")

            doseSlice = dose[:, :, j]
            ax.imshow(doseSlice, vmin=0, vmax=30, cmap="jet", alpha=0.3*(doseSlice>1))
            for mat, color in colorMap:
                matSlice = mat[:, :, j]
                contours = measure.find_contours(matSlice)
                if len(contours) == 0:
                    continue
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], color=color)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            figurePath = os.path.join(patientViewFolder, "1-{:03d}.png".format(j+1))
            plt.savefig(figurePath)
            plt.close(fig)
            plt.clf()
            print(figurePath)
        break


def alignDose():
    resultFolder = "/mnt/shengdata1/qifan/Pancreas/DoseAligned"
    for i in range(2, numPatients):
        patientName = "Pancreas-CT-CB_{:03d}".format(i+1)
        patientFolder = os.path.join(rootFolder, patientName)
        subFolders = os.listdir(patientFolder)
        if len(subFolders) == 2:
            subFolders = [a for a in subFolders if "CRANE" in a]
        patientFolder1 = os.path.join(patientFolder, subFolders[0])
        domains = os.listdir(patientFolder1)

        # find CT Folder
        domains_CT_align = []
        domains_CT_others = []
        rtDose = None
        for domain in domains:
            domainFolder = os.path.join(patientFolder1, domain)
            domainFiles = os.listdir(domainFolder)
            if len(domainFiles) > 1:
                if "Aligned" in domain:
                    domains_CT_align.append((domain, len(domainFiles)))
                else:
                    domains_CT_others.append((domain, len(domainFiles)))
            else:
                domainFile = os.path.join(domainFolder, domainFiles[0])
                doseDataset = pydicom.dcmread(domainFile)
                if doseDataset.Modality == "RTDOSE":
                    rtDose = doseDataset
        if rtDose is None:
            continue

        CT_align_slices = domains_CT_align[0][1]
        for entry in domains_CT_align:
            assert entry[1] == CT_align_slices
        planCT = [a[0] for a in domains_CT_others if a[1] == CT_align_slices]
        assert len(planCT) == 1
        planCT = os.path.join(patientFolder1, planCT[0])

        CTData = []
        for file in os.listdir(planCT):
            file_ = os.path.join(planCT, file)
            CTData.append(pydicom.dcmread(file_))
        doseValue = doseInterp(CTData, rtDose)
        doseFile = os.path.join(resultFolder, patientName + ".npy")
        np.save(doseFile, doseValue)
        print(patientName)


def doseInterp(CTData, rtDose):
    # CTData: List of dataset
    # rtDose is the dose dataset
    CTData.sort(key=lambda a: int(a.InstanceNumber))
    ImagePositionPatient = CTData[0].ImagePositionPatient
    sliceShape = (CTData[0].Rows, CTData[0].Columns, len(CTData))
    PixelSpacing = CTData[0].PixelSpacing

    coords_x = ImagePositionPatient[0] + np.arange(sliceShape[0]) * PixelSpacing[0]
    coords_y = ImagePositionPatient[1] + np.arange(sliceShape[1]) * PixelSpacing[1]
    coords_z = np.zeros(len(CTData))
    for i in range(coords_z.size):
        coords_z[i] = CTData[i].ImagePositionPatient[2]
    coords_x = np.expand_dims(coords_x, axis=(1, 2))
    coords_y = np.expand_dims(coords_y, axis=(0, 2))
    coords_z = np.expand_dims(coords_z, axis=(0, 1))
    coordsShape = (sliceShape[0], sliceShape[1], coords_z.size, 3)
    coordsArray = np.zeros(coordsShape)
    coordsArray[:, :, :, 0] = coords_x
    coordsArray[:, :, :, 1] = coords_y
    coordsArray[:, :, :, 2] = coords_z

    GridFrameOffsetVector = rtDose.GridFrameOffsetVector
    ImagePositionPatientDose = rtDose.ImagePositionPatient
    PixelSpacingDose = rtDose.PixelSpacing
    doseArray = rtDose.pixel_array
    doseArray = np.transpose(doseArray, axes=(2, 1, 0))
    shape_dose = doseArray.shape
    SliceThicknessDose = GridFrameOffsetVector[1] - GridFrameOffsetVector[0]

    ImagePositionPatientDose = np.array(ImagePositionPatientDose)
    ImagePositionPatientDose = np.expand_dims(ImagePositionPatientDose, axis=(0, 1, 2))
    coordsArray -= ImagePositionPatientDose
    resDose = (PixelSpacingDose[0], PixelSpacingDose[1], SliceThicknessDose)
    resDose = np.array(resDose)
    resDose = np.expand_dims(resDose, axis=(0, 1, 2))
    coordsArray /= resDose

    nPoints = coordsShape[0] * coordsShape[1] * coordsShape[2]
    coordsArray = np.reshape(coordsArray, (nPoints, 3))

    doseCoordsX = np.arange(shape_dose[0])
    doseCoordsY = np.arange(shape_dose[1])
    doseCoordsZ = np.arange(shape_dose[2])
    doseInterpFunc = RegularGridInterpolator(
        (doseCoordsX, doseCoordsY, doseCoordsZ), doseArray,
        bounds_error=False, fill_value=0)
    doseValues = doseInterpFunc(coordsArray)
    doseValues = np.reshape(doseValues, coordsShape[:3])
    doseValues = np.transpose(doseValues, axes=(1, 0, 2))
    return doseValues


def dvhPtv():
    """
    This function draws the DVH of the ptv area.
    According to the dataset description, the given mask is dilated by 1 cm
    """
    doseFolder = "/mnt/shengdata1/qifan/Pancreas/DoseAligned"
    dvhFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/dvhPlot"
    if not os.path.isdir(dvhFolder):
        os.mkdir(dvhFolder)

    for i in range(numPatients):
        patientName = "Pancreas-CT-CB_{:03d}".format(i+1)
        patientFolder = os.path.join(rootFolder, patientName)
        subFolders = os.listdir(patientFolder)
        if len(subFolders) == 2:
            subFolders = [a for a in subFolders if "CRANE" in a]
        patientFolder1 = os.path.join(patientFolder, subFolders[0])
        domains = os.listdir(patientFolder1)

        # find the planning CT folder, and RTDOSE
        domains_CT_align = []
        domains_CT_others = []
        for domain in domains:
            domainFolder = os.path.join(patientFolder1, domain)
            domainFiles = os.listdir(domainFolder)
            if len(domainFiles) > 1:
                if "Aligned" in domain:
                    domains_CT_align.append((domain, len(domainFiles)))
                else:
                    domains_CT_others.append((domain, len(domainFiles)))

        CT_align_slices = domains_CT_align[0][1]
        for entry in domains_CT_align:
            assert entry[1] == CT_align_slices
        planCT = [a[0] for a in domains_CT_others if a[1] == CT_align_slices]
        assert len(planCT) == 1
        planCT = os.path.join(patientFolder1, planCT[0])

        # find the rt struct file
        rtDomain = None
        for domain in domains:
            if "BSP" in domain:
                rtDomain = domain
                break
        rtFile = os.path.join(patientFolder1, rtDomain, "1-1.dcm")
        rtStruct = RTStructBuilder.create_from(dicom_series_path=planCT, rt_struct_path=rtFile)
        names = rtStruct.get_roi_names()
        ptv = "ROI"
        assert ptv in names
        ptv = rtStruct.get_roi_mask_by_name(ptv)
        ptv = np.flip(ptv, axis=2)

        doseFile = os.path.join(doseFolder, patientName + ".npy")
        if not os.path.isfile(doseFile):
            continue
        dose = np.load(doseFile)

        ptvDose = dose[ptv.astype(bool)]
        thresh = np.percentile(ptvDose, 10)
        ptvDose *= 20 / thresh
        ptvDose = np.sort(ptvDose)
        ptvDose = np.insert(ptvDose, 0, 0)
        nPoints = np.sum(ptv) + 1
        yAxis = (1 - np.arange(nPoints) / (nPoints-1)) * 100
        plt.plot(ptvDose, yAxis)
        plt.xlabel("Dose (Gy)")
        plt.ylabel("Percentile (%)")
        plt.title("DVH for {}".format(patientName))
        figureFile = os.path.join(dvhFolder, patientName + ".png")
        plt.savefig(figureFile)
        plt.clf()
        print(figureFile)


if __name__ == "__main__":
    # viewDataFormat()
    # rtMatch()
    # alignDose()
    dvhPtv()