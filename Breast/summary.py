import os
import numpy as np
import pydicom
from rt_utils import RTStructBuilder
import json
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import measure, transform
from scipy.interpolate import RegularGridInterpolator

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
colors[7] = colors[-1]
rootFolder = "/data/qifan/projects/FastDoseWorkplace/Breast"

def summary():
    DVHFolder = os.path.join(rootFolder, "DVH")
    if not os.path.isdir(DVHFolder):
        os.mkdir(DVHFolder)
    
    patients = os.listdir(rootFolder)
    for patient in patients:
        patientFolder = os.path.join(rootFolder, patient)
        DVHFile = os.path.join(patientFolder, "DVHComp.png")
        targetFile = os.path.join(DVHFolder, "DVH_{}.png".format(patient))
        command = "cp {} {}".format(DVHFile, targetFile)
        print(command)
        os.system(command)


def draw_dose_wash():
    patients = os.listdir(rootFolder)
    patients = [a for a in patients if a[:8].isnumeric()]
    patients = [a for a in patients if "Left" not in a]
    print(patients)

    doseWashFolder = os.path.join(rootFolder, "DoseWash")
    if not os.path.isdir(doseWashFolder):
        os.mkdir(doseWashFolder)
    for patient in patients:
        patientFolder = os.path.join(rootFolder, patient)
        CTFolder = os.path.join(patientFolder, "dicom")
        dicomDoseFile = os.path.join(patientFolder, "RTDose.dcm")
        dicomDoseArray = examineDose(dicomDoseFile, CTFolder)

        dimensionFile = os.path.join(patientFolder, "expFolder", "prep_output", "dimension.txt")
        with open(dimensionFile, "r") as f:
            lines = f.readlines()
        doseShape = lines[0]
        doseShape = doseShape.split(" ")
        doseShape = [int(a) for a in doseShape]
        doseShape.reverse()
        doseShape = tuple(doseShape)

        dicomDoseArray = np.flip(dicomDoseArray, axis=0)
        dicomDoseArray = transform.resize(dicomDoseArray, doseShape)

        optDoseFile = os.path.join(patientFolder, "expFolder", "plan1", "dose.bin")
        optDoseArray = np.fromfile(optDoseFile, dtype=np.float32)
        optDoseArray = np.reshape(optDoseArray, doseShape)

        densityFile = os.path.join(patientFolder, "expFolder", "density_raw.bin")
        densityArray = np.fromfile(densityFile, dtype=np.uint16)
        densityArray = np.reshape(densityArray, doseShape)

        roi_list_file = os.path.join(patientFolder, "expFolder", "prep_output", "roi_list.h5")
        structures = getStructures(roi_list_file)

        structures_metadata_file = os.path.join(patientFolder, "expFolder", "structures.json")
        with open(structures_metadata_file, "r") as f:
            structures_metadata = json.load(f)
        PTVName = structures_metadata["ptv"]
        BodyName = structures_metadata["oar"][0]

        # Exclude RingStructure
        exclude = ["RingStructure"]
        structures = [a for a in structures if a[0] not in exclude]
        PTVEntry = [a for a in structures if a[0] == PTVName]
        assert len(PTVEntry) == 1, "More or no PTV found"
        PTVMask = PTVEntry[0][1]
        PTVMask = PTVMask > 0

        BodyEntry = [a for a in structures if a[0] == BodyName]
        assert len(BodyEntry) == 1, "More or no Body found"
        BodyMask = BodyEntry[0][1]
        BodyMask = BodyMask > 0
        BodyMaskComplement = np.logical_not(BodyMask)

        # normalize
        PTVDose_dicom = dicomDoseArray[PTVMask]
        thresh_dicom = np.percentile(PTVDose_dicom, 5)
        dicomDoseArray = dicomDoseArray * 30 / thresh_dicom

        PTVDose_opt = optDoseArray[PTVMask]
        thresh_opt = np.percentile(PTVDose_opt, 5)
        optDoseArray = optDoseArray * 30 / thresh_opt
        optDoseArray[BodyMaskComplement] = 0

        doseWashFolder_patient = os.path.join(doseWashFolder, patient)
        if not os.path.isdir(doseWashFolder_patient):
            os.mkdir(doseWashFolder_patient)
        nSlices = doseShape[0]
        for j in range(nSlices):
            densitySlice = densityArray[j, :, :]
            densitySlice = np.concatenate((densitySlice, densitySlice), axis=1)
            optDoseSlice = optDoseArray[j, :, :]
            dicomDoseSlice = dicomDoseArray[j, :, :]
            doseSlice = np.concatenate((optDoseSlice, dicomDoseSlice), axis=1)
            fig, ax = plt.subplots(figsize=(14, 6))
            plt.imshow(densitySlice, cmap="gray", vmin=0, vmax=2000)
            plt.imshow(doseSlice, cmap="jet", vmin=0, vmax=40, alpha=0.3)
            for k, entry in enumerate(structures):
                color = colors[k]
                name, mask = entry
                maskSlice = mask[j, :, :]
                maskSlice = np.concatenate((maskSlice, maskSlice), axis=1)
                contours = measure.find_contours(maskSlice)
                initial = True
                for contour in contours:
                    if initial:
                        plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                        initial = False
                    else:
                        plt.plot(contour[:, 1], contour[:, 0], color=color)
            plt.text(10, 10, "4-pi", color="white", fontsize=16)
            plt.text(10 + doseShape[2], 10, "clinic", color="white", fontsize=16)
            plt.legend()
            plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
            plt.tight_layout()
            file = os.path.join(doseWashFolder_patient, "{:03d}.png".format(j))
            plt.savefig(file)
            plt.clf()
            plt.close(fig)
            print(file)


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
        


def examineDose(doseFile: str, CTFolder: str):
    CTData = []
    SourceFiles = os.listdir(CTFolder)
    ImOrienPatient_CT = None
    for file in SourceFiles:
        file = os.path.join(CTFolder, file)
        dataset = pydicom.dcmread(file)
        if dataset.Modality == "CT":
            InstanceNumber = int(dataset.InstanceNumber)
            if ImOrienPatient_CT is None:
                ImOrienPatient_CT = dataset.ImageOrientationPatient
            elif ImOrienPatient_CT != dataset.ImageOrientationPatient:
                print("ImageOrientationPatient attribute inconsistent")
            CTData.append((InstanceNumber, dataset))
    CTData.sort(key=lambda a: a[0])
    dimY, dimX = CTData[0][1].pixel_array.shape
    PixelSpacingY, PixelSpacingX = CTData[0][1].PixelSpacing
    CTcoordsShape = (dimX, dimY, len(CTData), 3)
    coords_array = np.zeros(CTcoordsShape, dtype=float)

    for i in range(len(CTData)):
        baseCoords = np.array(CTData[i][1].ImagePositionPatient)
        baseCoords = np.expand_dims(baseCoords, axis=(0, 1))
        coords_array[:, :, i, :] = baseCoords
        
    # Take into account the influence of the x coordinates
    vector_x = np.array(ImOrienPatient_CT[:3]) * PixelSpacingX
    vector_x = np.expand_dims(vector_x, axis=(0, 1, 2))
    voxelIdx = np.arange(dimX)
    voxelIdx = np.expand_dims(voxelIdx, axis=(1, 2, 3))
    offset_x = vector_x * voxelIdx

    # Take into account the influence of the y coordinates
    vector_y = np.array(ImOrienPatient_CT[3:]) * PixelSpacingY
    vector_y = np.expand_dims(vector_y, axis=(0, 1, 2))
    voxelIdx = np.arange(dimY)
    voxelIdx = np.expand_dims(voxelIdx, axis=(0, 2, 3))
    offset_y = vector_y * voxelIdx

    coords_array = coords_array + offset_x + offset_y


    doseDataset = pydicom.dcmread(doseFile)
    ImOrienPatient_dose = doseDataset.ImageOrientationPatient
    doseArray = doseDataset.pixel_array
    doseArray = np.transpose(doseArray, axes=(2, 1, 0))
    doseShape = doseArray.shape
    ImagePositionPatient_dose = doseDataset.ImagePositionPatient
    DosePixelSpacing = doseDataset.PixelSpacing
    GridFrameOffsetVector = doseDataset.GridFrameOffsetVector
    SliceThickness_Dose = GridFrameOffsetVector[1] - GridFrameOffsetVector[0]
    res_dose = (DosePixelSpacing[0], DosePixelSpacing[1], SliceThickness_Dose)
    res_dose = np.array(res_dose)
    doseCoordsX = np.arange(doseShape[0]) * res_dose[0] * ImOrienPatient_dose[0]
    doseCoordsY = np.arange(doseShape[1]) * res_dose[1] * ImOrienPatient_dose[4]
    sign_z = ImOrienPatient_dose[0] * ImOrienPatient_dose[4]
    doseCoordsZ = np.arange(doseShape[2]) * res_dose[2] * sign_z
    doseInterpFunc = RegularGridInterpolator(
        (doseCoordsX, doseCoordsY, doseCoordsZ), doseArray,
        bounds_error=False, fill_value=0.0)
    
    ImagePositionPatient_dose = np.expand_dims(ImagePositionPatient_dose, axis=(0, 1, 2))
    coords_array = coords_array - ImagePositionPatient_dose
    nPoints = CTcoordsShape[0] * CTcoordsShape[1] * CTcoordsShape[2]
    coords_array = np.reshape(coords_array, (nPoints, 3))
    doseValues = doseInterpFunc(coords_array)
    doseValues = np.reshape(doseValues, tuple(CTcoordsShape[:3]))
    doseValues = np.transpose(doseValues, axes=(2, 1, 0))
    return doseValues


if __name__ == "__main__":
    # summary()
    draw_dose_wash()