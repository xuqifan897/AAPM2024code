"""
This code runs on the compute server
"""
import os
import glob
import numpy as np
import random
import json
import pydicom
from rt_utils import RTStructBuilder
from scipy.interpolate import RegularGridInterpolator
from skimage import transform, measure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

SourceFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASupp"
# patients = [2, 3, 9, 13, 70, 125, 132, 159]
patients = [190]
# patients = [3]
TargetRes = 2.5  # mm
StructsMetadata = None

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())

def PhantomConvert():
    """
    This function converts the Dicom file formats into binary ones
    """
    for patient in patients:
        PatientFolder = os.path.join(SourceFolder, "{:03d}".format(patient))
        CTFolder = os.path.join(PatientFolder, "data")
        DoseFile = os.path.join(PatientFolder, "dose.dcm")
        CTArray, DoseArray, VoxelSize, RescaleSlope, RescaleIntercept \
            = getCTandDose(CTFolder, DoseFile)
        
        # Transpose to (slice, height, width)
        CTArray = np.transpose(CTArray, axes=(2, 0, 1))
        CTArray = np.flip(CTArray, axis=0)
        DoseArray = np.transpose(DoseArray, axes=(2, 0, 1))
        DoseArray = np.flip(DoseArray, axis=0)
        VoxelSize = np.array((VoxelSize[2], VoxelSize[0], VoxelSize[1]))
        DimOrg = np.array(CTArray.shape)
        DimNew = DimOrg * VoxelSize / TargetRes
        DimNew = DimNew.astype(int)

        CTArray = CTArray.astype(np.float32)
        CTArray = transform.resize(CTArray, DimNew)
        CTArray = CTArray.astype(np.uint16)
        DoseArray = transform.resize(DoseArray, DimNew)
        DoseArray = DoseArray.astype(np.float32)

        # Then get the structures
        RTFile = os.path.join(PatientFolder, "RTStruct.dcm")
        RTStruct = RTStructBuilder.create_from(dicom_series_path=CTFolder, rt_struct_path=RTFile)
        names = RTStruct.get_roi_names()
        MaskDict = {}
        for name in names:
            try:
                MaskArray = RTStruct.get_roi_mask_by_name(name)
            except:
                print("Cannot access mask {} of patient {}".format(name, patient))
                continue
            MaskArray = np.transpose(MaskArray, axes=(2, 0, 1))
            MaskArray = MaskArray.astype(float)
            MaskArray = transform.resize(MaskArray, DimNew)
            MaskArray = MaskArray > 0
            MaskArray = MaskArray.astype(np.uint8)
            MaskDict[name] = MaskArray
        
        DensityFile = os.path.join(PatientFolder, "density_raw.bin")
        DoseFile = os.path.join(PatientFolder, "dose.bin")
        MaskFolder = os.path.join(PatientFolder, "InputMask")
        if not os.path.isdir(MaskFolder):
            os.mkdir(MaskFolder)

        CTArray.tofile(DensityFile)
        DoseArray.tofile(DoseFile)
        for name, MaskArray in MaskDict.items():
            FileName = os.path.join(MaskFolder, name.replace(" ", "") + ".bin")
            MaskArray.tofile(FileName)
        metadata = "{} {} {}\n{}\n{}".format(DimNew[0], DimNew[1], DimNew[2],
            RescaleSlope, RescaleIntercept)
        MetadataFile = os.path.join(PatientFolder, "metadata.txt")
        with open(MetadataFile, "w") as f:
            f.write(metadata)
        print("Patient {} done. Dimension: {}, Slope: {}, Intercept: {}".format(
            patient, DimNew, RescaleSlope, RescaleIntercept))


def getCTandDose(CTFolder, doseFile):
    """
    This function gets aligned CT array and dose array
    Returns CTArray, DoseArray, VoxelSize in the order (height, width, slice)
    """
    CTData = []
    SourceFiles = os.listdir(CTFolder)
    ImageOrientationPatient = None
    SliceThickness = None
    for file in SourceFiles:
        path = os.path.join(CTFolder, file)
        dataset = pydicom.dcmread(path)
        if dataset.Modality == "CT":
            InstanceNumber = int(dataset.InstanceNumber)
            CTData.append((InstanceNumber, dataset.ImagePositionPatient, dataset.pixel_array))
            if ImageOrientationPatient is None:
                ImageOrientationPatient = dataset.ImageOrientationPatient
                shape = CTData[0][2].shape
                SliceThickness = dataset.SliceThickness
                PixelSpacing = dataset.PixelSpacing
                RescaleSlope = dataset.RescaleSlope
                RescaleIntercept = dataset.RescaleIntercept
            else:
                assert SliceThickness == dataset.SliceThickness
    CTData.sort(key=lambda a: a[0])
    numSlices = len(CTData)
    shape_CT = (shape[0], shape[1], numSlices)
    coordsShape = shape_CT + (3,)
    coords_array = np.zeros(coordsShape, dtype=float)

    ImagePositionPatient = CTData[0][1]
    coords_x = ImagePositionPatient[0] + np.arange(shape[0]) * PixelSpacing[0]
    coords_y = ImagePositionPatient[1] + np.arange(shape[1]) * PixelSpacing[1]
    coords_z = np.zeros(numSlices, dtype=coords_x.dtype)
    for i in range(numSlices):
        coords_z[i] = CTData[i][1][2]

    if float(SliceThickness) == 0.0:
        # deal with special caase
        coords_z_diff = np.diff(coords_z)
        SliceThickness = coords_z_diff[0]
        flag = coords_z_diff == SliceThickness
        flag = np.all(flag)
        assert flag

    coords_x = np.expand_dims(coords_x, axis=(1, 2))
    coords_y = np.expand_dims(coords_y, axis=(0, 2))
    coords_z = np.expand_dims(coords_z, axis=(0, 1))
    coords_array[:, :, :, 0] = coords_x
    coords_array[:, :, :, 1] = coords_y
    coords_array[:, :, :, 2] = coords_z

    doseDataset = pydicom.dcmread(doseFile)
    GridFrameOffsetVector = doseDataset.GridFrameOffsetVector
    ImagePositionPatientDose = doseDataset.ImagePositionPatient
    PixelSpacing_Dose = doseDataset.PixelSpacing
    doseArray = doseDataset.pixel_array
    doseArray = np.transpose(doseArray, axes=(2, 1, 0))
    shape_Dose = doseDataset.pixel_array.shape
    shape_Dose = (shape_Dose[2], shape_Dose[1], shape_Dose[0])
    SliceThickness_Dose = GridFrameOffsetVector[1] - GridFrameOffsetVector[0]

    ImagePositionPatientDose = np.array(ImagePositionPatientDose)
    ImagePositionPatientDose = np.expand_dims(ImagePositionPatientDose, axis=(0, 1, 2))
    coords_array -= ImagePositionPatientDose
    res_dose = (PixelSpacing_Dose[0], PixelSpacing_Dose[1], SliceThickness_Dose)
    res_dose = np.array(res_dose)
    res_dose = np.expand_dims(res_dose, axis=(0, 1, 2))
    coords_array /= res_dose

    nPoints = shape_CT[0] * shape_CT[1] * shape_CT[2]
    coords_array = np.reshape(coords_array, (nPoints, 3))

    doseCoordsX = np.arange(shape_Dose[0])
    doseCoordsY = np.arange(shape_Dose[1])
    doseCoordsZ = np.arange(shape_Dose[2])
    doseInterpFunc = RegularGridInterpolator(
        (doseCoordsX, doseCoordsY, doseCoordsZ), doseArray,
        bounds_error=False, fill_value=0.0)
    doseValues = doseInterpFunc(coords_array)
    doseValues = np.reshape(doseValues, shape_CT)
    doseValues = np.transpose(doseValues, axes=(1, 0, 2))

    CTArray = np.zeros(shape_CT, dtype=CTData[0][2].dtype)
    for i in range(numSlices):
        CTArray[:, :, i] = CTData[i][2]

    VoxelSize = np.array((PixelSpacing[0], PixelSpacing[1], SliceThickness))
    return CTArray, doseValues, VoxelSize, RescaleSlope, RescaleIntercept


def dataView():
    """
    This function overlays the CT density, anatomy, and dose
    """
    for patient in patients:
        PatientFolder = os.path.join(SourceFolder, "{:03d}".format(patient))
        metadata = os.path.join(PatientFolder, "metadata.txt")
        with open(metadata, "r") as f:
            line = f.readline()
        dimension = eval(line.replace(" ", ", "))

        CTArrayFile = os.path.join(PatientFolder, "density_raw.bin")
        CTArray = np.fromfile(CTArrayFile, dtype=np.uint16)
        CTArray = np.reshape(CTArray, dimension)

        DoseArrayFile = os.path.join(PatientFolder, "dose.bin")
        DoseArray = np.fromfile(DoseArrayFile, dtype=np.float32)
        DoseArray = np.reshape(DoseArray, dimension)
        DoseRoof = np.percentile(DoseArray, 99)
        DoseArray *= 70 / DoseRoof

        MaskFolder = os.path.join(PatientFolder, "InputMask")
        MaskDict = {}
        for file in os.listdir(MaskFolder):
            name = file.split(".")[0]
            file = os.path.join(MaskFolder, file)
            MaskArray = np.fromfile(file, dtype=np.uint8)
            MaskArray = np.reshape(MaskArray, dimension)
            MaskDict[name] = MaskArray
        
        ViewFolder = os.path.join(PatientFolder, "view")
        if not os.path.isdir(ViewFolder):
            os.mkdir(ViewFolder)
        for i in range(dimension[0]):
            CTSlice = CTArray[i, :, :]
            DoseSlice = DoseArray[i, :, :]
            plt.imshow(CTSlice, cmap="gray", vmin=0, vmax=1500)
            plt.imshow(DoseSlice, cmap="jet", vmin=0, vmax=70, alpha=0.3)
            for j, entry in enumerate(MaskDict.items()):
                name, MaskArray = entry
                color = colors[j]
                MaskSlice = MaskArray[i, :, :]
                if np.sum(MaskSlice) == 0:
                    continue
                contours = measure.find_contours(MaskSlice)
                initial = True
                for contour in contours:
                    if initial:
                        plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                        initial = False
                    else:
                        plt.plot(contour[:, 1], contour[:, 0], color=color)
            plt.legend()
            plt.tight_layout()
            FigureFile = os.path.join(ViewFolder, "{:03d}.png".format(i))
            plt.savefig(FigureFile)
            plt.clf()
            print(FigureFile)


def GenNiftiCTArray():
    for patient in patients:
        PatientFolder = os.path.join(SourceFolder, "{:03d}".format(patient))
        metadata = os.path.join(PatientFolder, "metadata.txt")
        with open(metadata, "r") as f:
            line = f.readline()
        dimension = eval(line.replace(" ", ", "))
        DensityFile = os.path.join(PatientFolder, "density_raw.bin")
        CTArray = np.fromfile(DensityFile, dtype=np.uint16)
        CTArray = np.reshape(CTArray, dimension)
        nifti_img = nib.Nifti1Image(CTArray, affine=np.eye(4))
        outputFile = os.path.join(PatientFolder, "density_raw.nii.gz")
        nib.save(nifti_img, outputFile)
        print(outputFile)


def MaskGen():
    """
    This function generates the segmentation masks
    """
    for patient in patients:
        if patient < 70:
            continue
        PatientFolder = os.path.join(SourceFolder, "{:03d}".format(patient))
        DicomFolder = os.path.join(PatientFolder, "data")
        OutputFolder = os.path.join(PatientFolder, "TotalSeg")
        totalsegmentator(DicomFolder, OutputFolder)
        print(OutputFolder)


def FixPatient159():
    """
    The method above failed with patient 159, saying the slice spacing inconsistent among slices
    """
    patient = 159
    PatientFolder = os.path.join(SourceFolder, "{:03d}".format(patient))
    CTFolder = os.path.join(PatientFolder, "data_org")
    CTData = []
    for file in os.listdir(CTFolder):
        file = os.path.join(CTFolder, file)
        dataset = pydicom.dcmread(file)
        InstanceNumber = int(dataset.InstanceNumber)
        CTData.append([InstanceNumber, file, dataset])
    CTData.sort(key=lambda a: a[0])

    Displacement = [a[2].ImagePositionPatient for a in CTData]
    OffsetsZ = [a[2] for a in Displacement]
    OffsetsZ = np.array(OffsetsZ)
    OffsetsZDiff = np.diff(OffsetsZ)
    # print(OffsetsZDiff)
    # we got that the average slice thickness is -2.5
    SliceThickness = np.mean(OffsetsZDiff)
    SliceThickness = round(SliceThickness, 2)
    SliceThicknessLiteral = abs(SliceThickness)
    OffsetsZBase = OffsetsZ[0]
    print(Displacement[0])
    TargetCTFolder = os.path.join(PatientFolder, "data")
    if not os.path.isdir(TargetCTFolder):
        os.mkdir(TargetCTFolder)
    for i in range(len(Displacement)):
        CTData[i][2].ImagePositionPatient[2] = OffsetsZBase + i * SliceThickness
        CTData[i][2].SliceThickness = SliceThicknessLiteral
        TargetCTFile = os.path.join(TargetCTFolder, "1-{:03d}.dcm".format(i+1))
        CTData[i][2].save_as(TargetCTFile)
        print(TargetCTFile)


def CopyBrainMask():
    """
    This function copies the brain mask to the folder "InputMask"
    """
    for patient in patients:
        PatientFolder = os.path.join(SourceFolder, "{:03d}".format(patient))
        BrainSourceFile = os.path.join(PatientFolder, "TotalSeg", "brain.nii.gz")
        BrainMask = nib.load(BrainSourceFile).get_fdata()
        BrainMask = (BrainMask > 0).astype(np.uint8)
        BrainMask = np.transpose(BrainMask, axes=(2, 1, 0))
        BrainMask = np.flip(BrainMask, axis=1)

        metadata = os.path.join(PatientFolder, "metadata.txt")
        with open(metadata, "r") as f:
            line = f.readline()
        line = eval(line.replace(" ", ", "))
        BrainMask = BrainMask.astype(np.float32)
        BrainMask = transform.resize(BrainMask, line)
        BrainMask = (BrainMask > 0).astype(np.uint8)

        BrainFileTarget = os.path.join(PatientFolder, "InputMask", "BRAIN.bin")
        BrainMask.tofile(BrainFileTarget)
        print(BrainFileTarget)


def StructsExclude():
    """
    This function removes the structures that are irrelevant in the optimization
    """
    global StructsMetadata
    StructsMetadata = {
        2: {"exclude": ["TransPTV56", "CTV56", "TransPTV70", "GTV", "CTV56", "avoid"],
            "PTV": ["PTV70", "PTV56"],
            "BODY": "SKIN"},
        3: {"exclude": ["GTV", "ptv54combo", "transvol70"],
            "PTV": ["CTV56", "PTV56", "PTV70", "leftptv56"],
            "BODY": "SKIN"},
        9: {"exclude": ["ptv_70+", "GTV", "CTV70", "ltpar+", "rtpar+"],
            "PTV": ["CTV56", "PTV56", "PTV70"],
            "BODY": "SKIN"},
        13: {"exclude": ["CTV70", "GTV"],
             "PTV": ["CTV56", "PTV56", "PTV70"],
             "BODY": "SKIN"},
        70: {"exclude": ["CTV56", "CTV70", "GTV"],
             "PTV": ["PTV56", "PTV70"],
             "BODY": "SKIN"},
        125: {"exclude": ["CTV56", "CTV70", "GTV"],
              "PTV": ["PTV56", "PTV70"],
              "BODY": "SKIN"},
        132: {"exclude": ["CTV70", "GTV"],
              "PTV": ["CTV56", "PTV56", "PTV70"],
              "BODY": "SKIN"},
        159: {"exclude": ["CTV56", "CTV63", "CTV70", "GTV"],
              "PTV": ["PTV56", "PTV63", "PTV70"],
              "BODY": "SKIN"},
        190: {"exclude": ["CTV70", "GTV"],
              "PTV": ["CTV56", "PTV56", "PTV70"],
              "BODY": "SKIN"}
    }
    for patient, MetaInfo in StructsMetadata.items():
        exclude = MetaInfo["exclude"]
        PTVs = MetaInfo["PTV"]
        Body = MetaInfo["BODY"]
        PatientFolder = os.path.join(SourceFolder, "{:03d}".format(patient))
        metadata = os.path.join(PatientFolder, "metadata.txt")
        with open(metadata, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        MaskFolder = os.path.join(PatientFolder, "InputMask")
        Structs = os.listdir(MaskFolder)
        Structs = [a.split(".")[0] for a in Structs]
        for ex in exclude:
            assert ex in Structs, f"{ex} not in Patient {patient} Structs"
        Structs = [a for a in Structs if a not in exclude]
        for ex in PTVs:
            assert ex in Structs, f"{ex} not in Patient {patient} Structs"
        assert Body in Structs, f"{Body} not in Patient {patient} Structs"


def PostDataView():
    """
    Plot selected structures
    This function overlays the CT density, anatomy, and dose
    """
    for patient in patients:
        PatientFolder = os.path.join(SourceFolder, "{:03d}".format(patient))
        metadata = os.path.join(PatientFolder, "metadata.txt")
        with open(metadata, "r") as f:
            line = f.readline()
        dimension = eval(line.replace(" ", ", "))

        CTArrayFile = os.path.join(PatientFolder, "density_raw.bin")
        CTArray = np.fromfile(CTArrayFile, dtype=np.uint16)
        CTArray = np.reshape(CTArray, dimension)

        DoseArrayFile = os.path.join(PatientFolder, "dose.bin")
        DoseArray = np.fromfile(DoseArrayFile, dtype=np.float32)
        DoseArray = np.reshape(DoseArray, dimension)
        DoseRoof = np.percentile(DoseArray, 99)
        DoseArray *= 70 / DoseRoof

        StructsRemove = StructsMetadata[patient]["exclude"]
        MaskFolder = os.path.join(PatientFolder, "InputMask")
        MaskDict = {}
        for file in os.listdir(MaskFolder):
            name = file.split(".")[0]
            if name in StructsRemove:
                continue
            file = os.path.join(MaskFolder, file)
            MaskArray = np.fromfile(file, dtype=np.uint8)
            MaskArray = np.reshape(MaskArray, dimension)
            MaskDict[name] = MaskArray
        
        ViewFolder = os.path.join(PatientFolder, "View")
        if not os.path.isdir(ViewFolder):
            os.mkdir(ViewFolder)
        for i in range(dimension[0]):
            CTSlice = CTArray[i, :, :]
            DoseSlice = DoseArray[i, :, :]
            plt.imshow(CTSlice, cmap="gray", vmin=0, vmax=1500)
            plt.imshow(DoseSlice, cmap="jet", vmin=0, vmax=70, alpha=0.3)
            for j, entry in enumerate(MaskDict.items()):
                name, MaskArray = entry
                color = colors[j]
                MaskSlice = MaskArray[i, :, :]
                if np.sum(MaskSlice) == 0:
                    continue
                contours = measure.find_contours(MaskSlice)
                initial = True
                for contour in contours:
                    if initial:
                        plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                        initial = False
                    else:
                        plt.plot(contour[:, 1], contour[:, 0], color=color)
            plt.legend()
            plt.tight_layout()
            FigureFile = os.path.join(ViewFolder, "{:03d}.png".format(i))
            plt.savefig(FigureFile)
            plt.clf()
            print(FigureFile)


def MaskTrim():
    "This function merges PTV masks of the same dose, and crops masks so that PTV masks do not " \
    "overlap with each other, and OAR masks do not overlap with PTV masks"
    for patient in patients:
        PatientFolder = os.path.join(SourceFolder, "{:03d}".format(patient))
        InputMaskFolder = os.path.join(PatientFolder, "InputMask")
        OutputMaskFolder = os.path.join(PatientFolder, "PlanMask")
        if not os.path.isdir(OutputMaskFolder):
            os.mkdir(OutputMaskFolder)
        
        ExcludeList = (a:=StructsMetadata[patient])["exclude"]
        PTVList = a["PTV"]
        BODY = a["BODY"]

        SpecialComb = ExcludeList + PTVList + [BODY]
        OARs = [b for a in os.listdir(InputMaskFolder) if (b:=a.split(".")[0]) not in SpecialComb]
        # group PTVs into different dose levels
        PTVGroups = {}
        for ptv in PTVList:
            dose = "".join(a for a in ptv if a.isdigit())
            dose = eval(dose)
            if dose not in PTVGroups:
                PTVGroups[dose] = [ptv]
            else:
                PTVGroups[dose].append(ptv)
        
        PTVMasksMerge = []
        for dose, group in PTVGroups.items():
            canvas = None
            for name in group:
                MaskFile = os.path.join(InputMaskFolder, name + ".bin")
                MaskArray = np.fromfile(MaskFile, dtype=np.uint8)
                if canvas is None:
                    canvas = MaskArray
                else:
                    canvas = np.logical_or(canvas, MaskArray)
            PTVMasksMerge.append([dose, canvas])
        PTVMasksMerge.sort(key=lambda a: a[0], reverse=True)

        # deal with overlap
        canvas = None
        for i in range(len(PTVMasksMerge)):
            PTVMask = PTVMasksMerge[i][1]
            if canvas is None:
                canvas = PTVMask
            else:
                PTVMask = np.logical_and(PTVMask, np.logical_not(canvas))
                canvas = np.logical_or(PTVMask, canvas)
                PTVMask = PTVMask.astype(np.uint8)
                PTVMasksMerge[i][1] = PTVMask
        
        OARMaskDict = {}
        for name in OARs:
            OARMaskFile = os.path.join(InputMaskFolder, "{}.bin".format(name))
            OARMask = np.fromfile(OARMaskFile, dtype=np.uint8)
            OARMask = np.logical_and(OARMask, np.logical_not(canvas))
            OARMask = OARMask.astype(np.uint8)
            OARMaskDict[name] = OARMask
        
        # write results
        for dose, mask in PTVMasksMerge:
            destFile = os.path.join(OutputMaskFolder, "PTV{}.bin".format(dose))
            mask.tofile(destFile)
        for name, mask in OARMaskDict.items():
            destFile = os.path.join(OutputMaskFolder, "{}.bin".format(name))
            mask.tofile(destFile)
        BODYSource = os.path.join(InputMaskFolder, "{}.bin".format(BODY))
        BODYDest = os.path.join(OutputMaskFolder, "{}.bin".format(BODY))
        command = "cp \"{}\" \"{}\"".format(BODYSource, BODYDest)
        os.system(command)
        print("Patient {} done!".format(patient))


def TrimDataView():
    """
    Plot selected structures
    This function overlays the CT density, anatomy, and dose
    """
    for patient in patients:
        PatientFolder = os.path.join(SourceFolder, "{:03d}".format(patient))
        metadata = os.path.join(PatientFolder, "metadata.txt")
        with open(metadata, "r") as f:
            line = f.readline()
        dimension = eval(line.replace(" ", ", "))

        CTArrayFile = os.path.join(PatientFolder, "density_raw.bin")
        CTArray = np.fromfile(CTArrayFile, dtype=np.uint16)
        CTArray = np.reshape(CTArray, dimension)

        DoseArrayFile = os.path.join(PatientFolder, "dose.bin")
        DoseArray = np.fromfile(DoseArrayFile, dtype=np.float32)
        DoseArray = np.reshape(DoseArray, dimension)
        DoseRoof = np.percentile(DoseArray, 99)
        DoseArray *= 70 / DoseRoof

        MaskFolder = os.path.join(PatientFolder, "PlanMask")
        MaskDict = {}
        for file in os.listdir(MaskFolder):
            name = file.split(".")[0]
            file = os.path.join(MaskFolder, file)
            MaskArray = np.fromfile(file, dtype=np.uint8)
            MaskArray = np.reshape(MaskArray, dimension)
            MaskDict[name] = MaskArray
        
        ViewFolder = os.path.join(PatientFolder, "PostView")
        if not os.path.isdir(ViewFolder):
            os.mkdir(ViewFolder)
        for i in range(dimension[0]):
            CTSlice = CTArray[i, :, :]
            DoseSlice = DoseArray[i, :, :]
            plt.imshow(CTSlice, cmap="gray", vmin=0, vmax=1500)
            plt.imshow(DoseSlice, cmap="jet", vmin=0, vmax=80, alpha=0.3)
            for j, entry in enumerate(MaskDict.items()):
                name, MaskArray = entry
                color = colors[j]
                MaskSlice = MaskArray[i, :, :]
                if np.sum(MaskSlice) == 0:
                    continue
                contours = measure.find_contours(MaskSlice)
                initial = True
                for contour in contours:
                    if initial:
                        plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                        initial = False
                    else:
                        plt.plot(contour[:, 1], contour[:, 0], color=color)
            plt.legend()
            plt.tight_layout()
            FigureFile = os.path.join(ViewFolder, "{:03d}.png".format(i))
            plt.savefig(FigureFile)
            plt.clf()
            print(FigureFile)


def PTVSeg():
    """
    This function follows the method proposed by Qihui et cl in the paper
    "Many-isocenter optimization for robotic radiotherpay"
    """
    for patient in patients:
        PatientFolder = os.path.join(SourceFolder, "{:03d}".format(patient))
        metadata = os.path.join(PatientFolder, "metadata.txt")
        with open(metadata, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        MaskPattern = os.path.join(PatientFolder, "PlanMask", "PTV*[0-9][0-9]*.bin")
        PTVList = [os.path.basename(a).split(".")[0] for a in glob.glob(MaskPattern)]

        MaskFolder = os.path.join(PatientFolder, "PlanMask")
        PTVMaskMerge = None
        for name in PTVList:
            file = os.path.join(MaskFolder, "{}.bin".format(name))
            maskArray = np.fromfile(file, dtype=np.uint8)
            if PTVMaskMerge is None:
                PTVMaskMerge = maskArray
            else:
                PTVMaskMerge = np.logical_or(PTVMaskMerge, maskArray)
        PTVMaskMerge = np.reshape(PTVMaskMerge, dimension)
        
        # we find the minimum bounding box that encapsulate the whole PTV area
        # and then divide the whole PTV volume into 2 x 2 sub-blocks
        AxisX = np.any(PTVMaskMerge, axis=(0, 1))
        indices = [a for a in range(AxisX.size) if AxisX[a]]
        AxisXLower = min(indices)
        AxisXUpper = max(indices) + 1
        AxisXMiddle = int((AxisXLower + AxisXUpper) / 2)
        AxisXPoints = [AxisXLower, AxisXMiddle, AxisXUpper]

        AxisY = np.any(PTVMaskMerge, axis=(0, 2))
        indices = [a for a in range(AxisY.size) if AxisY[a]]
        AxisYLower = min(indices)
        AxisYUpper = max(indices) + 1

        AxisZ = np.any(PTVMaskMerge, axis=(1, 2))
        indices = [a for a in range(AxisZ.size) if AxisZ[a]]
        AxisZLower = min(indices)
        AxisZUpper = max(indices) + 1
        AxisZMiddle = int((AxisZLower + AxisZUpper) / 2)
        AxisZPoints = [AxisZLower, AxisZMiddle, AxisZUpper]

        for i in range(2):
            IdxXBegin = AxisXPoints[i]
            IdxXEnd = AxisXPoints[i+1]
            for j in range(2):
                IdxZBegin = AxisZPoints[j]
                IdxZEnd = AxisZPoints[j+1]
                Mask = np.zeros_like(PTVMaskMerge)
                Mask[IdxZBegin: IdxZEnd, AxisYLower:AxisYUpper, IdxXBegin: IdxXEnd] = 1
                PTVAndMask = np.logical_and(PTVMaskMerge, Mask)
                PTVAndMask = PTVAndMask.astype(np.uint8)

                PTVSegIdx = i * 2 + j
                OutputFile = os.path.join(MaskFolder, "PTVSeg{}.bin".format(PTVSegIdx))
                PTVAndMask.tofile(OutputFile)
                print(OutputFile)

        PTVMaskMerge = PTVMaskMerge.astype(np.uint8)
        PTVMergeFile = os.path.join(MaskFolder, "PTVMerge.bin")
        PTVMaskMerge.tofile(PTVMergeFile)
        print(PTVMergeFile)
        print()


def ShowPTVSeg():
    """
    This function shows the PTVsegs
    """
    for patient in patients:
        PatientFolder = os.path.join(SourceFolder, "{:03d}".format(patient))
        metadata = os.path.join(PatientFolder, "metadata.txt")
        with open(metadata, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)

        CTArrayFile = os.path.join(PatientFolder, "density_raw.bin")
        CTArray = np.fromfile(CTArrayFile, dtype=np.uint16)
        CTArray = np.reshape(CTArray, dimension)

        DoseArrayFile = os.path.join(PatientFolder, "dose.bin")
        DoseArray = np.fromfile(DoseArrayFile, dtype=np.float32)
        DoseArray = np.reshape(DoseArray, dimension)
        DoseRoof = np.percentile(DoseArray, 99)
        DoseArray *= 70 / DoseRoof

        MaskFiles = [os.path.join(PatientFolder, "PlanMask", "PTVSeg{}.bin".format(i)) for i in range(4)]
        Masks = []
        for i, file in enumerate(MaskFiles):
            MaskArray = np.fromfile(file, dtype=np.uint8)
            MaskArray = np.reshape(MaskArray, dimension)
            print(np.sum(MaskArray))
            Masks.append(("PTVSeg{}".format(i), MaskArray))

        FigureFolder = os.path.join(SourceFolder, "{:03d}".format(patient), "PostView")
        if not os.path.isdir(FigureFolder):
            os.mkdir(FigureFolder)
        for i in range(dimension[0]):
            CTSlice = CTArray[i, :, :]
            DoseSlice = DoseArray[i, :, :]
            plt.imshow(CTSlice, cmap="gray", vmin=500, vmax=1500)
            LegendFlag = False
            for j, entry in enumerate(Masks):
                color = colors[j]
                name, MaskArray = entry
                MaskSlice = MaskArray[i, :, :]
                if np.sum(MaskSlice) == 0:
                    continue
                LegendFlag = True
                contours = measure.find_contours(MaskSlice)
                Initial = True
                for contour in contours:
                    if Initial:
                        plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                        Initial = False
                    else:
                        plt.plot(contour[:, 1], contour[:, 0], color=color)
            plt.imshow(DoseSlice, cmap="jet", vmin=0, vmax=80, alpha=0.3)
            if LegendFlag:
                plt.legend()
            FigureFile = os.path.join(FigureFolder, "{:03d}.png".format(i))
            plt.savefig(FigureFile)
            plt.clf()
            print(FigureFile)


def structsInfoGen():
    """
    This function generates the "structures.json" and "StructureInfo.csv" files for all patients
    """
    PTVName = "PTVMerge"
    BBoxName = "SKIN"
    for patient in patients:
        PatientFolder = os.path.join(SourceFolder, "{:03d}".format(patient))
        MaskFolder = os.path.join(PatientFolder, "PlanMask")
        Structures = [a.split(".")[0] for a in os.listdir(MaskFolder)]
        assert PTVName in Structures and BBoxName in Structures, \
            "Either PTV or BBox not in Structures"
        Structures.remove(PTVName)
        Structures.remove(BBoxName)
        Structures.insert(0, BBoxName)
        content = {
            "prescription": 70,
            "ptv": PTVName,
            "oar": Structures
        }
        content = json.dumps(content, indent=4)
        
        FastDoseFolder = os.path.join(PatientFolder, "FastDose")
        if not os.path.isdir(FastDoseFolder):
            os.mkdir(FastDoseFolder)
        contentFile = os.path.join(FastDoseFolder, "structures.json")
        with open(contentFile, "w") as f:
            f.write(content)


        Auxiliary = ["PTVSeg0", "PTVSeg1", "PTVSeg2", "PTVSeg3", "SKIN"]
        Structures = [a for a in Structures if a not in Auxiliary]
        PTVs = [a for a in Structures if  "PTV" in a]
        OARs = [a for a in Structures if a not in PTVs]
        PTVDose = []
        for name in PTVs:
            dose = "".join(a for a in name if a.isdigit())
            dose = eval(dose)
            PTVDose.append((name, dose))
        PTVDose.sort(key=lambda a: a[1], reverse=True)
        
        content = "Name,maxWeights,maxDose,minDoseTargetWeights,minDoseTarget,OARWeights,IdealDose"
        for name, dose in PTVDose:
            line = "{},100,{},100,{},NaN,{}".format(name, dose, dose, dose)
            content = content + "\n" + line
        for name in OARs:
            line = "{},0,18,NaN,NaN,5,0".format(name)
            content = content + "\n" + line
        # add RingStructure
        line = "RingStructure,0,18,NaN,NaN,2,0"
        content = content + "\n" + line
        contentFile = os.path.join(FastDoseFolder, "StructureInfo.csv")
        with open(contentFile, "w") as f:
            f.write(content)
        
        print("Patient {} done!".format(patient))


def BeamListGen():
    """
    Due to memory limit, we can not use the full set of beams
    """
    SamplingRatio = 0.2
    BeamListFullPath = "/data/qifan/projects/AAPM2024/TCIASupp/beamlistHN.txt"
    TargetFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASupp"

    with open(BeamListFullPath, "r") as f:
        lines = f.readlines()
    lines = [a[:-1] if a[-1]=="\n" else (a + "") for a in lines]
    SegNum = 4
    for i in range(4):
        LocalList = [(i, lines[i]) for i in range(len(lines))]
        random.shuffle(LocalList)
        NumRemain = int(SamplingRatio * len(lines))
        LocalList = LocalList[: NumRemain]
        LocalList.sort(key = lambda a: a[0])
        LocalList = [a[1] for a in LocalList]
        LocalContent = "\n".join(LocalList)
        BeamListFile = os.path.join(TargetFolder, "BeamListPTVSeg{}.txt".format(i))
        with open(BeamListFile, "w") as f:
            f.write(LocalContent)


if __name__ == "__main__":
    # PhantomConvert()
    # dataView()
    # GenNiftiCTArray()
    # MaskGen()
    # FixPatient159()
    # CopyBrainMask()
    # StructsExclude()
    # PostDataView()
    # MaskTrim()
    # TrimDataView()
    # PTVSeg()
    # ShowPTVSeg()
    structsInfoGen()
    # BeamListGen()