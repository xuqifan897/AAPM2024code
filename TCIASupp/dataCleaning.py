import os
import re
import numpy as np
import pydicom
from rt_utils import RTStructBuilder
from scipy.interpolate import RegularGridInterpolator
import concurrent.futures
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import measure

OnDataServer = False
if OnDataServer:
    # on data server
    SourceFolder = "/data/SharedFolder/qifan/TCIA/Head-Neck-Cetuximab" \
        "/TCIA_Head-Neck_Cetuximab_06-22-2015/Head-Neck_Cetuximab"
    TargetFolder = "/data/SharedFolder/qifan/TCIA/DataView"
else:
    # on compute server
    SourceFolder = "/mnt/shengdata1/qifan/TCIA/Head-Neck-Cetuximab/" \
        "TCIA_Head-Neck_Cetuximab_06-22-2015/Head-Neck_Cetuximab"
    TargetFolder = "/data/qifan/projects/FastDoseWorkplace/TCIASupp"

DataMapping = {}  # patient: path
colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())

def checkStructs():
    def getPatientName(path: str):
        return path.split("/")[8]
    allFiles = getAllFiles(SourceFolder)
    pattern = "RT Dose"
    doseFiles = [a for a in allFiles if pattern in a]
    patient = getPatientName(doseFiles[0])
    
    mapping = {}
    for file in doseFiles:
        patient = getPatientName(file)
        if patient in mapping:
            mapping[patient].append(file)
        else:
            mapping[patient] = [file]
    
    washedMapping = {}
    filePattern = "totalhetero"
    for patient, filelist in mapping.items():
        if len(filelist) > 1:
            filelist = [a for a in filelist if filePattern in a]
            assert len(filelist) in [0, 1]
            if len(filelist) == 1:
                washedMapping[patient] = filelist[0]
        else:
            washedMapping[patient] = filelist[0]
    global DataMapping
    for patient, dosefile in washedMapping.items():
        groupFolder = os.path.dirname(os.path.dirname(dosefile))
        groupFolderChildren = os.listdir(groupFolder)
        patterns = ["CTs from rtog", "RTStruct from rtog"]
        for pattern in patterns:
            flag = False
            for child in groupFolderChildren:
                if pattern in child:
                    flag = True
                    break
            if flag:
                DataMapping[patient] = (groupFolder, dosefile)


def getAllFiles(path: str):
    """
    This function gets all files in all directories and subdirectories
    """
    result = []
    items = os.listdir(path)
    for item in items:
        fullpath = os.path.join(path, item)
        if os.path.isfile(fullpath):
            result.append(fullpath)
        else:
            sub_result = getAllFiles(fullpath)
            result.extend(sub_result)
    return result


def parallel_dose_processing(entry):
    patient, groupFolder, dosefile = entry
    subFolders = os.listdir(groupFolder)
    CTFolder = [a for a in subFolders if "CTs" in a]
    RTFolder = [a for a in subFolders if "RTStruct" in a]
    if len(CTFolder) != 1 or len(RTFolder) != 1:
        # print(patient, ": ", CTFolder, RTFolder)
        return
    CTFolder = os.path.join(groupFolder, CTFolder[0])
    RTFolder = os.path.join(groupFolder, RTFolder[0])

    CTArray, DoseArray = getCTandDose(CTFolder, dosefile)
    PatientFolder = os.path.join(TargetFolder, patient)
    if not os.path.isdir(PatientFolder):
        os.mkdir(PatientFolder)
    CTFile = os.path.join(PatientFolder, "CTArray.npy")
    DoseFile = os.path.join(PatientFolder, "DoseArray.npy")
    np.save(CTFile, CTArray)
    np.save(DoseFile, DoseArray)
    print(PatientFolder)


def DataView():
    """
    This function prints out the dose map and structure annotations
    """
    arguments = []
    for patient, entry in DataMapping.items():
        groupFolder, doseFile = entry
        arguments = (patient, groupFolder, doseFile)
        parallel_dose_processing(arguments)
        print(patient)


def getCTandDose(CTFolder, doseFile):
    """
    This function gets aligned CT array and dose array
    """
    CTData = []
    SourceFiles = os.listdir(CTFolder)
    ImageOrientationPatient = None
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

    return CTArray, doseValues


def CTStructDoseShow():
    """
    This function shows the image combining CT, RTStruct, and dose
    """
    skip_first = True
    for patient, entry in DataMapping.items():
        if skip_first:
            skip_first = False
            continue
        groupFolder, dosefile = entry
        subFolders = os.listdir(groupFolder)
        CTFolder = [a for a in subFolders if "CTs" in a]
        RTFolder = [a for a in subFolders if "RTStruct" in a]
        if len(CTFolder) != 1 or len(RTFolder) != 1:
            continue
        CTFolder = os.path.join(groupFolder, CTFolder[0])
        RTFile = os.path.join(groupFolder, RTFolder[0], '1-1.dcm')
        assert os.path.isdir(CTFolder) and os.path.isfile(RTFile)
        RTStruct = RTStructBuilder.create_from(
            dicom_series_path=CTFolder, rt_struct_path=RTFile)
        StructNames = RTStruct.get_roi_names()
        PTVs_lower = [a for a in StructNames if "tv" in a.lower()]
        primaryPTV = None
        primaryDose = 0
        for name in PTVs_lower:
            dose = re.findall(r'\d+', name)
            if len(dose) != 1:
                continue
            dose = eval(dose[0])
            if dose > primaryDose:
                primaryDose = dose
                primaryPTV = name
        if primaryPTV is None:
            if len(PTVs_lower) == 0:
                print("No PTV, GTV, and CTV volumes found")
                continue
            else:
                primaryPTV = PTVs_lower[0]
                primaryDose = 70

        primaryPTVMask = RTStruct.get_roi_mask_by_name(primaryPTV)
        MaskDict = {}
        for structname in StructNames:
            try:
                structmask = RTStruct.get_roi_mask_by_name(structname)
            except:
                print("Failed loading structure {}".format(structname))
                continue
            structmask = np.flip(structmask, axis=2)
            MaskDict[structname] = structmask

        dosefile = os.path.join(TargetFolder, patient, "DoseArray.npy")
        DoseArray = np.load(dosefile)
        primaryPTVDose = DoseArray[primaryPTVMask]
        thresh = np.percentile(primaryPTVDose, 5)
        DoseArray *= primaryDose / thresh

        CTfile = os.path.join(TargetFolder, patient, "CTArray.npy")
        CTArray = np.load(CTfile)

        imageFolder = os.path.join(TargetFolder, patient, "View")
        if not os.path.isdir(imageFolder):
            os.mkdir(imageFolder)
        for i in range(CTArray.shape[2]):
            plt.imshow(CTArray[:, :, i], cmap="gray", vmin=500, vmax=1500)
            for j, entry in enumerate(MaskDict.items()):
                name, array = entry
                MaskSlice = array[:, :, i]
                if np.sum(MaskSlice) == 0:
                    continue
                contours = measure.find_contours(MaskSlice)
                initial = True
                for contour in contours:
                    if initial:
                        plt.plot(contour[:, 1], contour[:, 0], color=colors[j], label=name)
                        initial = False
                    else:
                        plt.plot(contour[:, 1], contour[:, 0], color=colors[j])
            doseSlice = DoseArray[:, :, i]
            plt.imshow(doseSlice, cmap="jet", vmin=0, vmax=90, alpha=0.3)
            plt.legend(loc="upper left", bbox_to_anchor=[1.01, 1.0])
            plt.tight_layout()
            imageFile = os.path.join(imageFolder, "{:03d}.png".format(i))
            plt.savefig(imageFile)
            plt.clf()
        print("Patient {} done.".format(patient))


def dataMove():
    """
    This function runs on the compute server, not the data server.
    It copies data from the data server to the compute server
    """
    patients = [2, 3, 9, 13, 70, 125, 132, 159, 190]
    PatternDose = "RT Dose"
    PatternCT = "CTs from rtog"
    PatternRT = "RTStruct from rtog"
    for patient in patients:
        PatientSource = os.path.join(SourceFolder, "0522c{:04d}".format(patient))
        PatientChildFolders = [os.path.join(PatientSource, a) for a in os.listdir(PatientSource)]
        PatientGrandchildFolders = {a: os.listdir(a) for a in PatientChildFolders}
        FolderWeWant = None
        for a, grandchildren in PatientGrandchildFolders.items():
            patternFind = False
            for b in grandchildren:
                if PatternDose in b:
                    patternFind = True
                    break
            if patternFind:
                FolderWeWant = a
                break
        
        # find dose file
        grandchildren = os.listdir(FolderWeWant)
        DoseFolders = [a for a in grandchildren if PatternDose in a]
        assert len(DoseFolders) > 0
        if len(DoseFolders) > 1:
            PatternDoseSub = "totalhetero"
            DoseFolders = [a for a in DoseFolders if PatternDoseSub]
            assert len(DoseFolders) == 1
        DoseFolder = DoseFolders[0]
        DoseFile = os.path.join(FolderWeWant, DoseFolder, "1-1.dcm")
        
        # find CT Folder
        CTFolders = [a for a in grandchildren if PatternCT in a]
        assert len(CTFolders) == 1
        CTFolder = os.path.join(FolderWeWant, CTFolders[0])

        # find RTFile
        RTFolders = [a for a in grandchildren if PatternRT in a]
        assert len(RTFolders) == 1
        RTFile = os.path.join(FolderWeWant, RTFolders[0], "1-1.dcm")

        assert os.path.isfile(DoseFile) and os.path.isdir(CTFolder) and os.path.isfile(RTFile)

        # prepare target folder
        destFolder = os.path.join(TargetFolder, "{:03d}".format(patient))
        if not os.path.isdir(destFolder):
            os.mkdir(destFolder)
        destCTFolder = os.path.join(destFolder, "data")
        if not os.path.isdir(destCTFolder):
            os.mkdir(destCTFolder)
        destRTFile = os.path.join(destFolder, "RTStruct.dcm")
        destDoseFile = os.path.join(destFolder, "dose.dcm")
        command1 = "cp \"{}\"/* {}".format(os.path.join(CTFolder), destCTFolder)
        command2 = "cp '{}' {}".format(RTFile, destRTFile)
        command3 = "cp '{}' {}".format(DoseFile, destDoseFile)
        os.system(command1)
        os.system(command2)
        os.system(command3)


if __name__ == "__main__":
    if OnDataServer:
        checkStructs()
    # DataView()
    # CTStructDoseShow()
    dataMove()