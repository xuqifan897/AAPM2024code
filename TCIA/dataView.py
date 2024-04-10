import os
import numpy as np
import pydicom
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
import matplotlib.colors  as mcolors
from skimage import measure


def viewInitial0001():
    CTFolder = "/data/qifan/projects/FastDoseWorkplace/TCIA/examples" \
        "/0522c0001/08-23-1999-NeckHeadNeckPETCT-03251/2.000000-CT 5.0 H30s-55580"
    RTFile = "/data/qifan/projects/FastDoseWorkplace/TCIA/examples" \
        "/0522c0001/08-31-1999-522-26555/2.000000-RTStruct from rtog conversion-555.3/1-1.dcm"
    RTDoseFile = "/data/qifan/projects/FastDoseWorkplace/TCIA/" \
        "examples/0522c0001/08-31-1999-522-26555/4.000000-RT Dose - fx1hetero-5.5.1/1-1.dcm"
    
    # To extract the RTStruct
    files = os.listdir(CTFolder)
    rtstruct = RTStructBuilder.create_from(dicom_series_path=CTFolder, rt_struct_path=RTFile)
    names = rtstruct.get_roi_names()
    print(names)


def viewInitial0002():
    # CTFolder = "/data/qifan/projects/FastDoseWorkplace/TCIA/examples/" \
    #     "0522c0002/12-12-1999-NeckHeadNeckPETCT-72105/2.000000-CT 5.0 H30s-21956"
    CTFolder = "/data/qifan/projects/FastDoseWorkplace/TCIA/examples" \
        "/0522c0002/09-06-1999-522-26556/1.000000-CTs from rtog conversion-556.2"
    RTFile = "/data/qifan/projects/FastDoseWorkplace/TCIA/examples" \
        "/0522c0002/09-06-1999-522-26556/2.000000-RTStruct from rtog conversion-556.3/1-1.dcm"
    RTDoseFile = "/data/qifan/projects/FastDoseWorkplace/TCIA/examples" \
        "/0522c0002/09-06-1999-522-26556/4.000000-RT Dose - fx1hetero-6.5.1/1-1.dcm"
    
    CTFolder = CTFolder.replace(" ", "\ ")
    RTFile = RTFile.replace(" ", "\ ")
    RTDoseFile = RTDoseFile.replace(" ", "\ ")

    # rtstruct = RTStructBuilder.create_from(dicom_series_path=CTFolder, rt_struct_path=RTFile)
    # names = rtstruct.get_roi_names()
    # print(names)

    targetFolder = "/data/qifan/projects/FastDoseWorkplace/TCIA/case02"
    targetCTFolder = os.path.join(targetFolder, "data")
    if not os.path.isdir(targetCTFolder):
        os.makedirs(targetCTFolder)
    command = "cp {}/* {}".format(CTFolder, targetCTFolder)
    # print(command)
    os.system(command)

    command = "cp {} {}/RTstruct.dcm".format(RTFile, targetCTFolder)
    # print(command)
    os.system(command)

    command = "cp {} {}/RTDose.dcm".format(RTDoseFile, targetFolder)
    # print(command)
    os.system(command)


def viewInitial0003():
    CTFolder = "/data/qifan/projects/FastDoseWorkplace/TCIA/examples" \
        "/0522c0003/09-13-1999-522-26557/1.000000-CTs from rtog conversion-557.2"
    RTFile = "/data/qifan/projects/FastDoseWorkplace/TCIA/examples" \
        "/0522c0003/09-13-1999-522-26557/2.000000-RTStruct from rtog conversion-557.3/1-1.dcm"
    RTDoseFile = "/data/qifan/projects/FastDoseWorkplace/TCIA/examples" \
        "/0522c0003/09-13-1999-522-26557/4.000000-RT Dose - fx1hetero-7.5.1/1-1.dcm"
    
    CTFolder = CTFolder.replace(" ", "\ ")
    RTFile = RTFile.replace(" ", "\ ")
    RTDoseFile = RTDoseFile.replace(" ", "\ ")
    
    # rtstruct = RTStructBuilder.create_from(dicom_series_path=CTFolder, rt_struct_path=RTFile)
    # names = rtstruct.get_roi_names()
    # print(names)

    targetFolder = "/data/qifan/projects/FastDoseWorkplace/TCIA/case03"
    targetCTFolder = os.path.join(targetFolder, "data")
    if not os.path.isdir(targetCTFolder):
        os.makedirs(targetCTFolder)
    command = "cp {}/* {}".format(CTFolder, targetCTFolder)
    # print(command)
    os.system(command)

    command = "cp {} {}/RTstruct.dcm".format(RTFile, targetCTFolder)
    # print(command)
    os.system(command)

    command = "cp {} {}/RTDose.dcm".format(RTDoseFile, targetFolder)
    # print(command)
    os.system(command)


def inspect0002():
    globalFolder = "/data/qifan/projects/FastDoseWorkplace/TCIA/case02"
    CTFolder = os.path.join(globalFolder, "data")
    RTFile = os.path.join(globalFolder, "RTDose.dcm")
    resultFolder = os.path.join(globalFolder, "view")
    if not os.path.isdir(resultFolder):
        os.mkdir(resultFolder)

    ctDatasets = []
    rtFiles = []
    files = os.listdir(CTFolder)
    for file in files:
        path = os.path.join(CTFolder, file)
        dataset = pydicom.dcmread(path)
        if dataset.Modality == "CT":
            InstanceNumber = int(dataset.InstanceNumber)
            ctDatasets.append((InstanceNumber, dataset))
        else:
            rtFiles.append(path)
    ctDatasets.sort(key = lambda a: a[0])
    
    rtstruct = RTStructBuilder.create_from(dicom_series_path=CTFolder, rt_struct_path=rtFiles[0])
    names = rtstruct.get_roi_names()
    masks = {}
    for name in names:
        mask = rtstruct.get_roi_mask_by_name(name)
        masks[name] = mask

    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
    
    numSlices = len(ctDatasets)
    for i in range(numSlices):
        image = ctDatasets[i][1].pixel_array
        plt.imshow(image, cmap="gray")
        for j, entry in enumerate(masks.items()):
            name, mask = entry
            color = colors[j]
            maskSlice = mask[:, :, -i-1]
            contours = measure.find_contours(maskSlice)
            initial = True
            for contour in contours:
                if initial:
                    plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                    initial = False
                else:
                    plt.plot(contour[:, 1], contour[:, 0], color=color)
        plt.legend()
        file = os.path.join(resultFolder, "{:03d}.png".format(i))
        plt.savefig(file)
        plt.clf()
        print(file)


def inspect0003():
    globalFolder = "/data/qifan/projects/FastDoseWorkplace/TCIA/case03"
    CTFolder = os.path.join(globalFolder, "data")
    RTFile = os.path.join(globalFolder, "RTDose.dcm")
    resultFolder = os.path.join(globalFolder, "view")
    if not os.path.isdir(resultFolder):
        os.mkdir(resultFolder)

    ctDatasets = []
    rtFiles = []
    files = os.listdir(CTFolder)
    for file in files:
        path = os.path.join(CTFolder, file)
        dataset = pydicom.dcmread(path)
        if dataset.Modality == "CT":
            InstanceNumber = int(dataset.InstanceNumber)
            ctDatasets.append((InstanceNumber, dataset))
        else:
            rtFiles.append(path)
    ctDatasets.sort(key = lambda a: a[0])
    
    rtstruct = RTStructBuilder.create_from(dicom_series_path=CTFolder, rt_struct_path=rtFiles[0])
    names = rtstruct.get_roi_names()
    names = [a for a in names if "GTV" not in a and "CTV" not in a]
    masks = {}
    for name in names:
        mask = rtstruct.get_roi_mask_by_name(name)
        masks[name] = mask

    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
    
    numSlices = len(ctDatasets)
    for i in range(numSlices):
        image = ctDatasets[i][1].pixel_array
        plt.imshow(image, cmap="gray")
        for j, entry in enumerate(masks.items()):
            name, mask = entry
            color = colors[j]
            maskSlice = mask[:, :, -i-1]
            contours = measure.find_contours(maskSlice)
            initial = True
            for contour in contours:
                if initial:
                    plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                    initial = False
                else:
                    plt.plot(contour[:, 1], contour[:, 0], color=color)
        plt.legend()
        file = os.path.join(resultFolder, "{:03d}.png".format(i))
        plt.savefig(file)
        plt.clf()
        print(file)   


def dataExtract():
    sourceFolder = "/mnt/shengdata1/qifan/TCIA/Head-Neck-Cetuximab" \
        "/TCIA_Head-Neck_Cetuximab_06-22-2015/Head-Neck_Cetuximab"
    targetFolder = "/data/qifan/projects/FastDoseWorkplace/TCIA"
    textFile = "/data/qifan/projects/AAPM2024/TCIA/listOutput.txt"

    if False:
        if os.path.isfile(textFile):
            os.remove(textFile)
        command = "find {} -name \"*rtog*\" > {}".format(sourceFolder, textFile)
        os.system(command)
        return
    
    with open(textFile, "r") as f:
        lines = f.readlines()
    lines = [a[:-1] for a in lines]
    content = {}
    for line in lines:
        components = line.split("/")
        patient = [a for a in components if "0522c" in a][0]
        if patient in content:
            content[patient].append(line)
        else:
            content[patient] = [line]

    if not os.path.isdir(targetFolder):
        os.mkdir(targetFolder)

    # filter out the contents that contain two entries
    content_filtered = {a: b for a, b in content.items() if len(b) == 2}
    for patient, entry in content_filtered.items():
        CTFolder = [a for a in entry if "CT" in a][0]
        file2 = [a for a in entry if "RTStruct" in a][0]
        rtFile = os.listdir(file2)[0]
        rtFile = os.path.join(file2, rtFile)
        targetPatient = os.path.join(targetFolder, patient)
        dataFolder = os.path.join(targetPatient, "data")
        if not os.path.isdir(dataFolder):
            os.makedirs(dataFolder)
        
        rtFile = rtFile.replace(" ", "\ ")
        CTFolder = CTFolder.replace(" ", "\ ")
        command0 = "cp {} {}/RTStruct.dcm".format(rtFile, dataFolder)
        command1 = "cp {}/* {}".format(CTFolder, dataFolder)
        os.system(command0)
        os.system(command1)
        print(patient)


def findSubdirectories(parent: str) -> list[str]: 
    children = os.listdir(parent)
    children = [os.path.join(parent, a) for a in children]
    children = [a for a in children if os.path.isdir(a)]
    if len(children) == 0:
        return [parent]
    subs = [findSubdirectories(child) for child in children]
    subs = [item for sublist in subs for item in sublist]
    subs.append(parent)
    return subs


def dataVisualization():
    globalFolder = "/data/qifan/projects/FastDoseWorkplace/TCIA"
    patients = os.listdir(globalFolder)
    patients.sort()
    for patient in patients:
        patientFolder = os.path.join(globalFolder, patient)
        dataFolder = os.path.join(patientFolder, "data")
        imageFolder = os.path.join(patientFolder, "view")
        if not os.path.isdir(imageFolder):
            os.mkdir(imageFolder)
        generateImage(dataFolder, imageFolder)


def generateImage(CTFolder, imageFolder):
    ctDatasets = []
    rtFiles = []
    files = os.listdir(CTFolder)
    for file in files:
        path = os.path.join(CTFolder, file)
        dataset = pydicom.dcmread(path)
        if dataset.Modality == "CT":
            InstanceNumber = int(dataset.InstanceNumber)
            ctDatasets.append((InstanceNumber, dataset))
        else:
            rtFiles.append(path)
    ctDatasets.sort(key = lambda a: a[0])
    
    rtstruct = RTStructBuilder.create_from(dicom_series_path=CTFolder, rt_struct_path=rtFiles[0])
    names = rtstruct.get_roi_names()
    names = [a for a in names if "GTV" not in a and "CTV" not in a]
    masks = {}
    for name in names:
        mask = rtstruct.get_roi_mask_by_name(name)
        masks[name] = mask
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
    
    numSlices = len(ctDatasets)
    for i in range(numSlices):
        image = ctDatasets[i][1].pixel_array
        plt.imshow(image, cmap="gray")
        for j, entry in enumerate(masks.items()):
            name, mask = entry
            color = colors[j]
            maskSlice = mask[:, :, -i-1]
            contours = measure.find_contours(maskSlice)
            initial = True
            for contour in contours:
                if initial:
                    plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                    initial = False
                else:
                    plt.plot(contour[:, 1], contour[:, 0], color=color)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        file = os.path.join(imageFolder, "{:03d}.png".format(i))
        plt.savefig(file)
        plt.clf()
        print(file)


def doseInterpretation():
    """
    This function interprets different dose files
    """
    sourceFolder = "/mnt/shengdata1/qifan/TCIA/Head-Neck-Cetuximab/" \
        "TCIA_Head-Neck_Cetuximab_06-22-2015/Head-Neck_Cetuximab/" \
        "0522c0331/06-11-2001-522-26595"
    targetFolder = "/data/qifan/projects/FastDoseWorkplace/TCIA/0522c0331"

    CTFolder = os.path.join(sourceFolder, '1.000000-CTs from rtog conversion-595.2')
    RTFile = os.path.join(sourceFolder, '2.000000-RTStruct from rtog conversion-595.3', '1-1.dcm')
    DoseFx1File = os.path.join(sourceFolder, '4.000000-RT Dose - fx1hetero-5.5.1', '1-1.dcm')
    DoseFx2File = os.path.join(sourceFolder, '6.000000-RT Dose - fx2hetero-5.5.2', '1-1.dcm')
    DoseFx3File = os.path.join(sourceFolder, '8.000000-RT Dose - fx3hetero-5.5.3', '1-1.dcm')
    DoseFx4File = os.path.join(sourceFolder, '10.000000-RT Dose - fx4hetero-5.5.4', '1-1.dcm')
    DoseTotalFile = os.path.join(sourceFolder, '12.000000-RT Dose - totalhetero-5.5.5', '1-1.dcm')

    # prepare dose
    DoseFiles = [DoseFx1File, DoseFx2File, DoseFx3File, DoseFx4File, DoseTotalFile]
    DoseMatrices = []
    ImagePositionsDose = []
    for file in DoseFiles:
        dataset = pydicom.dcmread(file)
        DoseMatrices.append(dataset.pixel_array)
        ImagePositionsDose.append(dataset.ImagePositionPatient)

    # prepare folder
    imageFolders = ["Dose1View", "Dose2View", "Dose3View", "Dose4View", "DoseTotalView"]
    imageFolders = [os.path.join(targetFolder, a) for a in imageFolders]
    for folder in imageFolders:
        if not os.path.isdir(folder):
            os.mkdir(folder)
    
    # prepare CT matrix
    CTSlices = []
    ImagePositionsCT = []
    files = os.listdir(CTFolder)
    shape = None
    datatype = None
    for file in files:
        path = os.path.join(CTFolder, file)
        dataset = pydicom.dcmread(path)
        InstanceNumber = int(dataset.InstanceNumber)
        CTSlices.append((InstanceNumber, dataset.pixel_array))
        # ImagePositionsCT.append(dataset.)
        if shape is None:
            shape = CTSlices[0][1].shape
            datatype = CTSlices[0][1].dtype
        print(file)
    CTSlices.sort(key=lambda a: a[0])
    CTMatrixSize = [len(CTSlices), shape[0], shape[1]]
    CTMatrix = np.zeros(CTMatrixSize, dtype=datatype)
    for mat in DoseMatrices:
        print(mat.shape)
    print(CTMatrix.shape)
    return

    # prepare rtstruct
    rtstruct = RTStructBuilder.create_from(dicom_series_path=CTFolder, rt_struct_path=RTFile)
    names = rtstruct.get_roi_names()
    names = [a for a in names if "CTV" not in a and "GTV" not in a]
    masks = {}
    for name in names:
        mask = rtstruct.get_roi_mask_by_name(name)
        masks[name] = mask
        print(name)


if __name__ == '__main__':
    # viewInitial0002()
    # viewInitial0003()
    # inspect0002()
    # inspect0003()
    # dataExtract()
    # dataVisualization()
    doseInterpretation()