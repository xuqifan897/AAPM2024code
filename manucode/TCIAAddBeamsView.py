import os
import numpy as np
from rt_utils import RTStructBuilder
from skimage import transform
import nrrd
import matplotlib.colors as mcolors
from scipy.signal import convolve
from scipy.ndimage import label, find_objects

sourceFolder1 = "/data/qifan/projects/FastDoseWorkplace/TCIASupp"
sourceFolder2 = "/data/qifan/projects/FastDoseWorkplace/TCIAAdd"
patients = ["002", "003", "009", "013", "070", "125", "132", "190"]

exclude = ["PTVMerge", "rind", "PTVSeg0", "PTVSeg1", "PTVSeg2",
           "PTVSeg3", "PTVMerge", "GTV", "avoid", "ptv54combo", "transvol70"]
Converge = {"BrainStem": ["BRAIN_STEM", "Brainstem", "BRAIN_STEM_PRV"],
            "OralCavity": ["oralcavity", "oralCavity", "ORAL_CAVITY", "OralCavity"],
            "OPTIC_NERVE": ["OPTIC_NERVE", "OPTC_NERVE"]}
ConvergeReverse = {}
for name, collection in Converge.items():
    for child in collection:
        ConvergeReverse[child] = name


colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
colors_skip = [11, 13, 14, 16, 18]
idx = 18
for i in colors_skip:
    colors[i] = colors[idx]
    idx += 1
StructureList = []
colorMap = {}

def StructsInit():
    """
    This function is to generate a coherent structure list for all patients
    """
    global StructureList, colorMap
    for patient in patients:
        if ".txt" in patient:
            continue
        patientFolder = os.path.join(sourceFolder2, patient)
        InputMaskFolder = os.path.join(patientFolder, "PlanMask")
        structuresLocal = os.listdir(InputMaskFolder)
        structuresLocal = [a.split(".")[0].replace(" ", "") for a in structuresLocal]
        for a in structuresLocal:
            if a not in StructureList:
                StructureList.append(a)
    StructureList_copy = []
    for name in StructureList:
        if name in ConvergeReverse:
            name = ConvergeReverse[name]
        if name not in StructureList_copy and name not in exclude and "+" not in name:
            StructureList_copy.append(name)
    StructureList = StructureList_copy.copy()
    for i in range(len(StructureList)):
        colorMap[StructureList[i]] = colors[i]


def addAnnotationToDicom():
    """
    This function adds the segmentation masks added (EYE_LT, EYE_RT, OPTIC_NERVE, CHIASM) to the original RTSTRUCT file
    """
    # convolve to repair it
    convolveKernel = np.array((1, 1, 1))
    convolveKernel = np.expand_dims(convolveKernel, axis=(0, 1))
    for patient in patients:
        CTDicomFolder = os.path.join(sourceFolder1, patient, "data")

        # get SKIN mask from RTStruct
        RTFile = os.path.join(sourceFolder1, patient, "RTStruct.dcm")
        assert os.path.isdir(CTDicomFolder) and os.path.isfile(RTFile)
        dataset = RTStructBuilder.create_from(dicom_series_path=CTDicomFolder, rt_struct_path=RTFile)
        skin_name = "SKIN"
        assert skin_name in dataset.get_roi_names()
        SKIN_mask = dataset.get_roi_mask_by_name(skin_name)  # (y, x, z)
        # SKIN_mask = convolve(SKIN_mask, convolveKernel, mode="same")
        # SKIN_mask = SKIN_mask >= 2

        example_nrrd_file = os.path.join(sourceFolder2, patient, "RTSTRUCT.nrrd")
        segDict = readSegFile(example_nrrd_file)  # (z, y, x)
        segDict = {a: np.transpose(b, axes=(1, 2, 0)) for a, b in segDict.items()}
        structs = {b: c for a, c in segDict.items() if criterion(b:=a.replace(" ", ""))}
        # print(list(structs.keys()), "\n")

        # merge PTV
        PTVGroups = {}
        for name, mask in structs.items():
            if "ptv" in name.lower() or "ctv" in name.lower():
                # name is a ptv
                dose = "".join(a for a in name if a.isdigit())
                dose = eval(dose)
                if dose not in PTVGroups:
                    PTVGroups[dose] = [mask]
                else:
                    PTVGroups[dose].append(mask)
        for dose in PTVGroups:
            localList = PTVGroups[dose]
            if len(localList) == 1:
                PTVGroups[dose] = localList[0]
            else:
                PTVGroups[dose] = np.logical_or(*localList)
        
        PTVGroups = {"PTV{}".format(dose): mask for dose, mask in PTVGroups.items()}
        structsOAR = {a: b for a, b in structs.items() if "ptv" not in a.lower() and "ctv" not in a.lower()}
        structsOAR[skin_name] = SKIN_mask

        # Merge PTVGroups and structsOAR into a single dict
        MaskDict = PTVGroups.copy()
        for a, b in structsOAR.items():
            assert a not in MaskDict
            MaskDict[a] = b
        # MaskDict = maskPolish(MaskDict)
        
        RTOutput = RTStructBuilder.create_new(dicom_series_path=CTDicomFolder)
        for name, mask in MaskDict.items():
            if name in ConvergeReverse:
                name = ConvergeReverse[name]
            RTOutput.add_roi(
                mask = mask,
                color = hex_to_rgb(colorMap[name]),
                name = name)
        RTFile = os.path.join(sourceFolder2, patient, "RTExtend.dcm")
        RTOutput.save(RTFile)
        break


def criterion(name):
    return ("Trans" not in name) and ("+" not in name) \
        and (name not in exclude) and name != "SKIN"


def readSegFile(file: str):
    seg, header = nrrd.read(file)
    seg = np.transpose(seg, axes=(0, 3, 2, 1))
    result = {}
    idx = 0
    while True:
        keyRoot = "Segment{}_".format(idx)
        nameKey = keyRoot + "Name"
        layerKey = keyRoot + "Layer"
        labelValueKey = keyRoot + "LabelValue"
        if nameKey not in header:
            break
        name = header[nameKey]
        layer = int(header[layerKey])
        labelValue = int(header[labelValueKey])
        mask = seg[layer, :, :, :] == labelValue
        result[name] = mask
        idx += 1
    return result


def hex_to_rgb(hex_color):
    """Converts a color from hexadecimal format to RGB."""
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]


def packData():
    destFolder = os.path.join(sourceFolder2, "RTFull")
    if not os.path.isdir(destFolder):
        os.mkdir(destFolder)
    for patient in patients:
        CTFile = os.path.join(sourceFolder1, patient, "data", "*.dcm")
        RTFile = os.path.join(sourceFolder2, patient, "RTExtend.dcm")
        patientFolder = os.path.join(destFolder, patient)
        if not os.path.isdir(patientFolder):
            os.mkdir(patientFolder)
        command = "cp {} {}".format(CTFile, patientFolder)
        os.system(command)
        command = "cp {} {}".format(RTFile, patientFolder)
        os.system(command)


def viewExtend():
    folder = "/data/qifan/projects/FastDoseWorkplace/TCIAAdd/RTFull"
    target = os.path.join(folder, "002View")
    if not os.path.isdir(target):
        os.mkdir(target)
    patientFolder = os.path.join(folder, "002")
    RTFile = os.path.join(patientFolder, "RTExtend.dcm")
    dataset = RTStructBuilder.create_from(patientFolder, RTFile)
    structs = dataset.get_roi_names()
    for name in structs:
        try:
            mask = dataset.get_roi_mask_by_name(name)
            print("{} {}".format(name, np.sum(mask)))
        except:
            print("{} error".format(name))


def maskPolish(maskDict):
    """
    This function polishes the mask, to remove 
    """
    result = {}
    thresh = 0.1
    for name, mask in maskDict.items():
        labeled_array, num_features = label(mask)
        component_sizes = np.bincount(labeled_array.ravel())
        component_sizes = component_sizes[1:]
        component_sizes = [(a+1, component_sizes[a])
            for a in range(len(component_sizes))]
        component_sizes.sort(key=lambda a: a[1], reverse=True)
        MostSignificant = component_sizes[0][1]
        threshValue = thresh * MostSignificant
        component_sizes = [a for a, b in component_sizes if b>threshValue]
        maskPolished = np.isin(mask, component_sizes)
        
        voxels_org = np.sum(mask)
        voxels_after = np.sum(maskPolished)
        print("{} original: {}, post-polishing: {}".format(name, voxels_org, voxels_after))
        result[name] = maskPolished
    return result


if __name__ == "__main__":
    StructsInit()
    addAnnotationToDicom()
    # packData()
    # viewExtend()