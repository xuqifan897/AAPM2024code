import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import measure, transform
from scipy import signal, ndimage
import json

def RTview():
    numPatients = 5
    targetFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas"
    rootFolder = "/mnt/shengdata1/qifan/Pancreas"
    CTFolder = os.path.join(rootFolder, "CT")
    PTVFolder = os.path.join(rootFolder, "PTV")
    RTFolder = os.path.join(rootFolder, "RTSTRUCT_TotalSegmentor")

    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
    LUT_PTV = {1: "Bowel_sm_planCT", 2: "Lung_L", 3: "Lung_R", 4: "PTV", 5: "Stomach_duo_planCT"}
    CTViewFolder = os.path.join(targetFolder, "CTView")
    if not os.path.isdir(CTViewFolder):
        os.mkdir(CTViewFolder)
    for i in range(1, numPatients):
        CTPath = os.path.join(CTFolder, "Pancreas_{:03d}_0000.nii.gz".format(i+1))
        dataset = nib.load(CTPath)
        CTArray = dataset.get_fdata()
        CTArray -= np.min(CTArray)
        CTArray = np.transpose(CTArray, axes=(2, 1, 0))
        CTArray_max = np.max(CTArray)
        nSlices = CTArray.shape[0]

        PTVPath = os.path.join(PTVFolder, "PancreasRT_{:03d}.nii.gz".format(i+1))
        dataset = nib.load(PTVPath)
        PTVArray = dataset.get_fdata()
        PTVArray = np.transpose(PTVArray, axes=(2, 1, 0))
        ViewFolder = os.path.join(CTViewFolder, "Patient{:03d}".format(i+1))
        if not os.path.isdir(ViewFolder):
            os.mkdir(ViewFolder)
        for j in range(nSlices):
            CTSlice = CTArray[j, :, :]
            plt.imshow(CTSlice, cmap="gray", vmin=0, vmax=CTArray_max)
            PTVSlice = PTVArray[j, :, :]
            if np.max(PTVSlice) > 1e-4:
                # slice with annotation
                for k, entry in enumerate(LUT_PTV.items()):
                    number, name = entry
                    color = colors[k]
                    binaryMask = np.abs(PTVSlice - number) < 1e-4
                    contours = measure.find_contours(binaryMask)
                    initial = True
                    for contour in contours:
                        if initial:
                            plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                            initial = False
                        else:
                            plt.plot(contour[:, 1], contour[:, 0], color=color)
                plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
                plt.tight_layout()
            file = os.path.join(ViewFolder, "{:03d}.png".format(j))
            plt.savefig(file)
            plt.clf()
            print(file)


def totalSegView():
    numPatients = 5
    targetFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas"
    rootFolder = "/mnt/shengdata1/qifan/Pancreas"
    CTFolder = os.path.join(rootFolder, "CT")
    PTVFolder = os.path.join(rootFolder, "PTV")
    RTFolder = os.path.join(rootFolder, "RTSTRUCT_TotalSegmentator")

    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
    current_folder = os.path.dirname(os.path.abspath(__file__))
    LUTPath = os.path.join(current_folder, "lookUpTable.txt")
    with open(LUTPath, "r") as f:
        lines = f.readlines()
    LUT_totalseg = {}
    for line in lines:
        line = line.split("|")
        number = line[0]
        number = int(number)
        name = line[1]
        name = name.replace(" ", "")
        LUT_totalseg[number] = name
    
    imageFolder = os.path.join(targetFolder, "totalSegView")
    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)
    for i in range(numPatients):
        CTPath = os.path.join(CTFolder, "Pancreas_{:03d}_0000.nii.gz".format(i+1))
        dataset = nib.load(CTPath)
        CTArray = dataset.get_fdata()
        CTArray -= np.min(CTArray)
        CTArray = np.transpose(CTArray, axes=(2, 1, 0))
        CTArray_max = np.max(CTArray)
        nSlices = CTArray.shape[0]

        totalSegPath = os.path.join(RTFolder, "Pancreas_{:03d}.nii.gz".format(i+1))
        dataset = nib.load(totalSegPath)
        SegArray = dataset.get_fdata()
        SegArray = np.transpose(SegArray, axes=(2, 1, 0))
        SegArray = SegArray.astype(int)
        unique_elements, counts = np.unique(SegArray, return_counts=True)
        unique_elements = unique_elements[1:]  # remove the background 0

        patientImageFolder = os.path.join(imageFolder, "Patient{:03d}".format(i+1))
        if not os.path.isdir(patientImageFolder):
            os.mkdir(patientImageFolder)

        for i in range(nSlices):
            CTSlice = CTArray[i, :, :]
            plt.figure(figsize=(10, 6))
            plt.imshow(CTSlice, cmap="gray", vmin=0, vmax=CTArray_max)
            SegSlice = SegArray[i, :, :]
            if np.max(SegSlice) > 0:
                for element in unique_elements:
                    mask = SegSlice == element
                    if np.max(mask) == 0:
                        continue
                    name = LUT_totalseg[element]
                    contours = measure.find_contours(mask)
                    initial = True
                    for contour in contours:
                        if initial:
                            plt.plot(contour[:, 1], contour[:, 0], color=colors[element], label=name)
                            initial = False
                        else:
                            plt.plot(contour[:, 1], contour[:, 0], color=colors[element])
                plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            file = os.path.join(patientImageFolder, "{:03d}.png".format(i))
            plt.savefig(file)
            plt.clf()
            print(file)


def dataGen():
    """
    This function generates the binary CT array as well as the mask array
    """
    targetRoot = "/data/qifan/projects/FastDoseWorkplace/Pancreas"
    sourceRoot = "/mnt/shengdata1/qifan/Pancreas"
    sourceCT = os.path.join(sourceRoot, "CT")
    sourcePTV = os.path.join(sourceRoot, "PTV")
    sourceRT = os.path.join(sourceRoot, "RTSTRUCT_TotalSegmentator")
    numPatients = 5
    target_resolution = 2.5  # mm

    current_folder = os.path.dirname(os.path.abspath(__file__))
    LUTPath = os.path.join(current_folder, "lookUpTable.txt")
    with open(LUTPath, "r") as f:
        lines = f.readlines()
    LUT_totalseg = {}
    for line in lines:
        line = line.split("|")
        number = line[0]
        number = int(number)
        name = line[1]
        name = name.replace(" ", "")
        LUT_totalseg[number] = name
    
    merge_dict = {
        "kidney_right": ["kidney_right", "adrenal_gland_right", "kidney_cyst_left"],
        "kidney_left": ["kidney_left", "adrenal_gland_left", "kidney_cyst_right"],
        "lung_left": ["lung_upper_lobe_left", "lung_lower_lobe_left"],
        "lung_right": ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
        "spine": None,
        "ribs": None
    }
    spine = ["vertebrae_C{}".format(i) for i in range(1, 8)] + \
        ["vertebrae_T{}".format(i) for i in range(1, 13)] + \
        ["vertebrae_L{}".format(i) for i in range(1, 6)] + ["vertebrae_S1"]
    ribs = ["rib_left_{}".format(i) for i in range(1, 13)] + \
        ["rib_right_{}".format(i) for i in range(1, 13)]
    merge_dict["spine"] = spine
    merge_dict["ribs"] = ribs
    merge_dict_reverse = {}
    for name, subs in merge_dict.items():
        for small in subs:
            merge_dict_reverse[small] = name

    if False:
        structures_org = list(LUT_totalseg.values())
        structures_updated = set()
        for name in structures_org:
            if name in merge_dict_reverse:
                category = merge_dict_reverse[name]
                structures_updated.add(category)
            else:
                structures_updated.add(name)

    reduced_structures = {}
    for number, struct in LUT_totalseg.items():
        if struct in merge_dict_reverse:
            target = merge_dict_reverse[struct]
        else:
            target = struct
        if target in reduced_structures:
            reduced_structures[target].append(number)
        else:
            reduced_structures[target] = [number]
    
    to_keep = ["spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach",
               "lung_left", "lung_right", "esophagus", "trachea", "small_bowel", "duodenum",
               "colon", "urinary_bladder", "spine", "heart", "aorta", "spinal_cord"]
    reduced_structures = {a: reduced_structures[a] for a in to_keep}
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())

    for i in range(numPatients):
        patientTargetFolder = os.path.join(targetRoot, "Patient{:03d}".format(i+1))
        if not os.path.isdir(patientTargetFolder):
            os.makedirs(patientTargetFolder)

        sourceCTFile = os.path.join(sourceCT, "Pancreas_{:03d}_0000.nii.gz".format(i+1))
        CTDataset = nib.load(sourceCTFile)
        CTArray = CTDataset.get_fdata()
        CTArray -= np.min(CTArray)
        voxelSize = CTDataset.header.get_zooms()
        CTArray = np.transpose(CTArray, axes=(2, 1, 0))
        voxelSize = np.flip(voxelSize)
        dim_org = np.array(CTArray.shape)
        size_org = dim_org * voxelSize
        dim_new = size_org / target_resolution
        dim_new = dim_new.astype(int)
        CTArray = transform.resize(CTArray, dim_new)
        CTArray = CTArray.astype(np.uint16)
        
        sourcePTVFile = os.path.join(sourcePTV, "PancreasRT_{:03d}.nii.gz".format(i+1))
        PTVMask = nib.load(sourcePTVFile).get_fdata()
        PTVMask = np.transpose(PTVMask, axes=(2, 1, 0))
        PTVMask = np.abs(PTVMask - 4) < 1e-4
        PTVMask = transform.resize(PTVMask, dim_new)

        bodyMask = np.zeros(dim_new, dtype=bool)
        bodyMask[:, 120: 380, 50:450] = 1
        
        sourceRTFile = os.path.join(sourceRT, "Pancreas_{:03d}.nii.gz".format(i+1))
        RTMask = nib.load(sourceRTFile).get_fdata()
        RTMask = np.transpose(RTMask, axes=(2, 1, 0))
        RTMask = RTMask.astype(int)
        Masks = {}
        for struct, indices in reduced_structures.items():
            local_mask = np.isin(RTMask, indices)
            if np.sum(local_mask) == 0:
                continue
            local_mask = transform.resize(local_mask, dim_new)
            Masks[struct] = local_mask

        exclude = np.logical_not(PTVMask)
        
        normal_masks = []
        for key, mask in Masks.items():
            mask = np.logical_and(mask, exclude)
            normal_masks.append((key, mask))
        normal_masks.insert(0, ("PTV", PTVMask))
        
        if False:
            nSlices = CTArray.shape[0]
            for j in range(nSlices):
                CTSlice = CTArray[j, :, :]
                plt.figure(figsize=(10, 6))
                plt.imshow(CTSlice, cmap="gray")
                for k, entry in enumerate(normal_masks):
                    name, maskArray = entry
                    maskSlice = maskArray[j, :, :]
                    if np.sum(maskSlice) == 0:
                        continue
                    color = colors[k]
                    contours = measure.find_contours(maskSlice)
                    initial = True
                    for contour in contours:
                        if initial:
                            plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                            initial = False
                        else:
                            plt.plot(contour[:, 1], contour[:, 0], color=color)
                plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
                file = os.path.join(patientTargetFolder, "view", "{:03d}.png".format(j))
                plt.savefig(file)
                plt.clf()
                print(file)

        # write masks to file
        CTFile = os.path.join(patientTargetFolder, "density_raw.bin")
        CTArray.tofile(CTFile)
        print(CTFile)
        maskFolder = os.path.join(patientTargetFolder, "InputMask")
        if not os.path.isdir(maskFolder):
            os.mkdir(maskFolder)
        for name, mask in normal_masks:
            file = os.path.join(maskFolder, "{}.bin".format(name))
            mask = mask.astype(np.uint8)
            mask.tofile(file)
            print(file)
        print("\n")


def refDoseGen():
    targetFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas"
    sourceRoot = "/mnt/shengdata1/qifan/Pancreas"
    sourceDOSE = os.path.join(sourceRoot, "DOSE")
    sourceCT = os.path.join(sourceRoot, "CT")
    numPatients = 5
    target_resolution = 2.5

    for i in range(numPatients):
        patientTargetFolder = os.path.join(targetFolder, "Patient{:03d}".format(i+1))
        doseDataset = os.path.join(sourceDOSE, "Pancreas_{:03d}_0001.nii.gz".format(i+1))
        doseDataset = nib.load(doseDataset)
        doseArray = doseDataset.get_fdata()  # (x, y, z)
        doseVoxelSize = doseDataset.header.get_zooms()  # (x, y, z)

        ctDataset = os.path.join(sourceCT, "Pancreas_{:03d}_0000.nii.gz".format(i+1))
        ctDataset = nib.load(ctDataset)
        ctArray = ctDataset.get_fdata()
        ctVoxelSize = ctDataset.header.get_zooms()
        assert np.linalg.norm(np.array(doseVoxelSize) - np.array(ctVoxelSize)) < 1e-4
        assert doseArray.shape == ctArray.shape
        
        doseArray = np.transpose(doseArray, axes=(2, 1, 0))  # (z, y, x)
        voxelSize = np.flip(doseVoxelSize)   # (z, y, x)
        size_org = np.array(doseArray.shape) * voxelSize
        dim_new = size_org / target_resolution
        dim_new = dim_new.astype(int)
        doseArray = transform.resize(doseArray, dim_new).astype(np.float32)
        
        doseFile = os.path.join(patientTargetFolder, "doseRef.bin")
        doseArray.tofile(doseFile)
        print(doseFile)


def generate_body_mask():
    targetFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas"
    sourceRoot = "/mnt/shengdata1/qifan/Pancreas"
    sourceCT = os.path.join(sourceRoot, "CT")
    numPatients = 5
    target_resolution = 2.5
    for i in range(numPatients):
        CTFile = os.path.join(sourceCT, "Pancreas_{:03d}_0000.nii.gz".format(i+1))
        CTDataset = nib.load(CTFile)
        CTArray = CTDataset.get_fdata()
        CTArray -= np.min(CTArray)
        voxelSize = CTDataset.header.get_zooms()
        CTArray = np.transpose(CTArray, axes=(2, 1, 0))
        voxelSize = np.flip(voxelSize)
        dim_org = np.array(CTArray.shape)
        size_org = dim_org * voxelSize
        dim_new = size_org / target_resolution
        dim_new = dim_new.astype(int)
        CTArray = transform.resize(CTArray, dim_new)
        CTArray = CTArray.astype(np.uint16)
        threshold = 800
        Salient = CTArray > threshold

        # for every slice, apply connected components analysis and hole filling
        nSlices = Salient.shape[0]
        BinarySalient = np.zeros(Salient.shape, bool)
        filterKernel = np.ones((4, 4), dtype=int)
        for j in range(1, nSlices):
            slice = Salient[j, :, :]
            labeled_image, num_labels = measure.label(slice, background=0, return_num=True)
            props = measure.regionprops(labeled_image)
            props_sorted_by_area = sorted(props, key=lambda x: x.area, reverse=True)
            sorted_labels_by_area = [prop.label for prop in props_sorted_by_area]
            primaryLabel = sorted_labels_by_area[0]
            primarySlice = labeled_image == primaryLabel
            primarySlice = primarySlice.astype(int)
            primarySliceConvolve = signal.convolve(primarySlice, filterKernel, "same")
            primarySliceConvolve = primarySliceConvolve > 4
            
            # filter again
            labeled_image, num_labels = measure.label(primarySliceConvolve, background=0, return_num=True)
            props = measure.regionprops(labeled_image)
            props_sorted_by_area = sorted(props, key=lambda x: x.area, reverse=True)
            primaryLabel = props_sorted_by_area[0].label
            primarySliceConvolve = labeled_image == primaryLabel

            # fill holes
            primarySliceConvolve = ndimage.binary_fill_holes(primarySliceConvolve)
            BinarySalient[j, :, :] = primarySliceConvolve
        
        if False:
            patientFolder = os.path.join(targetFolder, "Patient{:03d}".format(i+1))
            maskFolder = os.path.join(patientFolder, "bodyMask")
            if not os.path.isdir(maskFolder):
                os.mkdir(maskFolder)
            nSlices = Salient.shape[0]
            for j in range(nSlices):
                bodySlice = BinarySalient[j, :, :]
                file = os.path.join(maskFolder, "{:03d}.png".format(j))
                plt.imsave(file, bodySlice)
                print(file)
            print("\n")
        
        patientFolder = os.path.join(targetFolder, "Patient{:03d}".format(i+1))
        patientMaskFolder = os.path.join(patientFolder, "InputMask")
        bodyMaskFile = os.path.join(patientMaskFolder, "SKIN.bin")
        BinarySalient = BinarySalient.astype(np.uint8)
        BinarySalient.tofile(bodyMaskFile)
        print(bodyMaskFile)


def viewShape():
    """
    This function prints the array shape for all the CT slices
    """
    rootFolder = "/mnt/shengdata1/qifan/Pancreas"
    CTFolder = os.path.join(rootFolder, "CT")
    numPatients = 5
    targetRes = 2.5
    for i in range(numPatients):
        file = os.path.join(CTFolder, "Pancreas_{:03d}_0000.nii.gz".format(i+1))
        dataset = nib.load(file)
        shape = dataset.get_fdata().shape
        voxelSize = dataset.header.get_zooms()
        shape = np.flip(shape)
        voxelSize = np.flip(voxelSize)
        size_org = shape * voxelSize
        dim_new = size_org / targetRes
        dim_new = dim_new.astype(int)
        print(dim_new)


def viewTotalMask():
    rootFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas"
    numPatients = 5
    targetDose = 20
    vmaxShow = 25
    shapes = [
        [160, 220, 220],
        [182, 200, 200],
        [111, 280, 280],
        [155, 240, 240],
        [128, 200, 200]
    ]
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())

    for i in range(numPatients):
        patientFolder = os.path.join(rootFolder, "Patient{:03d}".format(i+1))
        MaskFolder = os.path.join(patientFolder, "InputMask")
        files = os.listdir(MaskFolder)
        names = [a.split(".")[0] for a in files]
        names.remove("PTV")
        names.remove("SKIN")
        names.insert(0, "SKIN")
        names.insert(0, "PTV")
        structures = []
        shape = shapes[i]
        for name in names:
            file = os.path.join(MaskFolder, "{}.bin".format(name))
            array = np.fromfile(file, dtype=np.uint8)
            array = np.reshape(array, shape)
            structures.append((name, array))
        
        densityFile = os.path.join(patientFolder, "density_raw.bin")
        densityArray = np.fromfile(densityFile, dtype=np.uint16)
        densityArray = np.reshape(densityArray, shape)

        doseFile = os.path.join(patientFolder, "doseRef.bin")
        doseArray = np.fromfile(doseFile, dtype=np.float32)
        doseArray = np.reshape(doseArray, shape)
        # normalize
        PTVMask = structures[0][1].astype(bool)
        ptvDose = doseArray[PTVMask]
        doseThresh = np.percentile(ptvDose, 10)
        doseArray *= targetDose / doseThresh

        viewFolder = os.path.join(patientFolder, "view")
        if not os.path.isdir(viewFolder):
            os.mkdir(viewFolder)

        nSlices = shape[0]
        for j in range(nSlices):
            CTSlice = densityArray[j, :, :]
            doseSlice = doseArray[j, :, :]
            plt.figure(figsize=(10, 6))
            plt.imshow(CTSlice, cmap="gray")
            plt.imshow(doseSlice, cmap="jet", vmin=0, vmax=vmaxShow, alpha=(doseSlice>1)*0.3)
            for k, entry in enumerate(structures):
                name, array = entry
                color = colors[k]
                maskSlice = array[j, :, :]
                contours = measure.find_contours(maskSlice)
                initial = True
                for contour in contours:
                    if initial:
                        plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                        initial = False
                    else:
                        plt.plot(contour[:, 1], contour[:, 0], color=color)
            plt.legend(loc="upper right", bbox_to_anchor=(1.05, 1))
            plt.colorbar()
            plt.tight_layout()
            file = os.path.join(viewFolder, "{:03d}.png".format(j))
            plt.savefig(file)
            plt.clf()
            print(file)


def gen_structures_file():
    generalFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas"
    numPatients = 5
    for i in range(numPatients):
        patientFolder = os.path.join(generalFolder, "Patient{:03d}".format(i+1))
        maskFolder = os.path.join(patientFolder, "InputMask")
        structures = os.listdir(maskFolder)
        structures = [a.split(".")[0] for a in structures]
        structures.remove("PTV")
        structures.remove("SKIN")
        structures.insert(0, "SKIN")
        content = {
            "prescription": 20,
            "ptv": "PTV",
            "oar": structures
        }
        content = json.dumps(content, indent=4)
        print(content)
        file = os.path.join(patientFolder, "structures.json")
        with open(file, "w") as f:
            f.write(content)


if __name__ == "__main__":
    # RTview()
    # totalSegView()
    # dataGen()
    # refDoseGen()
    # generate_body_mask()
    # viewShape()
    viewTotalMask()
    # gen_structures_file()