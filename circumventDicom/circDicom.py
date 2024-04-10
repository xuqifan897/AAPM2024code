import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from rt_utils import RTStructBuilder
from skimage import transform, measure
import json

def VerifySliceOrder():
    """
    My assumption is that only the slice order in the z direction is flipped, otherwise kept the same
    """
    dicomFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/HN_dicom/000000"
    densityFile = "/data/qifan/projects/FastDoseWorkplace/CORTTune/HeadNeck/prep_output/density.raw"
    densityDim = [180, 193, 193]

    if False:
        dicomViewFolder = "/data/qifan/projects/FastDoseWorkplace/circDicom/dicomView"
        if not os.path.isdir(dicomViewFolder):
            os.mkdir(dicomViewFolder)
        files = os.listdir(dicomFolder)
        fileOrder = []
        for file in files:
            dicomPath = os.path.join(dicomFolder, file)
            dataset = pydicom.dcmread(dicomPath)
            if (dataset.Modality == "CT"):
                InstanceNumber = int(dataset.InstanceNumber)
                pixel_array = dataset.pixel_array
                fileOrder.append((InstanceNumber, pixel_array))
        fileOrder.sort(key = lambda a: a[0])
        for i in range(len(fileOrder)):
            outputFile = os.path.join(dicomViewFolder, "{:03d}.png".format(i))
            plt.imsave(outputFile, fileOrder[i][1])
            print(outputFile)
    if True:
        density = np.fromfile(densityFile, dtype=np.float32)
        density = np.reshape(density, densityDim)
        densityFolder = "/data/qifan/projects/FastDoseWorkplace/circDicom/densityView"
        if not os.path.isdir(densityFolder):
            os.mkdir(densityFolder)
        for i in range(density.shape[0]):
            file = os.path.join(densityFolder, "{:03d}.png".format(i))
            plt.imsave(file, density[i, :, :])
            print(file)


def prepHN():
    """
    This function prepares the binary array files for the head-and-neck case
    """
    caseFolder = "/data/qifan/projects/FastDoseWorkplace/circDicom/HeadNeck"
    dicomFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/HN_dicom/000000"
    resolutionNew = np.array((2.5, 2.5, 2.5))
    if not os.path.isdir(caseFolder):
        os.mkdir(caseFolder)
        
    files = os.listdir(dicomFolder)
    fileOrder = []
    dimension = None
    datatype = None
    sliceThickness = None
    pixelSpacing = None
    for file in files:
        dicomPath = os.path.join(dicomFolder, file)
        dataset = pydicom.dcmread(dicomPath)
        if (dataset.Modality == "CT"):
            InstanceNumber = int(dataset.InstanceNumber)
            pixel_array = dataset.pixel_array
            fileOrder.append((InstanceNumber, pixel_array))
            if dimension is None:
                dimension = pixel_array.shape
                datatype = pixel_array.dtype
                sliceThickness = dataset.SliceThickness
                pixelSpacing = dataset.PixelSpacing
    fileOrder.sort(key = lambda a: a[0])
    
    densityShape = np.array((len(fileOrder), dimension[0], dimension[1]))
    densityResolution = np.array((float(sliceThickness), float(pixelSpacing[0]), float(pixelSpacing[1])))
    densityArray = np.zeros(densityShape, dtype=datatype)
    index = 0
    for InstanceNumber, array in fileOrder:
        densityArray[index, :, :] = array
        index += 1
    
    densityDimension = densityShape * densityResolution
    newDimension = densityDimension / resolutionNew
    newDimension = np.round(newDimension)
    newDimension = newDimension.astype(int)
    
    densityArray = densityArray.astype(float)
    densityArray = transform.resize(densityArray, newDimension)
    densityArray = densityArray.astype(datatype)
    densityArray = np.flip(densityArray, axis=0)

    densityPath = "/data/qifan/projects/FastDoseWorkplace/circDicom/HeadNeck/density_raw.bin"
    densityArray.tofile(densityPath)
    print(densityShape, densityArray.shape, densityArray.dtype, np.max(densityArray))


def verifyHNDensity():
    """
    This function verifies the correctness of the head-and-neck density generated above
    """
    masterFolder = "/data/qifan/projects/FastDoseWorkplace/circDicom/HeadNeck"
    sourceFile = os.path.join(masterFolder, "density_raw.bin")
    sourceShape = (108, 193, 193)
    imageFolder = os.path.join(masterFolder, "densityView")
    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)

    densityArray = np.fromfile(sourceFile, dtype=np.uint16)
    densityArray = np.reshape(densityArray, sourceShape)
    for i in range(sourceShape[0]):
        file = os.path.join(imageFolder, "{:03d}.png".format(i))
        plt.imsave(file, densityArray[i, :, :])
        print(file)


def prepHN_masks():
    shape_source = (90, 512, 512)
    shape_target = (108, 193, 193)
    
    sourceFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/HN_binary"
    targetFolder = "/data/qifan/projects/FastDoseWorkplace/circDicom/HeadNeck/InputMask"
    if not os.path.isdir(targetFolder):
        os.mkdir(targetFolder)
    
    shape_org = (67, 160, 160)
    files = os.listdir(sourceFolder)
    PTV_crop_mask = None
    for file in files:
        sourcePath = os.path.join(sourceFolder, file)
        sourceArray = np.fromfile(sourcePath, dtype=np.uint8)
        sourceArray = np.reshape(sourceArray, shape_org)
        sourceArray = np.transpose(sourceArray, axes=(0, 2, 1))
        sourceArray = sourceArray.astype(float)
        sourceArray = transform.resize(sourceArray, shape_target)
        sourceArray = sourceArray > 0.5
        sourceArray = sourceArray.astype(np.uint8)
        sourceArray = np.flip(sourceArray, axis=0)
        targetPath = os.path.join(targetFolder, file)
        sourceArray.tofile(targetPath)
        print(targetPath)

        if file == "PTV70.bin":
            PTV_crop_mask = sourceArray
    
    print("preparing PTV_crop")
    PTV_crop_mask[:56, :, :] = 0
    targetPath = os.path.join(targetFolder, "PTV_crop.bin")
    PTV_crop_mask.tofile(targetPath)
    print(targetPath)


def verifyHNMask():
    """
    This function verifies the generated mask files
    """
    if False:
        bodyFile = "/data/qifan/projects/FastDoseWorkplace/circDicom" \
            "/HeadNeck/InputMask/External.bin"
        shape = (108, 193, 193)
        bodyMask = np.fromfile(bodyFile, dtype=np.uint8)
        bodyMask = np.reshape(bodyMask, shape)
        
        imageFolder = "/data/qifan/projects/FastDoseWorkplace/circDicom/HeadNeck/maskView"
        if not os.path.isdir(imageFolder):
            os.mkdir(imageFolder)
        for i in range(shape[0]):
            file = os.path.join(imageFolder, "{:03d}.png".format(i))
            plt.imsave(file, bodyMask[i, :, :])
            print(file)

    if True:
        maskFolder = "/data/qifan/projects/FastDoseWorkplace/circDicom/HeadNeck/InputMask"
        densityFile = "/data/qifan/projects/FastDoseWorkplace/circDicom/HeadNeck/density_raw.bin"
        imageFolder = "/data/qifan/projects/FastDoseWorkplace/circDicom/maskView"

        shape = (108, 193, 193)
        density = np.fromfile(densityFile, np.uint16)
        density = np.reshape(density, shape)

        files = os.listdir(maskFolder)
        structs = [a.split(".")[0] for a in files]
        numStructs = len(files)
        masks = {}
        for struct, file in zip(structs, files):
            if "PTV" in struct and struct != "PTV_crop":
                continue
            path = os.path.join(maskFolder, file)
            density_ = np.fromfile(path, dtype=np.uint8)
            density_ = np.reshape(density_, shape)
            masks[struct] = density_

        colors = np.linspace(0, 1, numStructs)
        color_map = plt.get_cmap()
        colors = [color_map(a) for a in colors]
        if not os.path.isdir(imageFolder):
            os.mkdir(imageFolder)
        
        for i in range(shape[0]):
            densitySlice = density[i, :, :]
            plt.imshow(densitySlice, cmap="gray")
            for j, entry in enumerate(masks.items()):
                name, mask = entry
                maskSlice = mask[i, :, :]
                contours = measure.find_contours(maskSlice)
                color = colors[j]
                initial = True
                for contour in contours:
                    if initial:
                        plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                        initial = False
                    else:
                        plt.plot(contour[:, 1], contour[:, 0], color=color)
            plt.legend()
            imageFile = os.path.join(imageFolder, "{:03d}.png".format(i))
            plt.savefig(imageFile)
            plt.clf()
            print(imageFile)


def prepLiver():
    LiverFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/Liver_dicom"
    resolutionNew = np.array((2.5, 2.5, 2.5))
    ctFiles = []
    rtFile = None
    files = os.listdir(LiverFolder)
    PixelSpacing = None
    SliceThickness = None
    SliceDimension = None
    datatype = None
    for file in files:
        path = os.path.join(LiverFolder, file)
        dataset = pydicom.dcmread(path)
        if dataset.Modality == "CT":
            InstanceNumber = int(dataset.InstanceNumber)
            ctFiles.append((InstanceNumber, dataset.pixel_array))
            if PixelSpacing is None:
                PixelSpacing = dataset.PixelSpacing
                SliceThickness = dataset.SliceThickness
                SliceDimension = dataset.pixel_array.shape
                datatype = dataset.pixel_array.dtype
        else:
            rtFile = path
        print(file)
    ctFiles.sort(key = lambda a: a[0])
    sourceShape = np.array((len(ctFiles), SliceDimension[0], SliceDimension[1]))
    sourceRes = np.array((float(SliceThickness), float(PixelSpacing[0]), float(PixelSpacing[1])))
    density = np.zeros(sourceShape, dtype=datatype)
    for i in range(sourceShape[0]):
        density[i, :, :] = ctFiles[i][1]
    
    densityDimension = sourceShape * sourceRes
    targetShape = densityDimension / resolutionNew
    density = density.astype(dtype=float)
    density = transform.resize(density, targetShape)
    density = density.astype(datatype)
    density = np.flip(density, axis=0)
    density[density < 0] = 0

    densityPath = "/data/qifan/projects/FastDoseWorkplace/circDicom/Liver/density_raw.bin"
    density.tofile(densityPath)
    print(sourceShape, density.shape, density.dtype, np.max(density))


def verifyLiverDensity():
    """
    This function verifies the correctness of the head-and-neck density generated above
    """
    masterFolder = "/data/qifan/projects/FastDoseWorkplace/circDicom/Liver"
    sourceFile = os.path.join(masterFolder, "density_raw.bin")
    sourceShape = (168, 260, 260)
    imageFolder = os.path.join(masterFolder, "densityView")
    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)

    densityArray = np.fromfile(sourceFile, dtype=np.uint16)
    densityArray = np.reshape(densityArray, sourceShape)
    for i in range(sourceShape[0]):
        file = os.path.join(imageFolder, "{:03d}.png".format(i))
        plt.imsave(file, densityArray[i, :, :])
        print(file)


def prepLiver_masks():
    liverFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/Liver_dicom"
    targetFolder = "/data/qifan/projects/FastDoseWorkplace/circDicom/Liver/InputMask"
    if not os.path.isdir(targetFolder):
        os.mkdir(targetFolder)
    
    rtFile = os.path.join(liverFolder, "liverRT.dcm")
    targetShape = (168, 260, 260)
    RTStruct = RTStructBuilder.create_from(dicom_series_path=liverFolder, rt_struct_path=rtFile)
    names = RTStruct.get_roi_names()
    PTVname = "PTV"
    assert PTVname in names, "PTV is not included in the structures"
    PTVmask = RTStruct.get_roi_mask_by_name(PTVname)
    for name in names:
        mask = RTStruct.get_roi_mask_by_name(name)
        if name != "PTV" and name != "Skin":
            mask[PTVmask] = 0
        mask = np.transpose(mask, axes=(2, 0, 1))
        mask = mask.astype(float)
        mask = transform.resize(mask, targetShape)
        mask = mask > 0.5
        mask = mask.astype(np.uint8)
        file = os.path.join(targetFolder, "{}.bin".format(name))
        mask.tofile(file)
        print(file)


def verifyLiverMask():
    maskFolder = "/data/qifan/projects/FastDoseWorkplace/circDicom/Liver/InputMask"
    densityFile = "/data/qifan/projects/FastDoseWorkplace/circDicom/Liver/density_raw.bin"
    imageFolder = "/data/qifan/projects/FastDoseWorkplace/circDicom/Liver/maskView"

    shape = (168, 260, 260)
    density = np.fromfile(densityFile, np.uint16)
    density = np.reshape(density, shape)

    files = os.listdir(maskFolder)
    structs = [a.split(".")[0] for a in files]
    numStructs = len(files)
    masks = {}
    for struct, file in zip(structs, files):
        # if "PTV" in struct and struct != "PTV_crop":
        #     continue
        path = os.path.join(maskFolder, file)
        density_ = np.fromfile(path, dtype=np.uint8)
        density_ = np.reshape(density_, shape)
        masks[struct] = density_

    colors = np.linspace(0, 1, numStructs)
    color_map = plt.get_cmap()
    colors = [color_map(a) for a in colors]
    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)
    
    for i in range(shape[0]):
        densitySlice = density[i, :, :]
        plt.imshow(densitySlice, cmap="gray")
        for j, entry in enumerate(masks.items()):
            name, mask = entry
            maskSlice = mask[i, :, :]
            contours = measure.find_contours(maskSlice)
            color = colors[j]
            initial = True
            for contour in contours:
                if initial:
                    plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                    initial = False
                else:
                    plt.plot(contour[:, 1], contour[:, 0], color=color)
        plt.legend()
        imageFile = os.path.join(imageFolder, "{:03d}.png".format(i))
        plt.savefig(imageFile)
        plt.clf()
        print(imageFile)


def prepProstate():
    caseFolder = "/data/qifan/projects/FastDoseWorkplace/circDicom/Prostate"
    dicomFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_dicom"
    resolutionNew = np.array((2.5, 2.5, 2.5))
    if not os.path.isdir(caseFolder):
        os.mkdir(caseFolder)
        
    files = os.listdir(dicomFolder)
    fileOrder = []
    dimension = None
    datatype = None
    sliceThickness = None
    pixelSpacing = None
    for file in files:
        dicomPath = os.path.join(dicomFolder, file)
        dataset = pydicom.dcmread(dicomPath)
        if (dataset.Modality == "CT"):
            InstanceNumber = int(dataset.InstanceNumber)
            pixel_array = dataset.pixel_array
            fileOrder.append((InstanceNumber, pixel_array))
            if dimension is None:
                dimension = pixel_array.shape
                datatype = pixel_array.dtype
                sliceThickness = dataset.SliceThickness
                pixelSpacing = dataset.PixelSpacing
    fileOrder.sort(key = lambda a: a[0])
    
    densityShape = np.array((len(fileOrder), dimension[0], dimension[1]))
    densityResolution = np.array((float(sliceThickness), float(pixelSpacing[0]), float(pixelSpacing[1])))
    densityArray = np.zeros(densityShape, dtype=datatype)
    index = 0
    for InstanceNumber, array in fileOrder:
        densityArray[index, :, :] = array
        index += 1
    
    densityDimension = densityShape * densityResolution
    newDimension = densityDimension / resolutionNew
    newDimension = np.round(newDimension)
    newDimension = newDimension.astype(int)
    
    densityArray = densityArray.astype(float)
    densityArray = transform.resize(densityArray, newDimension)
    densityArray = densityArray.astype(datatype)
    densityArray = np.flip(densityArray, axis=0)

    densityPath = os.path.join(caseFolder, "density_raw.bin")
    densityArray.tofile(densityPath)
    print(densityShape, densityArray.shape, densityArray.dtype, np.max(densityArray))


def prepProstateMasks():
    shape_source = (90, 184, 184)
    shape_target = (108, 221, 221)
    
    sourceFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_binary"
    targetFolder = "/data/qifan/projects/FastDoseWorkplace/circDicom/Prostate/InputMask"
    if not os.path.isdir(targetFolder):
        os.mkdir(targetFolder)
    
    shape_org = shape_source
    files = os.listdir(sourceFolder)
    PTVFile = "PTV_68.bin"
    PTVPath = os.path.join(sourceFolder, PTVFile)
    PTVMask = np.fromfile(PTVPath, dtype=np.uint8)
    PTVMask = np.reshape(PTVMask, shape_org)
    print("PTVMask: {}".format(np.sum(PTVMask)))
    to_continue = ["BODY.bin", "PTV_68.bin", "PTV_56.bin", "prostate_bed.bin"]
    for file in files:
        if file == "density.bin":
            continue
        sourcePath = os.path.join(sourceFolder, file)
        sourceArray = np.fromfile(sourcePath, dtype=np.uint8)
        sourceArray = np.reshape(sourceArray, shape_org)
        if file not in to_continue:
            before = np.sum(sourceArray)
            sourceArray[PTVMask>0] = 0
            after = np.sum(sourceArray)
            print(sourcePath, "Trim from {} to {}".format(before, after))
        else:
            print(sourcePath)
        sourceArray = np.transpose(sourceArray, axes=(0, 2, 1))
        sourceArray = sourceArray.astype(float)
        sourceArray = transform.resize(sourceArray, shape_target)
        sourceArray = sourceArray > 0.5
        sourceArray = sourceArray.astype(np.uint8)
        sourceArray = np.flip(sourceArray, axis=0)
        targetPath = os.path.join(targetFolder, file)
        sourceArray.tofile(targetPath)


def verifyProstateMask():
    maskFolder = "/data/qifan/projects/FastDoseWorkplace/circDicom/Prostate/InputMask"
    densityFile = "/data/qifan/projects/FastDoseWorkplace/circDicom/Prostate/density_raw.bin"
    imageFolder = "/data/qifan/projects/FastDoseWorkplace/circDicom/Prostate/maskView"

    shape = (108, 221, 221)
    density = np.fromfile(densityFile, np.uint16)
    density = np.reshape(density, shape)

    files = os.listdir(maskFolder)
    structs = [a.split(".")[0] for a in files]
    numStructs = len(files)

    masks = {}
    for struct, file in zip(structs, files):
        path = os.path.join(maskFolder, file)
        density_ = np.fromfile(path, dtype=np.uint8)
        density_ = np.reshape(density_, shape)
        masks[struct] = density_

    colors = np.linspace(0, 1, numStructs)
    color_map = plt.get_cmap()
    colors = [color_map(a) for a in colors]
    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)
    
    for i in range(shape[0]):
        densitySlice = density[i, :, :]
        plt.imshow(densitySlice, cmap="gray")
        for j, entry in enumerate(masks.items()):
            name, mask = entry
            maskSlice = mask[i, :, :]
            contours = measure.find_contours(maskSlice)
            color = colors[j]
            initial = True
            for contour in contours:
                if initial:
                    plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                    initial = False
                else:
                    plt.plot(contour[:, 1], contour[:, 0], color=color)
        plt.legend()
        imageFile = os.path.join(imageFolder, "{:03d}.png".format(i))
        plt.savefig(imageFile)
        plt.clf()
        print(imageFile)


def StructuresInfoGen():
    """
    This file generates structureInfo for three patients
    """
    if False:
        # head and neck case
        masterFolder = "/data/qifan/projects/FastDoseWorkplace/circDicom/HeadNeck"
        maskFolder = os.path.join(masterFolder, "InputMask")
        structures = [a.split(".")[0] for a in os.listdir(maskFolder)]
        print(structures)
        to_exclude = ["CTV56", "CTV63", "GTV", "PTV56", "PTV63", "PTV70"]

        # to ensure that the structures above are included
        flagValid = True
        for a in to_exclude:
            if a not in structures:
                flagValid = False
                break
        if not flagValid:
            print("structure {} not contained in the structure list".format(a))
        structures = [a for a in structures if a not in to_exclude]
        print(structures)

        if False:
            # to verify the correctness
            StructureVerify(os.path.join(masterFolder, "density_raw.bin"),
                            os.path.join(masterFolder, "InputMask"), structures,
                            (108, 193, 193), os.path.join(masterFolder, "maskView"))
        # generate structures File
        PTVname = "PTV_crop"
        BODYname = "External"
        assert PTVname in structures and BODYname in structures, \
            "{} or {} not contained in structures".format(PTVname, BODYname)
        OARs = structures.copy()
        OARs.remove(PTVname)
        OARs.remove(BODYname)
        OARs.insert(0, BODYname)
        content = {"prescription": 20,
                   "ptv": PTVname,
                   "oar": OARs}
        content_str = json.dumps(content, indent=4)
        structuresFile = os.path.join(masterFolder, "structures.json")
        with open(structuresFile, "w") as json_file:
            json_file.write(content_str)
        print(content_str)

    if False:
        # Liver case
        masterFolder = "/data/qifan/projects/FastDoseWorkplace/circDicom/Liver"
        maskFolder = os.path.join(masterFolder, "InputMask")
        structures = [a.split(".")[0] for a in os.listdir(maskFolder)]
        print(structures)

        to_exclude = ["DoseFalloff"]

        # to ensure that the structures above are included
        flagValid = True
        for a in to_exclude:
            if a not in structures:
                flagValid = False
                break
        if not flagValid:
            print("structure {} not contained in the structure list".format(a))
        structures = [a for a in structures if a not in to_exclude]
        print(structures)

        if False:
            StructureVerify(os.path.join(masterFolder, "density_raw.bin"),
                            os.path.join(masterFolder, "InputMask"), structures,
                            (168, 260, 260), os.path.join(masterFolder, "maskView"))
        PTVname = "PTV"
        BODYname = "Skin"
        OARs = structures.copy()
        OARs.remove(PTVname)
        OARs.remove(BODYname)
        OARs.insert(0, BODYname)
        content = {"prescription": 20,
                   "ptv": PTVname,
                   "oar": OARs}
        content_str = json.dumps(content, indent=4)
        structuresFile = os.path.join(masterFolder, "structures.json")
        with open(structuresFile, "w") as f:
            f.write(content_str)
        print(content_str)
    
    if True:
        # Prostate case
        masterFolder = "/data/qifan/projects/FastDoseWorkplace/circDicom/Prostate"
        maskFolder = os.path.join(masterFolder, "InputMask")
        structures = [a.split(".")[0] for a in os.listdir(maskFolder)]
        print(structures)
        to_exclude = ["prostate_bed", "PTV_56"]
        flagValid = True
        for a in to_exclude:
            if a not in structures:
                flagValid = False
                break
        if not flagValid:
            print("structure {} not contained in the structure list".format(a))
        structures = [a for a in structures if a not in to_exclude]
        print(structures)

        if False:
            StructureVerify(os.path.join(masterFolder, "density_raw.bin"),
                            os.path.join(masterFolder, "InputMask"), structures,
                            (108, 221, 221), os.path.join(masterFolder, "maskView"))
        
        PTV_name = "PTV_68"
        BODY_name = "BODY"
        OARs = structures.copy()
        OARs.remove(PTV_name)
        OARs.remove(BODY_name)
        OARs.insert(0, BODY_name)
        content = {
            "prescription": 20,
            "ptv": PTV_name,
            "oar": OARs}
        content_str = json.dumps(content, indent=4)
        structuresFile = os.path.join(masterFolder, "structures.json")
        with open(structuresFile, "w") as f:
            f.write(content_str)
        print(content_str)


def StructureVerify(densityFile: str, maskFolder: str,
                    
    structures: list[str], shape: tuple[int, int, int],
    imageFolder: str) -> None:
    density = np.fromfile(densityFile, dtype=np.uint16)
    density = np.reshape(density, shape)
    masks = {}
    for name in structures:
        file = os.path.join(maskFolder, "{}.bin".format(name))
        maskArray = np.fromfile(file, dtype=np.uint8)
        maskArray = np.reshape(maskArray, shape)
        masks[name] = maskArray
    
    numStructs = len(structures)
    values = np.linspace(0, 1, numStructs)
    colorMap = plt.get_cmap()
    colors = [colorMap(a) for a in values]

    if not os.path.isdir(imageFolder):
        os.mkdir(imageFolder)
    
    for i in range(shape[0]):
        densitySlice = density[i, :, :]
        plt.imshow(densitySlice, cmap="gray")
        for j, entry in enumerate(masks.items()):
            name, mask = entry
            color = colors[j]
            mask = mask[i, :, :]
            contours = measure.find_contours(mask)
            initial = True
            for contour in contours:
                if initial:
                    plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                    initial = False
                else:
                    plt.plot(contour[:, 1], contour[:, 0], color=color)
        plt.legend()
        imageFile = os.path.join(imageFolder, "{:03d}.png".format(i))
        plt.savefig(imageFile)
        plt.clf()
        print(imageFile)


def getInfo():
    exampleFile = "/data/qifan/projects/FastDoseWorkplace/PlanTune/Liver_dicom/anon14.dcm"
    dataset = pydicom.dcmread(exampleFile)
    print("RescaleSlope", dataset.RescaleSlope)
    print("RescaleIntercept", dataset.RescaleIntercept)


if __name__ == '__main__':
    # VerifySliceOrder()
    # prepHN()
    # verifyHNDensity()
    # prepHN_masks()
    # verifyHNMask()
    # prepLiver()
    # verifyLiverDensity()
    # prepLiver_masks()
    # verifyLiverMask()
    # prepProstate()
    # prepProstateMasks()
    # verifyProstateMask()
    # StructuresInfoGen()
    getInfo()