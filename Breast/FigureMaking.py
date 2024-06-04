import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import measure, transform
from scipy.interpolate import RegularGridInterpolator
import h5py
import json
import pydicom

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
rootFolder = "/data/qifan/projects/FastDoseWorkplace/Breast"
PTVNameList = [("01079255", "d_eval_PTV", "Body"), ("12908474Right", "PTV_PBI_R", "External"),
    ("33445300", "PTV_PBI_L", "External"), ("42028819", "PTV_PBI_L", "External"),
    ("42179205", "PTV_PBI_L", "External"), ("50023133", "PTV_PBI_L", "External"),
    ("55252173", "PTV_PBI_L", "External"), ("77881484", "PTV_PBI_L", "External")]

def drawDVHComp():
    FiguresFolder = os.path.join(rootFolder, "Figures")
    if not os.path.isdir(FiguresFolder):
        os.mkdir(FiguresFolder)
    for patientName, PTVName, BodyName in PTVNameList:
        patientFolder = os.path.join(rootFolder, patientName)
        expFolder = os.path.join(patientFolder, "expFolder")
        prep_output = os.path.join(expFolder, "prep_output")

        dimensionFile = os.path.join(prep_output, "dimension.txt")
        with open(dimensionFile, "r") as f:
            lines = f.readlines()
        doseShape = lines[0]
        doseShape = doseShape.split(" ")
        doseShape = [int(a) for a in doseShape]
        doseShape.reverse()
        doseShape = tuple(doseShape)

        densityFile = os.path.join(prep_output, "density.raw")
        density = np.fromfile(densityFile, dtype=np.float32)
        density = np.reshape(density, doseShape)

        doseFile = os.path.join(expFolder, "plan1", "dose.bin")
        doseArray = np.fromfile(doseFile, dtype=np.float32)
        doseArray = np.reshape(doseArray, doseShape)

        structuresFile = os.path.join(prep_output, "roi_list.h5")
        structures = getStructures(structuresFile)
        print("Structures Loaded")

        dicomDoseArrayFile = os.path.join(FiguresFolder, "RTDose{}.npy".format(patientName))
        dicomDoseArray = np.load(dicomDoseArrayFile)

        PTVMask = [a[1] for a in structures if a[0] == PTVName]
        assert len(PTVMask) == 1, "None or more than one PTV is found"
        PTVMask = PTVMask[0]
        PTVDoseOpt = doseArray[PTVMask]
        threshOpt = np.percentile(PTVDoseOpt, 10)
        print(threshOpt)
        doseArray = doseArray * 30 / threshOpt
        PTVDoseDicom = dicomDoseArray[PTVMask]
        threshDicom = np.percentile(PTVDoseDicom, 10)
        print(threshDicom)
        dicomDoseArray = dicomDoseArray * 30 / threshDicom

        # bring PTV forward
        exclude = [PTVName, BodyName, "RingStructure"]
        structures = [a for a in structures if a[0] not in exclude]
        structures.insert(0, (PTVName, PTVMask))

        # draw DVH
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, entry in enumerate(structures):
            color = colors[i]
            name, mask = entry
            mask = mask > 0
            optDose = doseArray[mask]
            optDose = np.sort(optDose)
            optDose = np.insert(optDose, 0, 0)
            dicomDose = dicomDoseArray[mask]
            dicomDose = np.sort(dicomDose)
            dicomDose = np.insert(dicomDose, 0, 0)
            y_axis = np.linspace(100, 0, dicomDose.size)
            plt.plot(optDose, y_axis, color=color, linestyle="-", label=name)
            plt.plot(dicomDose, y_axis, color=color, linestyle="--")
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
        plt.title("DVH comparison patient {}".format(patientName))
        plt.xlabel("Dose (Gy)")
        plt.ylabel("Percentage")
        plt.tight_layout()
        figFile = os.path.join(FiguresFolder, "Patient{}DVHComp.png".format(patientName))
        plt.savefig(figFile)
        print(figFile)


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


def doseGen():
    """
    This file generates the resized dose file
    """
    FiguresFolder = os.path.join(rootFolder, "Figures")
    if not os.path.isdir(FiguresFolder):
        os.mkdir(FiguresFolder)
    for patient, _, _ in PTVNameList:
        PatientFolder = os.path.join(rootFolder, patient)
        dimensionFile = os.path.join(PatientFolder, "expFolder", "prep_output", "dimension.txt")
        with open(dimensionFile, "r") as f:
            lines = f.readlines()
        doseShape = lines[0]
        doseShape = doseShape.split(" ")
        doseShape = [int(a) for a in doseShape]
        doseShape.reverse()
        doseShape = tuple(doseShape)

        CTFolder = os.path.join(PatientFolder, "dicom")
        dicomDoseFile = os.path.join(PatientFolder, "RTDose.dcm")
        dicomDoseArray = examineDose(dicomDoseFile, CTFolder)
        dicomDoseArray = np.flip(dicomDoseArray, axis=0)
        dicomDoseArray = transform.resize(dicomDoseArray, doseShape)

        dicomDoseArrayFile = os.path.join(FiguresFolder, "RTDose{}.npy".format(patient))
        np.save(dicomDoseArrayFile, dicomDoseArray)
        print(dicomDoseArrayFile)


def drawDoseWash():
    FiguresFolder = os.path.join(rootFolder, "Figures")
    ImgExpFolder = os.path.join(FiguresFolder, "Experiment")
    if not os.path.isdir(ImgExpFolder):
        os.mkdir(ImgExpFolder)
    ImageHeight = 80
    AxialWidth = 80
    CoronalWidth = 80
    SagittalWidth = 60
    DoseThresh = 1.0
    for patientName, PTVName, BodyName in PTVNameList:
        patientFolder = os.path.join(rootFolder, patientName)

        dimensionFile = os.path.join(patientFolder, "expFolder", "prep_output", "dimension.txt")
        with open(dimensionFile, "r") as f:
            lines = f.readlines()
        doseShape = lines[0]
        doseShape = doseShape.split(" ")
        doseShape = [int(a) for a in doseShape]
        doseShape.reverse()
        doseShape = tuple(doseShape)

        dicomDoseArrayFile = os.path.join(FiguresFolder, "RTDose{}.npy".format(patientName))
        dicomDoseArray = np.load(dicomDoseArrayFile)

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

        PTVEntry = [a for a in structures if a[0] == PTVName]
        assert len(PTVEntry) == 1, "More or no PTV found"
        PTVMask = PTVEntry[0][1]
        PTVMask = PTVMask > 0

        BodyEntry = [a for a in structures if a[0] == BodyName]
        assert len(BodyEntry) == 1, "More or no Body found"
        BodyMask = BodyEntry[0][1]
        BodyMask = BodyMask > 0
        BodyMaskComplement = np.logical_not(BodyMask)

        # bring PTV forward
        exclude = [PTVName, BodyName, "RingStructure"]
        structures = [a for a in structures if a[0] not in exclude]
        structures.insert(0, (PTVName, PTVMask))

        # normalize
        PTVDose_dicom = dicomDoseArray[PTVMask]
        thresh_dicom = np.percentile(PTVDose_dicom, 5)
        dicomDoseArray = dicomDoseArray * 30 / thresh_dicom

        PTVDose_opt = optDoseArray[PTVMask]
        thresh_opt = np.percentile(PTVDose_opt, 5)
        optDoseArray = optDoseArray * 30 / thresh_opt
        optDoseArray[BodyMaskComplement] = 0


        # Get the coordinates of the PTV
        PTVCentroid = FindCentroid3D(PTVMask)
        BodySlice = BodyMask[PTVCentroid[0], :, :]
        AxialDensity = densityArray[PTVCentroid[0], :, :]
        AxialOptDose = optDoseArray[PTVCentroid[0], :, :]
        AxialDicomDose = dicomDoseArray[PTVCentroid[0], :, :]
        MasksSlice = [(a[0], a[1][PTVCentroid[0], :, :]) for a in structures]
        file = os.path.join(ImgExpFolder, "{}AxialOpt.png".format(patientName))
        DrawSlice(AxialDensity, AxialOptDose, MasksSlice, BodySlice, ImageHeight, AxialWidth, file)
        
        file = os.path.join(ImgExpFolder, "{}AxialClinic.png".format(patientName))
        DrawSlice(AxialDensity, AxialDicomDose, MasksSlice, BodySlice, ImageHeight, AxialWidth, file)


        BodySlice = BodyMask[:, PTVCentroid[1], :]
        CoronalDensity = densityArray[:, PTVCentroid[1], :]
        CoronalOptDose = optDoseArray[:, PTVCentroid[1], :]
        CoronalDicomDose = dicomDoseArray[:, PTVCentroid[1], :]
        MasksSlice = [(a[0], a[1][:, PTVCentroid[1], :]) for a in structures]
        file = os.path.join(ImgExpFolder, "{}CoronalOpt.png".format(patientName))
        DrawSlice(CoronalDensity, CoronalOptDose, MasksSlice, BodySlice, ImageHeight, CoronalWidth, file)

        file = os.path.join(ImgExpFolder, "{}CoronalClinic.png".format(patientName))
        DrawSlice(CoronalDensity, CoronalDicomDose, MasksSlice, BodySlice, ImageHeight, CoronalWidth, file)
        

        BodySlice = BodyMask[:, :, PTVCentroid[2]]
        SagittalDensity = densityArray[:, :, PTVCentroid[2]]
        SagittalOptDose = optDoseArray[:, :, PTVCentroid[2]]
        SagittalDicomDose = dicomDoseArray[:, :, PTVCentroid[2]]
        MasksSlice = [(a[0], a[1][:, :, PTVCentroid[2]]) for a in structures]
        file = os.path.join(ImgExpFolder, "{}SagittalOpt.png".format(patientName))
        DrawSlice(SagittalDensity, SagittalOptDose, MasksSlice, BodySlice, ImageHeight, SagittalWidth, file)

        file = os.path.join(ImgExpFolder, "{}SagittalClinic.png".format(patientName))
        DrawSlice(SagittalDensity, SagittalDicomDose, MasksSlice, BodySlice, ImageHeight, SagittalWidth, file)


def DrawSlice(DensitySlice, DoseSlice, MasksSlice,
    BodySlice, height, width, file):
    DoseThresh = 3.0

    MaskCentroid = FindCentroid2D(BodySlice)
    Dim0LowerBound = MaskCentroid[0] - height
    Dim0LowerBound = max(0, Dim0LowerBound)
    Dim0HigherBound = MaskCentroid[0] + height
    Dim0HigherBound = min(Dim0HigherBound, BodySlice.shape[0])

    Dim1LowerBound = MaskCentroid[1] - width
    Dim1LowerBound = max(0, Dim1LowerBound)
    Dim1HigherBound = MaskCentroid[1] + width
    Dim1HigherBound = min(Dim1HigherBound, BodySlice.shape[1])

    RealHeight = Dim0HigherBound - Dim0LowerBound
    RealWidth = Dim1HigherBound - Dim1LowerBound

    DensitySlice = DensitySlice[Dim0LowerBound: Dim0HigherBound,
            Dim1LowerBound: Dim1HigherBound]
    DoseSlice = DoseSlice[Dim0LowerBound: Dim0HigherBound,
            Dim1LowerBound: Dim1HigherBound]
    fig, ax = plt.subplots(figsize=(RealWidth/10, RealHeight/10))
    ax.imshow(DensitySlice, cmap="gray", vmin=0, vmax=2000)

    for k, entry in enumerate(MasksSlice):
        name, mask = entry
        color = colors[k]
        maskSlice = mask[Dim0LowerBound: Dim0HigherBound,
            Dim1LowerBound: Dim1HigherBound]
        contours = measure.find_contours(maskSlice)
        initial = True
        for contour in contours:
            if initial:
                plt.plot(contour[:, 1], contour[:, 0], color=color, label=name)
                initial = False
            else:
                plt.plot(contour[:, 1], contour[:, 0], color=color)
    ax.imshow(DoseSlice, cmap="jet", vmin=0, vmax=40, alpha=(DoseSlice>DoseThresh)*0.3)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(file)
    plt.close(fig)
    plt.clf()


def FindCentroid3D(array):
    zWeight = np.sum(array, axis=(1, 2))
    zAxis = np.arange(zWeight.size)
    zCoord = np.sum(zWeight * zAxis) / np.sum(zWeight)
    zCoord = int(zCoord)

    yWeight = np.sum(array, axis=(0, 2))
    yAxis = np.arange(yWeight.size)
    yCoord = np.sum(yWeight * yAxis) / np.sum(yWeight)
    yCoord = int(yCoord)

    xWeight = np.sum(array, axis=(0, 1))
    xAxis = np.arange(xWeight.size)
    xCoord = np.sum(xWeight * xAxis) / np.sum(xWeight)
    xCoord = int(xCoord)

    return (zCoord, yCoord, xCoord)


def FindCentroid2D(array):
    yWeight = np.sum(array, axis=1)
    yAxis = np.arange(yWeight.size)
    yCoord = np.sum(yWeight * yAxis) / np.sum(yWeight)
    yCoord = int(yCoord)

    xWeight = np.sum(array, axis=0)
    xAxis = np.arange(xWeight.size)
    xCoord = np.sum(xWeight * xAxis) / np.sum(xWeight)
    xCoord = int(xCoord)

    return (yCoord, xCoord)


def DrawColorBar():
    """
    This function draws the colorbar with black background
    """
    fig, ax = plt.subplots(figsize=(1, 6))
    ax.axis('off')
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=0, vmax=40)
    cbar_ax = fig.add_axes([0.05, 0.05, 0.3, 0.85])  # Adjust as needed
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar_ax.set_facecolor('black')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    cbar.set_label('Dose (Gy)', color='white', fontsize=15)
    cbar.ax.tick_params(labelsize=12, labelcolor='white')
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    fig.patch.set_facecolor('black')

    FigureFolder = "/data/qifan/projects/FastDoseWorkplace/Breast/Figures"
    ExpFigureFolder = os.path.join(FigureFolder, "Experiment")
    FigureFile = os.path.join(ExpFigureFolder, "Colorbar.png")
    plt.savefig(FigureFile)
    plt.close(fig)
    plt.clf()


def ImagesGroup():
    """
    This function groups images generated in function "drawDoseWash"
    """
    FigureFolder = "/data/qifan/projects/FastDoseWorkplace/Breast/Figures"
    ExpFigureFolder = os.path.join(FigureFolder, "Experiment")

    ColorBarFile = os.path.join(ExpFigureFolder, "Colorbar.png")
    ColorBarImage = plt.imread(ColorBarFile)
    for patient, _, _ in PTVNameList:
        group = "Opt"
        AxialFile = os.path.join(ExpFigureFolder, "{}Axial{}.png".format(patient, group))
        CoronalFile = os.path.join(ExpFigureFolder, "{}Coronal{}.png".format(patient, group))
        SagittalFile = os.path.join(ExpFigureFolder, "{}Sagittal{}.png".format(patient, group))
        AxialImage = plt.imread(AxialFile)
        CoronalImage = plt.imread(CoronalFile)
        CoronalImage = np.flip(CoronalImage, axis=0)
        SagittalImage = plt.imread(SagittalFile)
        SagittalImage = np.flip(SagittalImage, axis=0)
        
        MaxHeight = max(AxialImage.shape[0], CoronalImage.shape[0], SagittalImage.shape[0])
        AxialImage = ImagePadding(AxialImage, MaxHeight)
        CoronalImage = ImagePadding(CoronalImage, MaxHeight)
        SagittalImage = ImagePadding(SagittalImage, MaxHeight)
        DoseWashOpt = np.concatenate((AxialImage, CoronalImage, SagittalImage), axis=1)
        

        group = "Clinic"
        AxialFile = os.path.join(ExpFigureFolder, "{}Axial{}.png".format(patient, group))
        CoronalFile = os.path.join(ExpFigureFolder, "{}Coronal{}.png".format(patient, group))
        SagittalFile = os.path.join(ExpFigureFolder, "{}Sagittal{}.png".format(patient, group))
        AxialImage = plt.imread(AxialFile)
        CoronalImage = plt.imread(CoronalFile)
        CoronalImage = np.flip(CoronalImage, axis=0)
        SagittalImage = plt.imread(SagittalFile)
        SagittalImage = np.flip(SagittalImage, axis=0)
        
        MaxHeight = max(AxialImage.shape[0], CoronalImage.shape[0], SagittalImage.shape[0])
        AxialImage = ImagePadding(AxialImage, MaxHeight)
        CoronalImage = ImagePadding(CoronalImage, MaxHeight)
        SagittalImage = ImagePadding(SagittalImage, MaxHeight)
        DoseWashClinic = np.concatenate((AxialImage, CoronalImage, SagittalImage), axis=1)


        DoseWashFull = np.concatenate((DoseWashOpt, DoseWashClinic), axis=0)
        FullHeight = DoseWashFull.shape[0]
        ColorBarImageHeight = ColorBarImage.shape[0]
        ColorBarImageWidth = ColorBarImage.shape[1]
        WidthNew = FullHeight * ColorBarImageWidth / ColorBarImageHeight
        WidthNew = int(WidthNew)
        ShapeNew = (FullHeight, WidthNew, 4)
        ColorBarImageLocal = transform.resize(ColorBarImage, ShapeNew)

        DoseWashFull = np.concatenate((DoseWashFull, ColorBarImageLocal), axis=1)
        figureFile = os.path.join(FigureFolder, "{}DoseWash.png".format(patient))
        plt.imsave(figureFile, DoseWashFull)
        print(figureFile)


def ImagePadding(Image, Height):
    if Image.shape[0] == Height:
        return Image
    topping = (Height - Image.shape[0]) / 2
    topping = int(topping)
    ending = topping + Image.shape[0]
    canvas = np.zeros((Height, Image.shape[1], 4), dtype=Image.dtype)
    canvas[:, :, -1] = 1
    canvas[topping:ending, :, :] = Image
    return canvas


if __name__ == "__main__":
    # drawDVHComp()
    # doseGen()
    # drawDoseWash()
    # ImagesGroup()
    # DrawColorBar()
    ImagesGroup()