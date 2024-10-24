import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rt_utils import RTStructBuilder
import pydicom
from skimage import transform, measure
import json
from scipy.interpolate import RegularGridInterpolator

rootFolder = "/mnt/shengdata1/qifan/TCIAPancreas/" \
    "Pancreatic-CT-CBCT-SEG_v2_20220823/Pancreatic-CT-CBCT-SEG"
target1 = "/data/qifan/projects/FastDoseWorkplace/Pancreas"
target2 = os.path.join(target1, "plansSIB")
numPatients = 5
resTarget = 2.5  # mm

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
# (CTFolder, RTFolder)
folderMap = []

def dvhPtv():
    """
    Plot the PTV without normalization
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
        if True:
            # normalization
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


def folderMapInit():
    global folderMap
    for i in range(numPatients):
        patientNameTarget = "Patient{:03d}".format(i+1)
        patientNameRoot = "Pancreas-CT-CB_{:03d}".format(i+1)
        patientFolderRoot = os.path.join(rootFolder, patientNameRoot)
        subFolders = os.listdir(patientFolderRoot)
        if len(subFolders) == 2:
            subFolders = [a for a in subFolders if 'CRANE' in a]
        patientFolder1 = os.path.join(patientFolderRoot, subFolders[0])
        domains = os.listdir(patientFolder1)

        # find the planning CT folder
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
        folderMap.append((planCT, rtFile))


def PTVGen():
    """
    This function generates the PTV mask, whose resolution is aligned with the CT
    """
    excludeFromOrg = ["duodenum", "lung_left", "lung_right", "PTV", "small_bowel", "stomach"]
    doseFolder = "/mnt/shengdata1/qifan/Pancreas/DoseAligned"
    for i in range(numPatients):
        CTFolder, RTFile = folderMap[i]
        rtStruct = RTStructBuilder.create_from(dicom_series_path=CTFolder, rt_struct_path=RTFile)
        names = rtStruct.get_roi_names()
        masks = {}
        roiShape = None
        for name in names:
            maskLocal = rtStruct.get_roi_mask_by_name(name)
            masks[name] = maskLocal
            if roiShape is None:
                roiShape = maskLocal.shape
            else:
                assert roiShape == maskLocal.shape
        
        # get the CT voxel resolution
        CTData = []  # (InstanceNumber, ImagePositionPatient)
        PixelSpacing = None
        Rows = None
        Columns = None
        for file in os.listdir(CTFolder):
            file = os.path.join(CTFolder, file)
            dataset = pydicom.dcmread(file)
            if PixelSpacing is None:
                PixelSpacing = dataset.PixelSpacing
                Rows = dataset.Rows
                Columns = dataset.Columns
            else:
                assert PixelSpacing == dataset.PixelSpacing and Rows == dataset.Rows \
                    and Columns == dataset.Columns
            CTData.append((int(dataset.InstanceNumber), dataset.ImagePositionPatient))
        CTData.sort(key=lambda a: a[0])
        sliceThickness = abs(CTData[0][1][2] - CTData[-1][1][2]) / (len(CTData) - 1)

        CTShape = (int(Rows), int(Columns), len(CTData))
        CTRes = (float(PixelSpacing[0]), float(PixelSpacing[1]), sliceThickness)
        assert CTShape == roiShape
        dimNew = np.array(CTShape) * np.array(CTRes) / resTarget
        dimNew = dimNew.astype(int)

        dimRef = os.path.join(target1, "Patient{:03d}".format(i+1),
            "FastDose", "prep_output", "dimension.txt")
        with open(dimRef, "r") as f:
            dimRef = f.readline()
        dimRef = dimRef.replace(" ", ", ")
        dimRef = np.array(eval(dimRef))
        print(dimRef, dimNew)

        for name in masks:
            roi = masks[name].astype(np.float32)
            roi = transform.resize(roi, dimNew)  # (y, x, z) in column major
            roi = np.transpose(roi, axes=(2, 0, 1))  # (z, y, x) in column major
            roi = (roi > 0).astype(np.uint8)
            masks[name] = roi

        # load other structures
        dimensionFlip = (dimNew[2], dimNew[0], dimNew[1])  # (z, y, x)
        target1PatientFolder = os.path.join(target1, "Patient{:03d}".format(i+1))
        target1MaskFolder = os.path.join(target1PatientFolder, "InputMask")
        structs = [a.split(".")[0] for a in os.listdir(target1MaskFolder)]
        for a in excludeFromOrg:
            assert a in structs
        for a in structs:
            if a in excludeFromOrg:
                continue
            maskFile = os.path.join(target1MaskFolder, a + ".bin")
            maskArray = np.fromfile(maskFile, dtype=np.uint8)
            masks[a] = np.reshape(maskArray, dimensionFlip)

        # crop out the PTV
        ptvName = "ROI"
        bodyName = "SKIN"
        ptvMask = masks[ptvName]
        notPtvMask = np.logical_not(ptvMask)
        for name in masks:
            if name in [ptvName, bodyName]:
                continue
            mask = masks[name]
            mask = np.logical_and(mask, notPtvMask)
            mask = mask.astype(np.uint8)
            masks[name] = mask

        target2PatientFolder = os.path.join(target2, 'Patient{:03d}'.format(i+1))
        if not os.path.isdir(target2PatientFolder):
            os.mkdir(target2PatientFolder)
        target2MaskFolder = os.path.join(target2PatientFolder, "InputMask")
        if not os.path.isdir(target2MaskFolder):
            os.mkdir(target2MaskFolder)
        for name, mask in masks.items():
            file = os.path.join(target2MaskFolder, name+".bin")
            mask.tofile(file)

        # also copy the original density
        command = "cp {} {}".format(os.path.join(target1PatientFolder, "density_raw.bin"),
            os.path.join(target2PatientFolder, "density_raw.bin"))
        os.system(command)

        # also copy the original dose
        doseOrg = os.path.join(doseFolder, "Pancreas-CT-CB_{:03d}.npy".format(i+1))
        doseOrg = np.load(doseOrg)  # (y, x, z)  in column major
        doseOrg = np.flip(doseOrg, axis=2)
        dose = transform.resize(doseOrg, dimNew).astype(np.float32)
        dose = np.transpose(dose, axes=(2, 0, 1))  # (z, y, x) in column major
        doseFile = os.path.join(target2PatientFolder, "doseRef.bin")
        dose.tofile(doseFile)
        print(i)


def densityAnatomyDoseCheck():
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        dimension = os.path.join(target1, patientName, "FastDose", "prep_output", "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)  # (x, y, z)
        dimension_flip = np.flip(dimension)  # (z, y, x)

        target2PatientFolder = os.path.join(target2, "Patient{:03d}".format(i+1))
        density = os.path.join(target2PatientFolder, "density_raw.bin")
        density = np.fromfile(density, dtype=np.uint16)
        density = np.reshape(density, dimension_flip)  # (z, y, x)

        dose = os.path.join(target2PatientFolder, "doseRef.bin")
        dose = np.fromfile(dose, dtype=np.float32)
        dose = np.reshape(dose, dimension_flip)  # (z, y, x)

        masks = {}
        maskFolder = os.path.join(target2PatientFolder, "InputMask")
        for a in os.listdir(maskFolder):
            name = a.split(".")[0]
            file = os.path.join(maskFolder, a)
            mask = np.fromfile(file, dtype=np.uint8)
            mask = np.reshape(mask, dimension_flip)  # (z, y, x)
            masks[name] = mask
        
        ptvName = "ROI"
        ptv = masks[ptvName].astype(bool)
        ptvDose = dose[ptv]
        thresh = np.percentile(ptvDose, 10)
        dose *= 20 / thresh
        maxDose = np.max(dose)

        structs = list(masks.keys())
        structs.remove(ptvName)
        structs.sort()
        structs.insert(0, ptvName)
        colorMap = {structs[i]: colors[i] for i in range(len(structs))}

        viewFolder = os.path.join(target2PatientFolder, "View")
        if not os.path.isdir(viewFolder):
            os.mkdir(viewFolder)
        
        nSlices = ptv.shape[0]
        for i in range(nSlices):
            densitySlice = density[i, :, :]
            doseSlice = dose[i, :, :]
            fig, ax = plt.subplots()
            ax.imshow(densitySlice, cmap="gray", vmin=0, vmax=1600)
            doseMap = ax.imshow(doseSlice, cmap="jet", vmin=0, vmax=maxDose, alpha=(maxDose>3)*0.3)
            for name, maskArray in masks.items():
                colorLocal = colorMap[name]
                maskSlice = maskArray[i, :, :]
                if np.any(maskSlice) == 0:
                    continue
                contours = measure.find_contours(maskSlice)
                initial = True
                for contour in contours:
                    if initial:
                        ax.plot(contour[:, 1], contour[:, 0], color=colorLocal, label=name)
                        initial = False
                    else:
                        ax.plot(contour[:, 1], contour[:, 0], color=colorLocal)
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
            fig.colorbar(doseMap, ax=ax, location="left")
            fig.tight_layout()
            file = os.path.join(viewFolder, "{:03d}.png".format(i+1))
            plt.savefig(file)
            plt.close(fig)
            plt.clf()
            print(file)


def jsonGen():
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        target2PatientFolder = os.path.join(target2, patientName)
        FastDoseFolder = os.path.join(target2PatientFolder, "FastDose")
        if not os.path.isdir(FastDoseFolder):
            os.mkdir(FastDoseFolder)
        MaskFolder = os.path.join(target2PatientFolder, "InputMask")
        structures = [a.split(".")[0] for a in os.listdir(MaskFolder)]
        ptvName = "ROI"
        bodyName = "SKIN"
        structures.remove(ptvName)
        structures.remove(bodyName)
        structures.sort()
        structures.insert(0, bodyName)
        content = {
            "prescription": 70,
            "ptv": ptvName,
            "oar": structures
        }
        content = json.dumps(content, indent=4)
        file = os.path.join(FastDoseFolder, "structures.json")
        with open(file, "w") as f:
            f.write(content)
        print(file)


def doseNorm():
    """
    This function is to normalize the reference dose so that D90 is 20 Gy
    """
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        target2PatientFolder = os.path.join(target2, patientName)
        doseInput = os.path.join(target2PatientFolder, "doseRef.bin")
        doseInput = np.fromfile(doseInput, dtype=np.float32)
        roi = os.path.join(target2PatientFolder, "InputMask", "ROI.bin")
        roi = np.fromfile(roi, dtype=np.uint8).astype(bool)
        roiDose = doseInput[roi]
        thresh = np.percentile(roiDose, 10)
        doseOutput = doseInput * 20 / thresh
        doseOutput = doseOutput.astype(np.float32)
        doseOutputFile = os.path.join(target2PatientFolder, "doseNorm.bin")
        doseOutput.tofile(doseOutputFile)
        print(doseOutputFile)


def StructureInfoGen():
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        target2PatientFolder = os.path.join(target2, patientName)
        maskFolder = os.path.join(target2PatientFolder, "InputMask")
        structs = [a.split(".")[0] for a in os.listdir(maskFolder)]
        ptvName = "ROI"
        bodyName = "SKIN"
        assert ptvName in structs and bodyName in structs
        structs.remove(ptvName)
        structs.remove(bodyName)
        structs.sort()
        RingStructure = "RingStructure"
        structs.append(RingStructure)
        content = ["Name,maxWeights,maxDose,minDoseTargetWeights,"
            "minDoseTarget,OARWeights,IdealDose"]
        ptvContent = "ROI,100,60,100,60,NaN,60"  # the dose 60 is dummy, as 
            # the target dose is specified in the optimization
        content.append(ptvContent)
        for struct in structs:
            structLine = "{},0,18,NaN,NaN,5,0".format(struct)
            content.append(structLine)
        content = "\n".join(content)
        contentFile = os.path.join(target2PatientFolder, "FastDose", "StructureInfo.csv")
        with open(contentFile, "w") as f:
            f.write(content)
        print(contentFile)


def paramsCopy():
    """
    This function copies the parameters from plansAngleCorrect to the current folder
    """
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        sourceFile = os.path.join(target1, "plansAngleCorrect", 
            patientName, "FastDose", "params.txt")
        targetFolder = os.path.join(target2, patientName, "FastDose")
        command = "cp {} {}".format(sourceFile, targetFolder)
        os.system(command)


def fpanglesGen(angleRes: float):
    """
    Generate angles with angle resolution angleRes
    """
    # angleRes unit: rad
    eps = 1e-4
    angleList = []
    numTheta = round(np.pi / angleRes)
    for thetaIdx in range(numTheta + 1):
        theta = thetaIdx * angleRes
        phiTotal = 2 * np.pi * np.sin(theta)
        numPhi = int(np.ceil(phiTotal / angleRes))
        if numPhi == 0:
            numPhi = 1
        deltaPhi = 2 * np.pi / numPhi
        for phiIdx in range(numPhi):
            phi = phiIdx * deltaPhi
            sinTheta = np.sin(theta)
            cosTheta = np.cos(theta)
            sinPhi = np.sin(phi)
            cosPhi = np.cos(phi)
            direction = np.array((sinTheta * cosPhi, sinTheta * sinPhi, cosTheta))
            angleList.append((np.array((theta, phiIdx * deltaPhi)), direction))
    return angleList


def centroidCalc(ptv):
    ptv = ptv > 0
    totalVoxels = np.sum(ptv)
    
    ptvShape = ptv.shape
    xScale = np.arange(ptvShape[0])
    xScale = np.expand_dims(xScale, axis=(1, 2))
    xCoord = np.sum(ptv * xScale) / totalVoxels

    yScale = np.arange(ptvShape[1])
    yScale = np.expand_dims(yScale, axis=(0, 2))
    yCoord = np.sum(ptv * yScale) / totalVoxels

    zScale = np.arange(ptvShape[2])
    zScale = np.expand_dims(zScale, axis=(0, 1))
    zCoord = np.sum(ptv * zScale) / totalVoxels

    return np.array((xCoord, yCoord, zCoord))


def direction2VarianIEC(direction: np.array):
    eps = 1e-4
    cosGantry = direction[1]
    sinGantry = np.sqrt(1 - cosGantry ** 2)
    gantry = np.arccos(cosGantry)
    if sinGantry < eps:
        minusCouch = 0
    else:
        sinMinusCouch = direction[2] / sinGantry
        cosMinusCouch = - direction[0] / sinGantry
        cosMinusCouch = np.clip(cosMinusCouch, -1, 1)
        minusCouch = np.arccos(cosMinusCouch)
        if sinMinusCouch < 0:
            minusCouch = - minusCouch
    couch = - minusCouch
    return (gantry, couch)


def validAngleListGen():
    """
    This function generates the number of valid beams for each patient
    """
    eps = 1e-4
    angleResInit = 6 * np.pi / 180
    angleListInit = fpanglesGen(angleResInit)
    numBeamsDesired = 400

    for idx in range(1, numPatients + 1):
        patientName = "Patient{:03d}".format(idx)
        maskFolder = os.path.join(target2, patientName, "InputMask")
        patientTargetFolder = os.path.join(target2, patientName, "FastDose")

        patientDimension = os.path.join(patientTargetFolder, "prep_output", "dimension.txt")
        with open(patientDimension, "r") as f:
            dimension = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)
        dimension_flip = np.flip(dimension)
        
        # load the masks
        ptv = os.path.join(maskFolder, "ROI.bin")
        ptv = np.fromfile(ptv, dtype=np.uint8)
        ptv = np.reshape(ptv, dimension_flip)
        ptv = np.transpose(ptv, axes=(2, 1, 0))
        ptvCentroid = centroidCalc(ptv)

        skin = os.path.join(maskFolder, "SKIN.bin")
        skin = np.fromfile(skin, dtype=np.uint8)
        skin = np.reshape(skin, dimension_flip)
        skin = np.transpose(skin, axes=(2, 1, 0))
        skin = (skin > 0).astype(float)
        # generate the first and bottom slices
        skinProjZ = np.sum(skin, axis=(0, 1))
        idxValid = [i for i in range(skin.shape[2]) if skinProjZ[i] > 0]
        idxMin, idxMax = min(idxValid), max(idxValid)
        sliceBottom = skin[:, :, idxMin].copy()
        sliceBottom = RegularGridInterpolator((np.arange(dimension[0]), np.arange(dimension[1])),
            sliceBottom, bounds_error=False, fill_value=0)
        sliceTop = skin[:, :, idxMax]
        sliceTop = RegularGridInterpolator((np.arange(dimension[0]), np.arange(dimension[1])),
            sliceTop, bounds_error=False, fill_value=0)

        # calculate the list of valid angles
        validAnglesLocal = []
        for j in range(len(angleListInit)):
            angle, direction = angleListInit[j]
            if abs(direction[2]) < eps:
                validAnglesLocal.append((angle, direction))
                continue
            # calculate the intersection with the top slice
            k_value = (idxMax - ptvCentroid[2]) / direction[2]
            if k_value < 0:
                intersectionTop = ptvCentroid + k_value * direction
                intersectionTop = intersectionTop[:2]
                intersectionValue = sliceTop(intersectionTop)
                if intersectionValue < eps:
                    validAnglesLocal.append((angle, direction))
                continue
            
            # calculate the intersection with the bottom slice
            k_value = (idxMin - ptvCentroid[2]) / direction[2]
            if k_value < 0:
                intersectionBottom = ptvCentroid + k_value * direction
                intersectionBottom = intersectionBottom[:2]
                intersectionValue = sliceBottom(intersectionBottom)
                if intersectionValue < eps:
                    validAnglesLocal.append((angle, direction))
        numPreSelect = len(validAnglesLocal)

        # modify the angular resolution to get the desired number of beams
        angleResAdjust = angleResInit * np.sqrt(numPreSelect / numBeamsDesired)
        angleListAdjust = fpanglesGen(angleResAdjust)
        validAnglesAdjust = []
        for j in range(len(angleListAdjust)):
            angle, direction = angleListAdjust[j]
            if abs(direction[2]) < eps:
                validAnglesAdjust.append((angle, direction))
                continue
            # calculate the intersection with the top slice
            k_value = (idxMax - ptvCentroid[2]) / direction[2]
            if k_value < 0:
                intersectionTop = ptvCentroid + k_value * direction
                intersectionTop = intersectionTop[:2]
                intersectionValue = sliceTop(intersectionTop)
                if intersectionValue < eps:
                    validAnglesAdjust.append((angle, direction))
                continue
            
            # calculate the intersection with the bottom slice
            k_value = (idxMin - ptvCentroid[2]) / direction[2]
            if k_value < 0:
                intersectionBottom = ptvCentroid + k_value * direction
                intersectionBottom = intersectionBottom[:2]
                intersectionValue = sliceBottom(intersectionBottom)
                if intersectionValue < eps:
                    validAnglesAdjust.append((angle, direction))
        
        # calculate the gantry and couch angles
        VarianIECList = []
        for i in range(len(validAnglesAdjust)):
            _, direction = validAnglesAdjust[i]
            gantry, phi = direction2VarianIEC(direction)

            # then convert to degrees
            gantryDegree = gantry * 180 / np.pi
            phiDegree = phi * 180 / np.pi
            entry = "{:.4f} {:.4f} {:.4f}".format(gantryDegree, phiDegree, 0)
            VarianIECList.append(entry)
        
        beamList1 = VarianIECList[:200]
        beamList2 = VarianIECList[200:]
        if not os.path.isdir(patientTargetFolder):
            os.mkdir(patientTargetFolder)
        beamList = "\n".join(VarianIECList)
        beamListList = [beamList1, beamList2]
        beamListList = ["\n".join(a) for a in beamListList]
        beamListFile = os.path.join(patientTargetFolder, "beamlist.txt")
        with open(beamListFile, "w") as f:
            f.write(beamList)
        for i in range(len(beamListList)):
            content = beamListList[i]
            file = os.path.join(patientTargetFolder, "beamlist{}.txt".format(i+1))
            with open(file, "w") as f:
                f.write(content)
        print(patientName)


def closerLookAtTheDose():
    """
    Previously, the dose was calculated based on normalization. However, when calculating R50,
    we realized that the absolute value of the dose is of importance
    """
    targetFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB"
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
        # print(planCT, CT_align_slices)

        if False:
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

        doseFolder = None
        for domain in domains:
            domain_lower = domain.lower()
            if "dose" in domain_lower:
                domain_folder = os.path.join(patientFolder1, domain)
                if(len(os.listdir(domain_folder))) == 1:
                    doseFolder = domain_folder
                    break
        assert doseFolder is not None
        doseFile = os.path.join(doseFolder, '1-1.dcm')
        assert os.path.isfile(doseFile)
        doseDataset = pydicom.dcmread(doseFile)
        doseGridScaling = doseDataset.DoseGridScaling
        doseArray = doseDataset.pixel_array
        doseArrayMax = np.max(doseArray)
        doseArrayMaxReal = doseArrayMax * doseGridScaling
        
        # rescale the original dose to the real scale
        doseInput = os.path.join(targetFolder, "Patient{:03d}".format(i+1), "doseRef.bin")
        doseInput = np.fromfile(doseInput, dtype=np.float32)
        doseInputMax = np.max(doseInput)
        doseInput *= doseArrayMaxReal / doseInputMax
        outputFile = os.path.join(targetFolder, "Patient{:03d}".format(i+1), "dosePhysical.bin")
        doseInput.tofile(outputFile)
        print(outputFile)


if __name__ == "__main__":
    # dvhPtv()
    # folderMapInit()
    # PTVGen()
    # densityAnatomyDoseCheck()
    # jsonGen()
    # doseNorm()
    # StructureInfoGen()
    # paramsCopy()
    # validAngleListGen()
    closerLookAtTheDose()