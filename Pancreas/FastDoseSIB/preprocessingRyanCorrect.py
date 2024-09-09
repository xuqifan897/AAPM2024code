import os
import numpy as np

sourceFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/plansSIB"
numPatients = 5

contentHeader = \
    "# ====================================================================\n" \
    "# ==                         OMNI-HEADER                            ==\n" \
    "# ====================================================================\n" \
    "# | Nx Ny Nz           - Dicom Volume Dimensions                     |\n" \
    "# | dx dy dz           - Voxel Size (cm)                             |\n" \
    "# |  x  y  z           - Dicom Volume Start Coords (cm)              |\n" \
    "# |  i  j  k           - Dose Bounding Box Start Indices             |\n" \
    "# | Bx By Bz           - Dose Bounding Box Dimensions                |\n" \
    "# | Rx Ry Rz           - REV Convolution Array Dimensions            |\n" \
    "# | convlat            - Convolution ray lateral spacing (cm)        |\n" \
    "# | convstep           - Convolution step spacing (cm)               |\n" \
    "# | kernel_extent      - Dose kernel radius truncate distance (cm)   |\n" \
    "# | ss_factor          - Terma anti-aliasing (super-sampling) factor |\n" \
    "# | nphi ntheta nradii - CCK Kernel Dimensions                       |\n" \
    "# | penumbra           - beamlet transverse dose spread (cm)         |\n" \
    "# | beam_count         - Number of beams to pick from beam_list.txt  |\n" \
    "# | beam_spectrum      - beam energy spectrum file to use            |\n" \
    "# | target_structure   - Name of selected target contour             |\n" \
    "# | reduce coeff. mat  - M-matrix to A-matrix reduction requested?   |\n" \
    "# ===================================================================\n\n"

class beam:
    def __init__(self) -> None:
        self.azimuth = 0.  # in rad, not degree
        self.zenith = 0.
        self.coll = 0.
        self.sad = 0.
        self.iso_type = "ptv"
        self.iso_loc = "in"
        self.isocenter = np.zeros(3, np.float32)
        self.source = np.zeros(3, np.float32)
        self.orient_type = "auto"
        self.direction = np.zeros(3, np.float32)
        self.beamlet_size = np.zeros(3, np.float32)
        self.fmap_size = np.zeros(3, np.float32)
    
    def reconfigure(self, ) -> None:
        self.azimuth = self.azimuth % (2*np.pi)
        self.zenith = self.zenith % (2*np.pi)
        if (not closeto(self.zenith, 0.) and (closeto(self.azimuth, 0.) or closeto(self.azimuth, np.pi))):
            self.zenith = 0.
            print("WARNING: Beam zenith (couch) angle was set to 0 deg to resolve ambiguity")
        self.source = self.calc_source_from_angles(self.azimuth, self.zenith, self.isocenter, self.sad)
        self.direction = self.calc_dir_from_source(self.isocenter, self.source)
    
    @staticmethod
    def calc_source_from_angles(gantry_rot_rad: float, couch_rot_rad: float, iso: np.ndarray, sad: float):
        src = np.array((0., -sad, 0.))
        result = inverseRotateBeamAtOriginRHS(src, gantry_rot_rad, couch_rot_rad, 0.) + iso
        return result
    
    @staticmethod
    def calc_dir_from_source(iso: np.ndarray, source: np.ndarray):
        result = iso - source
        result /= np.linalg.norm(result)
        return result

def rotateAroundAxisAtOrigin(p: np.ndarray, r: np.ndarray, t: float):
    # ASSUMES r IS NORMALIZED ALREADY and center is (0, 0, 0)
    # p - vector to rotate
    # r - rotation axis
    # t - rotation angle
    sptr = np.sin(t)
    cptr = np.cos(t)
    result = np.array((
        (-r[0]*(-r[0]*p[0] - r[1]*p[1] - r[2]*p[2]))*(1-cptr) + p[0]*cptr + (-r[2]*p[1] + r[1]*p[2])*sptr,
        (-r[1]*(-r[0]*p[0] - r[1]*p[1] - r[2]*p[2]))*(1-cptr) + p[1]*cptr + (+r[2]*p[0] - r[0]*p[2])*sptr,
        (-r[2]*(-r[0]*p[0] - r[1]*p[1] - r[2]*p[2]))*(1-cptr) + p[2]*cptr + (-r[1]*p[0] + r[0]*p[1])*sptr
    ))
    return result


def inverseRotateBeamAtOriginRHS(vec: np.ndarray, theta: float, phi: float, coll: float):
    tmp = rotateAroundAxisAtOrigin(vec, np.array((0., 1., 0.)), -(phi+coll))  # coll rotation + correction
    sptr = np.sin(-phi)
    cptr = np.cos(-phi)
    rotation_axis = np.array((sptr, 0., cptr))
    result = rotateAroundAxisAtOrigin(tmp, rotation_axis, theta)
    return result


def closeto(a, b, tolerance=1e-6):
    return abs(a-b) <= tolerance


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


def bboxCalc(skin):
    skin = skin > 0

    skin_x = np.any(skin, axis=(1, 2))
    indicesXValid = [i for i in np.arange(skin.shape[0]) if skin_x[i]]
    x_min = min(indicesXValid)
    x_max = max(indicesXValid)

    skin_y = np.any(skin, axis=(0, 2))
    indicesYValid = [i for i in np.arange(skin.shape[1]) if skin_y[i]]
    y_min = min(indicesYValid)
    y_max = max(indicesYValid)

    skin_z = np.any(skin, axis=(0, 1))
    indicesZValid = [i for i in np.arange(skin.shape[2]) if skin_z[i]]
    z_min = min(indicesZValid)
    z_max = max(indicesZValid)

    BBoxStart = (x_min, y_min, z_min)
    BBoxDim = (x_max-x_min, y_max-y_min, z_max-z_min)
    return BBoxStart, BBoxDim


def metadataGen():
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i + 1)
        sourcePatientFolder = os.path.join(sourceFolder, patientName)
        FastDoseFolder = os.path.join(sourcePatientFolder, "FastDose")
        QihuiRyanFolder = os.path.join(sourcePatientFolder, "QihuiRyan")
        if not os.path.isdir(QihuiRyanFolder):
            os.mkdir(QihuiRyanFolder)
        expPrepFolder = os.path.join(QihuiRyanFolder, "preprocess")
        if not os.path.isdir(expPrepFolder):
            os.mkdir(expPrepFolder)
        
        inputFolder = os.path.join(FastDoseFolder, "prep_output")
        
        # calculate isocenter
        dimension = os.path.join(inputFolder, "dimension.txt")
        with open(dimension, "r") as f:
            dimension = f.readline()
            resolution = f.readline()
        dimension = dimension.replace(" ", ", ")
        dimension = eval(dimension)  # (x, y, z)
        dimension_flip = np.flip(dimension)

        resolution = resolution.replace(" ", ", ")
        resolution = eval(resolution)
        resolution_flip = np.flip(resolution)

        maskFolder = os.path.join(sourcePatientFolder, "InputMask")
        ptv = os.path.join(maskFolder, "ROI.bin")
        ptv = np.fromfile(ptv, dtype=np.uint8)
        ptv = np.reshape(ptv, dimension_flip)  # (z, y, x)
        centroid_flip = centroidCalc(ptv)  # (z, y, x)
        centroid = np.flip(centroid_flip)  # (x, y, z)

        # calculate bounding box
        bbox = "SKIN"
        bbox = os.path.join(maskFolder, bbox + ".bin")
        bbox = np.fromfile(bbox, dtype=np.uint8)
        bbox = np.reshape(bbox, dimension_flip)  # (z, y, x)
        BBoxStart, BBoxDim = bboxCalc(bbox)  # (z, y, x)
        BBoxStart = np.flip(BBoxStart)
        BBoxDim = np.flip(BBoxDim) + 1

        beamList = os.path.join(FastDoseFolder, "beamlist.txt")
        with open(beamList, "r") as f:
            beamList = f.readlines()
        
        dimension_text = "{} {} {}".format(*dimension)
        resolution_text = "{} {} {}".format(*resolution)
        dicomVolumeStartCoords = "{:.6f} {:.6f} {:.6f}".format(0, 0, 0)
        BBoxStart_text = "{} {} {}".format(*BBoxStart)
        BBoxDim_text = "{} {} {}".format(*BBoxDim)
        REVDimension_text = "800 800 800"
        convlatText = "0.250000"
        convstep = "0.250000"
        kernel_extent = "1.000000"
        ss_factor = "3"
        phiThetaRadii = "8 8 24"
        penumbra = "1.000000"
        beam_count = "{}".format(len(beamList))
        beam_spectrum = "spec_6mv"
        target_structure = "ROI"
        reduce = "0"

        contentData = [dimension_text, resolution_text, dicomVolumeStartCoords, BBoxStart_text,
            BBoxDim_text, REVDimension_text, convlatText, convstep, kernel_extent, ss_factor,
            phiThetaRadii, penumbra, beam_count, beam_spectrum, target_structure, reduce]
        contentData = "\n".join(contentData)
        content = contentHeader + contentData + "\n"
        contentFile = os.path.join(expPrepFolder, "omni-header.txt")
        with open(contentFile, "w") as f:
            f.write(content)
        

        # generate omni_beam_list
        lineInfoList = []
        for idx, line in enumerate(beamList):
            line = line.replace(" ", ", ")
            line = np.array(eval(line))
            azimuth_deg, zenith_deg, coll_deg = line
            line_rad = line * np.pi / 180
            azimuth_rad, zenith_rad, coll_rad = line_rad
            
            newBeam = beam()
            newBeam.azimuth = azimuth_rad
            newBeam.zenith = zenith_rad
            newBeam.coll = coll_rad
            newBeam.isocenter = (centroid + 0.5) * np.array(resolution)
            newBeam.sad = 100
            newBeam.beamlet_size = np.array((0.5, 0.5))
            newBeam.fmap_size = np.array((20, 20))
            newBeam.reconfigure()

            lineInfo = "{} ".format(idx) + "{:.6f} {:.6f} {:.6f} {:.6f} ".format(azimuth_deg, zenith_deg, coll_deg, newBeam.sad) \
                + "{} {} ".format(newBeam.iso_type, newBeam.iso_loc) \
                + "{:.6f} {:.6f} {:.6f} ".format(*(newBeam.isocenter)) \
                + "{:.6f} {:.6f} {:.6f} ".format(newBeam.source[0], newBeam.source[1], newBeam.source[2]) \
                + "{} {:.6f} {:.6f} {:.6f} ".format(newBeam.orient_type, newBeam.direction[0], newBeam.direction[1], newBeam.direction[2]) \
                + "{:.6f} {:.6f} ".format(newBeam.beamlet_size[0], newBeam.beamlet_size[1]) \
                + "{} {} ".format(newBeam.fmap_size[0], newBeam.fmap_size[1]) \
                + "fmap-{:06d}.raw".format(idx)
            lineInfoList.append(lineInfo)

        lineInfoList.insert(0, "{}".format(len(beamList)))
        lineInfoList = "\n".join(lineInfoList)
        lineInfoFile = os.path.join(expPrepFolder, "omni_beam_list.txt")
        with open(lineInfoFile, "w") as f:
            f.write(lineInfoList)
        print(patientName)


def copyOtherFiles():
    templateFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/PreProcessRyanSample"
    sourceFiles = [
        "convolution_phi_angles.raw",
        "convolution_theta_angles.raw",
        "cumulative_kernel.h5"
    ]

    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        sourcePatientFolder = os.path.join(sourceFolder, patientName)
        sourcePrepFolder = os.path.join(sourcePatientFolder, "FastDose", "prep_output")
        destFolder = os.path.join(sourcePatientFolder, "QihuiRyan", "preprocess")
        for file in sourceFiles:
            command = "cp {} {}".format(os.path.join(templateFolder, file), destFolder)
            os.system(command)
        
        command = "cp {} {}".format(os.path.join(sourcePrepFolder, "density.raw"), destFolder)
        os.system(command)
        command = "cp {} {}".format(os.path.join(sourcePrepFolder, "roi_list.h5"), destFolder)
        os.system(command)


def fluenceMapGen():
    fluenceShape = (20, 20)
    fluenceElements = fluenceShape[0] * fluenceShape[1]
    numChunks = 2
    for i in range(numPatients):
        patientName = "Patient{:03d}".format(i+1)
        fluenceArrayList = []
        for j in range(numChunks):
            fluenceMapFile = os.path.join(sourceFolder, patientName,
                "FastDose", "doseMat{}".format(j+1), "doseMatFolder", "fluenceMap.bin")
            fluenceArray = np.fromfile(fluenceMapFile, dtype=np.uint8)
            fluenceArrayList.append(fluenceArray)
        fluenceArrayConcat = np.concatenate(fluenceArrayList, axis=0)
        
        numBeams = len(fluenceArrayConcat) / fluenceElements
        numBeams = int(numBeams)

        # write to file
        fluenceFolder = os.path.join(sourceFolder, patientName,
            "QihuiRyan", "preprocess", "fluence_maps")
        if not os.path.isdir(fluenceFolder):
            os.mkdir(fluenceFolder)
        for j in range(numBeams):
            idx_start = i * fluenceElements
            idx_end = (i + 1) * fluenceElements
            beamFluence = fluenceArrayConcat[idx_start: idx_end].astype(np.float32)
            file = os.path.join(fluenceFolder, "fmap-{:06d}.raw".format(j))
            beamFluence.tofile(file)
            print(file)


if __name__ == "__main__":
    # metadataGen()
    # copyOtherFiles()
    fluenceMapGen()