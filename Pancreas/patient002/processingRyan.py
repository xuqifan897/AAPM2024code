import os
import numpy as np
import math

patientFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/Patient002"
isocenter = None


def getIsocenter():
    global isocenter
    dosecalcLogFile = os.path.join(patientFolder, "FastDose", "dosecalc1.log")
    with open(dosecalcLogFile, "r") as f:
        lines = f.readlines()
    lines = [a for a in lines if "Isocetner" in a]
    assert len(lines) == 1, "Multiple lines with the entry Isocenter found"
    line = lines[0]
    
    left_bracket_idx = line.find("(")
    right_bracket_idx = line.find(")")
    content = line[left_bracket_idx: right_bracket_idx+1]
    content = eval(content)
    content = np.array(content)


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
        self.azimuth = self.azimuth % (2*math.pi)
        self.zenith = self.zenith % (2*math.pi)
        if (not closeto(self.zenith, 0.) and (closeto(self.azimuth, 0.) or closeto(self.azimuth, math.pi))):
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
    sptr = math.sin(t)
    cptr = math.cos(t)
    result = np.array((
        (-r[0]*(-r[0]*p[0] - r[1]*p[1] - r[2]*p[2]))*(1-cptr) + p[0]*cptr + (-r[2]*p[1] + r[1]*p[2])*sptr,
        (-r[1]*(-r[0]*p[0] - r[1]*p[1] - r[2]*p[2]))*(1-cptr) + p[1]*cptr + (+r[2]*p[0] - r[0]*p[2])*sptr,
        (-r[2]*(-r[0]*p[0] - r[1]*p[1] - r[2]*p[2]))*(1-cptr) + p[2]*cptr + (-r[1]*p[0] + r[0]*p[1])*sptr
    ))
    return result


def inverseRotateBeamAtOriginRHS(vec: np.ndarray, theta: float, phi: float, coll: float):
    tmp = rotateAroundAxisAtOrigin(vec, np.array((0., 1., 0.)), -(phi+coll))  # coll rotation + correction
    sptr = math.sin(-phi)
    cptr = math.cos(-phi)
    rotation_axis = np.array((sptr, 0., cptr))
    result = rotateAroundAxisAtOrigin(tmp, rotation_axis, theta)
    return result


def closeto(a, b, tolerance=1e-6):
    return abs(a-b) <= tolerance


def fluenceMapGen():
    fluenceShape = (20, 20)
    fluenceElements = fluenceShape[0] * fluenceShape[1]
    fluenceMapFiles = [os.path.join(patientFolder, "FastDose",
        "doseMat{}".format(i), "doseMatFolder", "fluenceMap.bin") for i in range(1, 3)]
    
    QihuiRyanFolder = os.path.join(patientFolder, "QihuiRyan")
    preprocessFolder = os.path.join(QihuiRyanFolder, "preprocess")
    fluenceFolder = os.path.join(preprocessFolder, "fluence_maps")
    if not os.path.isdir(fluenceFolder):
        os.makedirs(fluenceFolder)
    
    idx = 0
    for path in fluenceMapFiles:
        data = np.fromfile(path, dtype=np.uint8)
        num_elements = data.size
        numBeams = num_elements / fluenceElements
        numBeams = int(numBeams)
        for i in range(numBeams):
            idx_start = i * fluenceElements
            idx_end = (i + 1) * fluenceElements
            beamFluence = data[idx_start: idx_end]
            beamFluence = beamFluence.astype(np.float32)
            outputFile = os.path.join(fluenceFolder, "fmap-{:06d}.raw".format(idx))
            idx += 1
            beamFluence.tofile(outputFile)
            print(outputFile)


def beamlist_preprocessing():
    sad = 100.0
    beamlet_size = np.array((0.5, 0.5))
    fmap_size = np.array((20, 20), dtype=int)
    beamlistInput = os.path.join(patientFolder, "FastDose", "beamlist.txt")
    with open(beamlistInput, "r") as f:
        lines = f.readlines()
    beamlist = []
    for line in lines:
        if line == "":
            continue
        line = line.split(" ")
        azimuth, zenith, coll = float(line[0]), float(line[1]), float(line[2])
        azimuth, zenith, coll = azimuth * math.pi / 180, zenith * math.pi / 180, coll * math.pi / 180
        newBeam = beam()
        newBeam.azimuth = azimuth
        newBeam.zenith = zenith
        newBeam.coll = coll
        newBeam.isocenter = isocenter
        newBeam.sad = sad
        newBeam.beamlet_size = beamlet_size
        newBeam.fmap_size = fmap_size
        newBeam.reconfigure()
        beamlist.append(newBeam)
    
    content = "{}\n".format(len(beamlist))
    idx = 0
    for entry in beamlist:
        deg_azimuth = entry.azimuth * 180 / math.pi
        deg_zenith = entry.zenith * 180 / math.pi
        deg_coll = entry.coll * 180 / math.pi
        line = "{} ".format(idx) + "{:.6f} {:.6f} {:.6f} {:.6f} ".format(deg_azimuth, deg_zenith, deg_coll, entry.sad) \
            + "{} {} ".format(entry.iso_type, entry.iso_loc) \
            + "{:.6f} {:.6f} {:.6f} ".format(entry.isocenter[0], entry.isocenter[1], entry.isocenter[2]) \
            + "{:.6f} {:.6f} {:.6f} ".format(entry.source[0], entry.source[1], entry.source[2]) \
            + "{} {:.6f} {:.6f} {:.6f} ".format(entry.orient_type, entry.direction[0], entry.direction[1], entry.direction[2]) \
            + "{:.6f} {:.6f} ".format(entry.beamlet_size[0], entry.beamlet_size[1]) \
            + "{} {} ".format(entry.fmap_size[0], entry.fmap_size[1]) \
            + "fmap-{:06d}.raw\n".format(idx)
        idx += 1
        content = content + line

    outputFile = os.path.join(patientFolder, "QihuiRyan", "preprocess/omni_beam_list.txt")
    with open(outputFile, "w") as f:
        f.write(content)


def omni_header_preprocessing():
    """
    This function generates the omni-header file
    """
    content = \
"""# ====================================================================
# ==                         OMNI-HEADER                            ==
# ====================================================================
# | Nx Ny Nz           - Dicom Volume Dimensions                     |
# | dx dy dz           - Voxel Size (cm)                             |
# |  x  y  z           - Dicom Volume Start Coords (cm)              |
# |  i  j  k           - Dose Bounding Box Start Indices             |
# | Bx By Bz           - Dose Bounding Box Dimensions                |
# | Rx Ry Rz           - REV Convolution Array Dimensions            |
# | convlat            - Convolution ray lateral spacing (cm)        |
# | convstep           - Convolution step spacing (cm)               |
# | kernel_extent      - Dose kernel radius truncate distance (cm)   |
# | ss_factor          - Terma anti-aliasing (super-sampling) factor |
# | nphi ntheta nradii - CCK Kernel Dimensions                       |
# | penumbra           - beamlet transverse dose spread (cm)         |
# | beam_count         - Number of beams to pick from beam_list.txt  |
# | beam_spectrum      - beam energy spectrum file to use            |
# | target_structure   - Name of selected target contour             |
# | reduce coeff. mat  - M-matrix to A-matrix reduction requested?   |
# ===================================================================
"""

    metaInfoFile = os.path.join(patientFolder, "FastDose", "prep_output", "dimension.txt")
    with open(metaInfoFile) as f:
        lines = f.readlines()
    dimension = lines[0]
    resolution = lines[1]
    dicomVolumeStart = "{:.6f} {:.6f} {:.6f}".format(0, 0, 0)

    doseCalcLogFile = os.path.join(patientFolder, "FastDose", "dosecalc1.log")
    with open(doseCalcLogFile, "r") as f:
        lines = f.readlines()
    filtered = [line for line in lines if ("BBoxStart" in line) and ("BBoxDim" in line)]
    assert len(filtered) == 1, "There is not line or multiple lines filtered"
    filtered = filtered[0]
    filtered = filtered.split("BBoxStart: ")[1]
    filtered = filtered.split(", BBoxDim: ")
    BBoxStart = eval(filtered[0])
    BBoxDim = eval(filtered[1])

    beamList = os.path.join(patientFolder, "FastDose", "beamlist.txt")
    with open(beamList) as f:
        lines = f.readlines()
    lines = [a for a in lines if a != ""]
    numBeams = len(lines)

    content_append = dimension + resolution + dicomVolumeStart \
        + "\n{} {} {}\n".format(BBoxStart[0], BBoxStart[1], BBoxStart[2]) \
        + "{} {} {}\n".format(BBoxDim[0], BBoxDim[1], BBoxDim[2]) \
        + "800 800 800\n0.250000\n0.250000\n1.000000\n3\n" \
        "8 8 24\n1.000000\n{}\nspec_6mv\nPTV\n0\n".format(numBeams)
    
    content = content + "\n" + content_append
    outputFile = os.path.join(patientFolder, "QihuiRyan", "preprocess", "omni-header.txt")
    with open(outputFile, "w") as f:
        f.write(content)
    print(content)


def copy_other_data():
    """
    This function copies other data, such as the convolution angles,
    cumulative kernels, and structures.
    """
    templateFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/PreProcessRyanSample"
    sourceFiles = [
        "convolution_phi_angles.raw",
        "convolution_theta_angles.raw",
        "cumulative_kernel.h5"
    ]
    sourceFiles = [os.path.join(templateFolder, a) for a in sourceFiles]
    sourceFilesAppend = ["density.raw", "roi_list.h5"]
    sourceFilesAppend = [os.path.join(patientFolder, "FastDose", "prep_output", a) for a in sourceFilesAppend]
    sourceFiles.extend(sourceFilesAppend)

    targetFolder = os.path.join(patientFolder, "QihuiRyan", "preprocess")
    for a in sourceFiles:
        command = "cp {} {}".format(a, targetFolder)
        print(command, "\n")
        os.system(command)


if __name__ == "__main__":
    getIsocenter()
    # fluenceMapGen()
    # beamlist_preprocessing()
    # omni_header_preprocessing()
    # copy_other_data()