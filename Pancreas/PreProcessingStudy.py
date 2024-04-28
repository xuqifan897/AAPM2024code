import os
import numpy as np
import math


def show_fluence_maps():
    rootFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/PreProcessRyanSample"
    targetFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/PreProcessShow"
    fluenceMapFolder = os.path.join(rootFolder, "fluence_maps")
    fluenceMapTarget = os.path.join(targetFolder, "fluence_maps")
    # if not os.path.isdir(fluenceMapTarget):
    #     os.makedirs(fluenceMapTarget)

    numAngles = 470
    fluenceShape = (20, 20)
    for i in range(numAngles):
        binaryFluenceFile = os.path.join(fluenceMapFolder, "fmap-{:06d}.raw".format(i))
        fluenceArray = np.fromfile(binaryFluenceFile, dtype=np.float32)
        fluenceArray = np.reshape(fluenceArray, fluenceShape)
        print(fluenceArray)
        break


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


def beamlist_processing_test():
    """
    This function tries to generate the beam list to mimic the one generated
    by the original Ryan Neph's dose calculation code
    """
    exampleFile = "/data/qifan/projects/FastDoseWorkplace/Pancreas/Patient001/FastDose/beamlist.txt"
    isocenter = np.array((2.8183e+01, 2.5177e+01, 2.1796e+01))
    sad = 100.0
    beamlet_size = np.array((0.5, 0.5))
    fmap_size = np.array((20, 20), dtype=int)
    targetFolder = "/data/qifan/projects/FastDoseWorkplace/Pancreas/PreProcessShow"

    with open(exampleFile, "r") as f:
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


if __name__ == "__main__":
    # show_fluence_maps()
    beamlist_processing_test()