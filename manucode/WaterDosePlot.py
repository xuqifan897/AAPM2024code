import os
import numpy as np
import matplotlib.pyplot as plt

ResultFolder = "/data/qifan/projects/FastDoseWorkplace/DoseBench/MCSuperSampling"
ManuFiguresFolder = "/data/qifan/projects/AAPM2024/manufigures"
CCCSWaterFolder = "/data/qifan/projects/FastDoseWorkplace/DoseBench/water"
Materials = None


def main():
    width_list = [0.5, 1.0, 2.0]
    CCCS_pixel_spacing = [0.08333, 0.125, 0.25]
    CCCS_pixel_on = [6, 8, 8]
    CCCS_fluence_dim = 24
    CCCS_long_spacing = 0.25

    MC_dim = [99, 99, 256]
    MC_long_spacing = 0.1
    lateral_xlim = (-4.0, 4.0)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i in range(3):
        width = width_list[i]
        CCCS_PS = CCCS_pixel_spacing[i]
        CCCS_PO = CCCS_pixel_on[i]
        CCCSArrayFile = os.path.join(CCCSWaterFolder, "width{}mm".format(int(width*10)),
            "BEVdose{}{}.bin".format(CCCS_fluence_dim, CCCS_PO))
        CCCSArray = np.fromfile(CCCSArrayFile, dtype=np.float32)
        assert CCCSArray.size % (CCCS_fluence_dim ** 2) == 0, "Dimension error."
        CCCSArrayDimZ = CCCSArray.size // (CCCS_fluence_dim ** 2)
        CCCSArrayShape = (CCCSArrayDimZ, CCCS_fluence_dim, CCCS_fluence_dim)
        CCCSArray = np.reshape(CCCSArray, CCCSArrayShape)

        MCArrayFile = os.path.join(ResultFolder, "MCDoseWater_width_{:.1f}cm.bin".format(width))
        MCArray = np.fromfile(MCArrayFile, dtype=np.float64)
        MCArray = np.reshape(MCArray, MC_dim)

        # Extract Centerline
        CCCS_centerline_idx = int(CCCS_fluence_dim / 2)
        CCCS_centerline  = CCCSArray[:, CCCS_centerline_idx, CCCS_centerline_idx].copy()
        CCCS_depth = np.arange(CCCSArrayDimZ) * CCCS_long_spacing
        if i == 2:
            CCCS_centerline[55] = (CCCS_centerline[54] + CCCS_centerline[56]) / 2

        MC_centerline_idx = (MC_dim[0] - 1) // 2
        MC_centerline = MCArray[MC_centerline_idx, MC_centerline_idx, :].copy()
        MC_depth = (np.arange(MC_dim[2]) + 0.5) * MC_long_spacing

        # upsample CCCS_depth
        CCCS_interpolated = np.interp(MC_depth, CCCS_depth, CCCS_centerline)
        CCCS_interpolated = CCCS_interpolated * 100 / np.max(CCCS_interpolated)  # normalize
        # rescale MC_centerline to minimize the difference
        scale = np.sum(CCCS_interpolated * MC_centerline) / np.sum(np.square(MC_centerline))
        MC_centerline *= scale
        ylim = [20, 110]
        axes[0, i].plot(MC_depth, CCCS_interpolated, label="CCCS", linewidth=2.0)
        axes[0, i].plot(MC_depth, MC_centerline, label="Monte Carlo", linestyle="--", linewidth=2.0)
        axes[0, i].set_xlabel("Depth (cm)", fontsize=16)
        axes[0, i].set_ylabel("Dose (% of Dmax)", fontsize=16)
        axes[0, i].tick_params(axis="x", labelsize=14)
        axes[0, i].tick_params(axis="y", labelsize=14)
        axes[0, i].set_title("{:.1f}cm".format(width), fontsize=16)
        if i == 2:
            axes[0, i].legend(loc="upper right", fontsize=16)
        axes[0, i].set_ylim(ylim[0], ylim[1])

        # lateral profile
        CCCSArray *= 100 / np.max(CCCSArray)
        SamplingDepth = 11.0
        depth_idx_cccs = int(SamplingDepth / CCCS_long_spacing)
        CCCS_lateral_profile = CCCSArray[depth_idx_cccs, CCCS_centerline_idx, :]
        CCCS_displacement_points = int((lateral_xlim[1] - lateral_xlim[0]) / CCCS_PS)
        leading = int((CCCS_displacement_points - CCCS_fluence_dim) / 2)
        CCCS_displacement = (np.arange(CCCS_displacement_points) -
            (CCCS_displacement_points - 1) / 2) * CCCS_PS
        CCCSMask = np.ones(CCCS_displacement_points, dtype=bool)
        CCCSMask[:leading] = False
        CCCSMask[-leading:] = False
        CCCS_lateral_extended = np.zeros_like(CCCS_displacement)
        CCCS_lateral_extended[CCCSMask] = CCCS_lateral_profile

        MCArray *= scale
        depth_idx_mc = int(SamplingDepth / MC_long_spacing)
        MC_lateral_profile = MCArray[MC_centerline_idx, :, depth_idx_mc]
        MC_displacement = (np.arange(MC_dim[1]) - (MC_dim[1]-1)/2) * MC_long_spacing
        axes[1, i].plot(CCCS_displacement, CCCS_lateral_extended, label="CCCS", linewidth=2.0)
        axes[1, i].plot(MC_displacement, MC_lateral_profile,
            label="Monte Carlo", linestyle="--", linewidth=2.0)
        axes[1, i].set_xlabel("Off-axis Distance (cm)", fontsize=16)
        axes[1, i].set_ylabel("Dose (% of Dmax)", fontsize=16)
        axes[1, i].set_xlim(lateral_xlim[0], lateral_xlim[1])
        axes[1, i].set_ylim(0, 60)
        axes[1, i].tick_params(axis="x", labelsize=14)
        axes[1, i].tick_params(axis="y", labelsize=14)

    plt.tight_layout()
    figureFile = os.path.join(ManuFiguresFolder, "WaterDose.png")
    plt.savefig(figureFile)
    figureFile = os.path.join(ManuFiguresFolder, "WaterDose.eps")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()


def comp_gamma_index():
    width_list = [0.5, 1.0, 2.0]
    CCCS_pixel_on = [6, 8, 8]
    CCCS_fluence_dim = 24
    CCCS_long_spacing = 0.25

    MC_dim = [99, 99, 256]
    MC_long_spacing = 0.1
    SuperSamplingFactor = 10
    criterion = 2
    for i in range(3):
        width = width_list[i]
        CCCS_PO = CCCS_pixel_on[i]
        CCCSArrayFile = os.path.join(CCCSWaterFolder, "width{}mm".format(int(width*10)),
            "BEVdose{}{}.bin".format(CCCS_fluence_dim, CCCS_PO))
        CCCSArray = np.fromfile(CCCSArrayFile, dtype=np.float32)
        assert CCCSArray.size % (CCCS_fluence_dim ** 2) == 0, "Dimension error."
        CCCSArrayDimZ = CCCSArray.size // (CCCS_fluence_dim ** 2)
        CCCSArrayShape = (CCCSArrayDimZ, CCCS_fluence_dim, CCCS_fluence_dim)
        CCCSArray = np.reshape(CCCSArray, CCCSArrayShape)

        MCArrayFile = os.path.join(ResultFolder, "MCDoseWater_width_{:.1f}cm.bin".format(width))
        MCArray = np.fromfile(MCArrayFile, dtype=np.float64)
        MCArray = np.reshape(MCArray, MC_dim)

        # Extract Centerline
        CCCS_centerline_idx = int(CCCS_fluence_dim / 2)
        CCCS_centerline  = CCCSArray[:, CCCS_centerline_idx, CCCS_centerline_idx].copy()
        CCCS_depth = np.arange(CCCSArrayDimZ) * CCCS_long_spacing * 10  # cm to mm
        if i == 2:
            CCCS_centerline[55] = (CCCS_centerline[54] + CCCS_centerline[56]) / 2

        MC_centerline_idx = (MC_dim[0] - 1) // 2
        MC_centerline = MCArray[MC_centerline_idx, MC_centerline_idx, :].copy()
        MC_depth = (np.arange(MC_dim[2]) + 0.5) * MC_long_spacing * 10  # cm to mm

        # upsample CCCS_depth
        CCCS_interpolated = np.interp(MC_depth, CCCS_depth, CCCS_centerline)
        CCCS_interpolated = CCCS_interpolated * 100 / np.max(CCCS_interpolated)  # normalize
        # rescale MC_centerline to minimize the difference
        scale = np.sum(CCCS_interpolated * MC_centerline) / np.sum(np.square(MC_centerline))
        MC_centerline *= scale

        MCDepthSuperSampling = (np.arange(MC_depth.size * SuperSamplingFactor) + 0.5) \
            * MC_long_spacing * 10 / SuperSamplingFactor
        MCCenterlineSuperSampling = np.interp(MCDepthSuperSampling, MC_depth, MC_centerline)

        if False:
            plt.plot(MC_depth, CCCS_interpolated)
            plt.plot(MCDepthSuperSampling, MCCenterlineSuperSampling)
            plt.xlabel("Depth (cm)")
            plt.ylabel("Dose (% of Dmax)")
            figure = os.path.join(ManuFiguresFolder, "check{}.png".format(i))
            plt.savefig(figure)
            plt.clf()
            continue

        MCDepthCCCS = np.expand_dims(MC_depth, axis=(1, 2))
        CCCSInterpolatedCCCS = np.expand_dims(CCCS_interpolated, axis=(1, 2))
        CoordsCCCS = np.concatenate((MCDepthCCCS, CCCSInterpolatedCCCS), axis=2)

        MCDepthSuperSampling = np.expand_dims(MCDepthSuperSampling, axis=(0, 2))
        MCCenterlineSuperSampling = np.expand_dims(MCCenterlineSuperSampling, axis=(0, 2))
        CoordsMC = np.concatenate((MCDepthSuperSampling, MCCenterlineSuperSampling), axis=2)

        diff = CoordsCCCS - CoordsMC
        distance = np.square(diff)
        distance = np.sum(distance, axis=2)
        min_distance = np.min(distance, axis=1)
        PassingMap = min_distance < criterion ** 2
        if False:
            print(PassingMap)
        PassingRate = np.sum(PassingMap) / PassingMap.size
        print("Width: {}cm, criterion: {}mm/{}%, passing rate: {}".format(
            width, criterion, criterion, PassingRate))


if __name__ == "__main__":
    main()
    comp_gamma_index()