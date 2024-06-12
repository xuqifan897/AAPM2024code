import os
import numpy as np
import matplotlib.pyplot as plt

MCResultFolder = "/data/qifan/projects/FastDoseWorkplace/DoseBench/MCSuperSampling"
CCCSSlabFolder = "/data/qifan/projects/FastDoseWorkplace/DoseBench/slab"
CCCSWaterFolder = "/data/qifan/projects/FastDoseWorkplace/DoseBench/water"

ManuFiguresFolder = "/data/qifan/projects/AAPM2024/manufigures"
Materials = None
densityMap = None

def densityMapInit():
    global densityMap
    matMap = {"Adipose": 0.92, "Muscle": 1.04, "Bone": 1.85, "Lung": 0.25}
    global Materials
    Materials = [
        ("Adipose", 16),
        ("Muscle", 16),
        ("Bone", 16),
        ("Muscle", 16),
        ("Lung", 96),
        ("Muscle", 16),
        ("Bone", 16),
        ("Adipose", 16),
        ("Bone", 16),
        ("Muscle", 16),
        ("Adipose", 16)
    ]
    totalSlices = 0
    for material, thickness in Materials:
        totalSlices += thickness
    densityMap = np.zeros(totalSlices)
    offset = 0
    for material, thickness in Materials:
        density = matMap[material]
        densityMap[offset: offset+thickness] = density
        offset += thickness

def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    width_list = [0.5, 1.0, 2.0]
    CCCS_pixel_spacing = [0.08333, 0.125, 0.25]
    CCCS_pixel_on = [6, 8, 8]
    CCCS_fluence_dim = 24
    CCCS_long_spacing = 0.25

    MC_dim = [99, 99, 256]
    MC_long_spacing = 0.1
    samplingWidth = 0
    lateral_xlim = (-4.0, 4.0)

    for i in range(3):
        width = width_list[i]
        CCCS_PS = CCCS_pixel_spacing[i]
        CCCS_PO = CCCS_pixel_on[i]
        width_mm = int(width * 10)
        CCCSArrayFile = os.path.join(CCCSSlabFolder, "width{}mm".format(width_mm),
            'BEVdose{}{}.bin'.format(CCCS_fluence_dim, CCCS_PO))
        CCCS_Array = np.fromfile(CCCSArrayFile, dtype=np.float32)
        assert CCCS_Array.size % (CCCS_fluence_dim **2) == 0, "Dimension error."
        CCCS_Array_DimZ = CCCS_Array.size // (CCCS_fluence_dim ** 2)
        CCCS_Array_Shape = (CCCS_Array_DimZ, CCCS_fluence_dim, CCCS_fluence_dim)
        CCCS_Array = np.reshape(CCCS_Array, CCCS_Array_Shape)

        MCArrayFile = os.path.join(MCResultFolder, "MCDose_width_{:.1f}cm.bin".format(width))
        MCArray = np.fromfile(MCArrayFile, dtype=np.float64)
        MCArray = np.reshape(MCArray, MC_dim)
        densityMap_ = np.expand_dims(densityMap, axis=(0, 1))
        MCArray /= densityMap_

        # Extract Centerline
        CCCS_centerline_idx = int(CCCS_fluence_dim / 2)
        CCCS_centerline = CCCS_Array[:, CCCS_centerline_idx, CCCS_centerline_idx].copy()
        CCCS_depth = np.arange(CCCS_Array_DimZ) * CCCS_long_spacing

        MC_centerline_idx = int((MC_dim[0]-1)/2)
        MC_centerline = MCArray[MC_centerline_idx-samplingWidth : MC_centerline_idx+samplingWidth+1,
                                MC_centerline_idx-samplingWidth : MC_centerline_idx+samplingWidth+1, :]
        MC_centerline = np.sum(MC_centerline, axis=(0, 1))
        MC_depth = (np.arange(MC_dim[2]) + 0.5) * MC_long_spacing

        # upsample CCCS_depth
        CCCS_interpolated = np.interp(MC_depth, CCCS_depth, CCCS_centerline)
        CCCS_interpolated = CCCS_interpolated * 100 / np.max(CCCS_interpolated)  # normalize
        # rescale MC_centerline to minimize the difference
        scale = np.sum(MC_centerline * CCCS_interpolated) / np.sum(MC_centerline**2)
        MC_centerline *= scale
        ylim = [20, 110]
        draw_indicators(axes[0, i], ylim)
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
        CCCS_Array *= 100 / np.max(CCCS_Array)
        SamplingDepth = 11.0
        depth_idx_cccs = int(SamplingDepth / CCCS_long_spacing)
        if i == 0 or i == 1:
            depth_idx_cccs -= 1
        CCCS_lateral_profile = CCCS_Array[depth_idx_cccs, CCCS_centerline_idx, :].copy()
        CCCS_displacement = np.arange(CCCS_fluence_dim) - (CCCS_centerline_idx - 0.5)
        CCCS_displacement *= CCCS_PS * (SamplingDepth / 12.8)
        # Extend the CCCS displacement to the full range
        CCCS_displacement_points = int((lateral_xlim[1] - lateral_xlim[0]) / CCCS_PS)
        leading = int((CCCS_displacement_points - CCCS_fluence_dim) / 2)
        ExtensionMask = np.ones(CCCS_displacement_points, dtype=bool)
        ExtensionMask[:leading] = False
        ExtensionMask[-leading:] = False
        CCCS_displacement_extended = (np.arange(CCCS_displacement_points)
            - (CCCS_displacement_points-1)/2) * CCCS_PS
        CCCS_lateral_profile_extended = np.zeros(CCCS_displacement_points)
        CCCS_lateral_profile_extended[ExtensionMask] = CCCS_lateral_profile

        MCArray *= scale
        depth_idx_mc = int(SamplingDepth / MC_long_spacing)
        MC_lateral_profile = MCArray[MC_centerline_idx, :, depth_idx_mc]
        MC_displacement = (np.arange(MC_dim[1]) - MC_centerline_idx) * MC_long_spacing

        # axes[1, i].plot(CCCS_displacement, CCCS_lateral_profile, label="CCCS", linewidth=2.0)
        axes[1, i].plot(CCCS_displacement_extended, CCCS_lateral_profile_extended, label="CCCS", linewidth=2.0)
        axes[1, i].plot(MC_displacement, MC_lateral_profile, label="Monte Carlo", linestyle="--", linewidth=2.0)
        axes[1, i].set_xlabel("Off-axis Distance (cm)", fontsize=16)
        axes[1, i].set_ylabel("Dose (% of Dmax)", fontsize=16)
        axes[1, i].tick_params(axis="x", labelsize=14)
        axes[1, i].tick_params(axis="y", labelsize=14)
        # axes[1, i].legend(loc="upper right", fontsize=16)
        axes[1, i].set_xlim(lateral_xlim[0], lateral_xlim[1])
        axes[1, i].set_ylim(0, 50)

    plt.tight_layout()
    figureFile = os.path.join(ManuFiguresFolder, "SlabDose.png")
    plt.savefig(figureFile)
    figureFile = os.path.join(ManuFiguresFolder, "SlabDose.eps")
    plt.savefig(figureFile)
    plt.close(fig)
    plt.clf()


def draw_indicators(axs, ylim):
    depths = []
    for material, depth in Materials:
        if len(depths) == 0:
            depths.append(depth)
        else:
            depths.append(depths[-1] + depth)
    depths = np.array(depths)
    depths = depths * 0.1
    for depth in depths:
        axs.plot((depth, depth), (ylim[0], ylim[1]), color="pink")


def comp_gamma_index():
    width_list = [0.5, 1.0, 2.0]
    CCCS_pixel_on = [6, 8, 8]
    CCCS_fluence_dim = 24
    CCCS_long_spacing = 0.25

    MC_dim = [99, 99, 256]
    MC_long_spacing = 0.1
    samplingWidth = 0
    criterion = 2  # 2mm/2%
    superSamplingFactor = 10

    for i in range(3):
        width = width_list[i]
        CCCS_PO = CCCS_pixel_on[i]
        width_mm = int(width * 10)
        CCCSArrayFile = os.path.join(CCCSSlabFolder, "width{}mm".format(width_mm),
            'BEVdose{}{}.bin'.format(CCCS_fluence_dim, CCCS_PO))
        CCCS_Array = np.fromfile(CCCSArrayFile, dtype=np.float32)
        assert CCCS_Array.size % (CCCS_fluence_dim **2) == 0, "Dimension error."
        CCCS_Array_DimZ = CCCS_Array.size // (CCCS_fluence_dim ** 2)
        CCCS_Array_Shape = (CCCS_Array_DimZ, CCCS_fluence_dim, CCCS_fluence_dim)
        CCCS_Array = np.reshape(CCCS_Array, CCCS_Array_Shape)

        MCArrayFile = os.path.join(MCResultFolder, "MCDose_width_{:.1f}cm.bin".format(width))
        MCArray = np.fromfile(MCArrayFile, dtype=np.float64)
        MCArray = np.reshape(MCArray, MC_dim)
        densityMap_ = np.expand_dims(densityMap, axis=(0, 1))
        MCArray /= densityMap_

        # Extract Centerline
        CCCS_centerline_idx = int(CCCS_fluence_dim / 2)
        CCCS_centerline = CCCS_Array[:, CCCS_centerline_idx, CCCS_centerline_idx].copy()
        CCCS_depth = np.arange(CCCS_Array_DimZ) * CCCS_long_spacing

        MC_centerline_idx = int((MC_dim[0]-1)/2)
        MC_centerline = MCArray[MC_centerline_idx-samplingWidth : MC_centerline_idx+samplingWidth+1,
                                MC_centerline_idx-samplingWidth : MC_centerline_idx+samplingWidth+1, :]
        MC_centerline = np.sum(MC_centerline, axis=(0, 1))
        MC_depth = (np.arange(MC_dim[2]) + 0.5) * MC_long_spacing

        # upsample CCCS_depth
        CCCS_interpolated = np.interp(MC_depth, CCCS_depth, CCCS_centerline)
        CCCS_interpolated = CCCS_interpolated * 100 / np.max(CCCS_interpolated)  # normalize
        # rescale MC_centerline to minimize the difference
        scale = np.sum(MC_centerline * CCCS_interpolated) / np.sum(MC_centerline**2)
        MC_centerline *= scale

        MC_depth *= 10
        if False:
            plt.plot(MC_depth, CCCS_interpolated)
            plt.plot(MC_depth, MC_centerline)
            plt.xlabel("Depth (mm)")
            plt.ylabel("Dose (% of Dmax)")
            figureFile = os.path.join(ManuFiguresFolder, "check{}.png".format(i))
            plt.savefig(figureFile)
            plt.clf()
        
        # take MC as base.
        MC_depth_ss_points = MC_depth.size * superSamplingFactor
        MCRes_ss = MC_long_spacing * 10 / superSamplingFactor
        MC_depth_ss = (np.arange(MC_depth_ss_points) + 0.5) * MCRes_ss
        MC_centerline_ss = np.interp(MC_depth_ss, MC_depth, MC_centerline)
        if False:
            plt.plot(MC_depth_ss, MC_centerline_ss)
            figureFile = os.path.join(ManuFiguresFolder, "checkSuper{}.png".format(i))
            plt.savefig(figureFile)
            plt.clf()
        
        MC_depth_CCCS = np.expand_dims(MC_depth, axis=(1, 2))
        CCCS_interpolated_dilate = np.expand_dims(CCCS_interpolated, axis=(1, 2))
        coords_CCCS = np.concatenate((MC_depth_CCCS, CCCS_interpolated_dilate), axis=2)

        MC_depth_ss = np.expand_dims(MC_depth_ss, axis=(0, 2))
        MC_centerline_ss = np.expand_dims(MC_centerline_ss, axis=(0, 2))
        coords_MC = np.concatenate((MC_depth_ss, MC_centerline_ss), axis=2)

        diff = coords_CCCS - coords_MC
        distance = np.square(diff)
        distance = np.sum(distance, axis=2)
        min_distance = np.min(distance, axis=1)
        PassRate = np.sum(min_distance < criterion**3) / min_distance.size
        print("Width: {}cm, criterion: {}mm/{}%, pass rate: {}".format(
            width, criterion, criterion, PassRate))



if __name__ == "__main__":
    densityMapInit()
    main()
    comp_gamma_index()