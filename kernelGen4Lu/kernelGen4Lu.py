import os
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

eps = 1e-7
rootFolder = "/data/qifan/projects/FastDoseWorkplace/kernelGen4Lu"

def examineKernel():
    kernelFile = os.path.join(rootFolder, "kernel.bin")
    kernel = np.fromfile(kernelFile, dtype=np.float64)
    marginHead = 100
    marginTail = 20
    radiusDim = 100
    kernelShape = (marginHead + marginTail, radiusDim)
    kernel = np.reshape(kernel, kernelShape)
    figureFile = os.path.join(rootFolder, "kernelShow.png")
    plt.imsave(figureFile, kernel, cmap="jet")


def Process():
    """
    This function transforms the cylindrical dose kernel into a polar kernel
    """
    specFile = os.path.join(rootFolder, "kernel.bin")
    nAngles = 8
    marginHead = 100
    marginTail = 20
    radiusDim = 100
    heightRes = 0.1  # cm
    radiusRes = 0.05 # cm
    superSampling = 100
    # superSampling = 1

    angleInterval = np.pi / nAngles
    CyShape = (marginTail + marginHead, radiusDim)
    CySpec = np.fromfile(specFile, dtype=np.float64)
    CySpec = np.reshape(CySpec, CyShape)
    coeffShape = (CyShape[0]*superSampling, CyShape[1]*superSampling)

    # Assume the radius resolution, as well as the radiusDim, are the same as the raw data.

    xcoordsMat = (np.arange(coeffShape[1]) + 0.5) * radiusRes / superSampling
    xcoordsMat = np.expand_dims(xcoordsMat, axis=0)
    xcoordsMat = [xcoordsMat] * coeffShape[0]
    xcoordsMat = np.concatenate(xcoordsMat, axis=0)

    ycoordsMat = (np.arange(coeffShape[0]) - 0.5 * superSampling - marginTail * superSampling) * heightRes / superSampling
    ycoordsMat = np.expand_dims(ycoordsMat, axis=1)
    ycoordsMat = [ycoordsMat] * coeffShape[1]
    ycoordsMat = np.concatenate(ycoordsMat, axis=1)

    cot_coords = ycoordsMat / xcoordsMat
    distance = np.sqrt(xcoordsMat**2 + ycoordsMat**2)

    # for debug purposes
    if False:
        cot_coords_file = os.path.join(folder, 'cot_coords.png')
        distance_file = os.path.join(folder, 'distance.png')
        plt.imsave(cot_coords_file, np.arctan(cot_coords))
        plt.imsave(distance_file, distance)

    kernelTable = np.zeros((nAngles, radiusDim*2))

    for angleIdx in range(nAngles):
        angleLower = angleIdx * angleInterval
        angleHigher = (angleIdx + 1) * angleInterval

        if angleLower < eps:
            cot_angleLower = 1 / eps
        else:
            cot_angleLower = 1 / np.tan(angleLower)

        if angleLower > np.pi - eps:
            cot_angleHigher = - 1 / eps
        else:
            cot_angleHigher = 1 / np.tan(angleHigher)
        
        flag_angleLower = cot_coords <= cot_angleLower
        flag_angleHigher = cot_coords > cot_angleHigher
        for radiusIdx in range(radiusDim*2):  # the radius resolution is half of the height resolution
            radiusLower = radiusIdx * radiusRes
            radiusHigher = (radiusIdx + 1) * radiusRes
            flag_radiusLower = distance >= radiusLower
            flag_radiusHigher = distance < radiusHigher
            result = flag_angleLower * flag_angleHigher * flag_radiusLower * flag_radiusHigher

            # for debug purposes
            if False and radiusIdx == 20:
                firstSegment_file = os.path.join(folder, 'firstSegment.png')
                plt.imsave(firstSegment_file, result)
                break
            
            # calculate the weight of individual pixels
            sub_result = [np.hsplit(row, CyShape[1]) for row in np.vsplit(result, CyShape[0])]
            sub_result = np.array(sub_result)
            sub_result = np.sum(sub_result, axis=(2, 3))
            sub_result = sub_result / superSampling**2

            if False and radiusIdx == 20:
                firstSegment_small_file = os.path.join(folder, 'firstSegmentSmall.png')
                plt.imsave(firstSegment_small_file, sub_result)
                break

            kernelTable[angleIdx, radiusIdx] = np.sum(sub_result * CySpec)
            print("angle: {}, radius: {}".format(angleIdx, radiusIdx))
    
    resultFile = os.path.join(rootFolder, 'tabulatedKernel_{}_{}.npy'.format(nAngles, radiusDim*2))
    np.save(resultFile, kernelTable)


def kernelView():
    """
    This function is to show the CCCS kernel generated
    """
    nAngles = 8
    radiusDim = 100
    radiusRes = 0.005
    kernelFile = os.path.join(rootFolder, 'tabulatedKernel_{}_{}.npy'.format(nAngles, radiusDim*2))
    kernel = np.load(kernelFile)
    xCoords = (np.arange(radiusDim*2) + 0.5) * radiusRes
    
    kernelViewFolder = os.path.join(rootFolder, "kernelView")
    if not os.path.isdir(kernelViewFolder):
        os.mkdir(kernelViewFolder)
    for i in range(nAngles):
        plt.plot(xCoords, kernel[i, :])
        plt.xlabel('radius (cm)')
        plt.ylabel('kernel value (a.u.)')
        plt.title('kernel of angle range: [{}pi/8, {}pi/8)'.format(i, i+1))
        # file = './figures/kernelPlot{}.png'.format(i)
        file = os.path.join(kernelViewFolder, "kernelPlot{}.png".format(i))
        plt.savefig(file)
        plt.clf()


if __name__ == "__main__":
    examineKernel()
    # Process()
    # kernelView()