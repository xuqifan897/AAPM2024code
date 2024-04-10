"""
This file examines the CORT dataset.
"""
import os
import io
import sys
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
from skimage import measure, transform

def HN_dicom_Examine():
    folder0 = "/data/qifan/projects/FastDoseWorkplace/PlanTune/HN_dicom/000000"
    # tell the file types of the dcm files in folder0
    folder0_files = os.listdir(folder0)
    ctFiles = []
    rtFiles = []
    for a in folder0_files:
        path = os.path.join(folder0, a)
        dataset = pydicom.dcmread(path)
        if (dataset.Modality == "CT"):
            ctFiles.append(path)
        else:
            rtFiles.append(path)
            print(a, dataset.Modality)
    ctFiles.sort()
    rtFiles.sort()
    outputFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/HN_dicom/view0"
    
    if False:
        rtList = ['BRAIN_STEM', 'BRAIN_STEM_PRV', 'CEREBELLUM', 'CHIASMA', 'LARYNX', 'LENS_LT',
              'LENS_RT', 'LIPS', 'OPTIC_NRV_LT', 'OPTIC_NRV_RT', 'PAROTID_LT', 'PAROTID_RT',
              'PTV70', 'SKIN', 'SPINAL_CORD', 'SPINL_CRD_PRV', 'TEMP_LOBE_LT', 'TEMP_LOBE_RT',
              'TM_JOINT_LT', 'TM_JOINT_RT']
        dicomView(ctFiles, rtFiles[0], outputFolder, rtList)

    if False:
        rtList = ["1cmptv", "BRAC_PLX", "GTV", "SKIN", "SPINAL_CORD"]
        anotherFile = "/data/qifan/projects/FastDoseWorkplace/PlanTune/HN_dicom/000001/000000.dcm"
        dicomView(ctFiles, anotherFile, outputFolder, rtList)
    
    if False:
        # To create a new plan, with only the first 40 slices of PTV
        rtFile_source = rtFiles[0]
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=os.path.dirname(ctFiles[0]), rt_struct_path=rtFile_source)
        PTV_mask = rtstruct.get_roi_mask_by_name("PTV70")
        PTV_mask[:, :, 0: -39] = 0
        rtstruct.add_roi(mask=PTV_mask, name="PTV_crop")
        rt_file_new = "/data/qifan/projects/FastDoseWorkplace/PlanTune/HN_dicom/rt_crop.dcm"
        rtstruct.save(rt_file_new)
    
    rtList = ['BRAIN_STEM', 'BRAIN_STEM_PRV', 'CEREBELLUM', 'CHIASMA', 'LARYNX', 'LENS_LT',
              'LENS_RT', 'LIPS', 'OPTIC_NRV_LT', 'OPTIC_NRV_RT', 'PAROTID_LT', 'PAROTID_RT',
              'PTV_crop', 'SKIN', 'SPINAL_CORD', 'SPINL_CRD_PRV', 'TEMP_LOBE_LT', 'TEMP_LOBE_RT',
              'TM_JOINT_LT', 'TM_JOINT_RT']
    rt_file_new = "/data/qifan/projects/FastDoseWorkplace/PlanTune/HN_dicom/rt_crop.dcm"
    dicomView(ctFiles, rt_file_new, outputFolder, rtList)


def dicomView(ctFiles, rtFile, outputFolder, rtList=None):
    # firstly, sort the ctFiles according to their idx
    if not os.path.isdir(outputFolder):
        os.mkdir(outputFolder)

    files_order = []
    for a in ctFiles:
        dataset = pydicom.dcmread(a)
        InstanceNumber = int(dataset.InstanceNumber)
        SeriesNumber = int(dataset.SeriesNumber)
        files_order.append((a, InstanceNumber, SeriesNumber))
    files_order.sort(key=lambda a: a[1])

    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=os.path.dirname(ctFiles[0]), rt_struct_path=rtFile)
    rt_struct_names = rtstruct.get_roi_names()
    if rtList is None:
        rtList = rt_struct_names
        print(rtList)
        return
    else:
        # ensure that all structures in rtList is in rt_struct_names
        for a in rtList:
            assert a in rt_struct_names, "The structure {} not found".format(a)
    
    number_of_structs = len(rtList)
    color_values = np.linspace(0, 1, number_of_structs)
    color_map = plt.get_cmap()
    colors = [color_map(a) for a in color_values]

    struct_list = []
    for name, color in zip(rtList, colors):
        mask = rtstruct.get_roi_mask_by_name(name)
        struct_list.append((name, color, mask))

    for i, a in enumerate(files_order):
        file = a[0]
        InstanceNumber = a[1]
        pixel_array = pydicom.dcmread(file).pixel_array
        pixel_array[pixel_array < 0] = 0
        plt.imshow(pixel_array, cmap="gray")
        for name, color, mask in struct_list:
            mask_slice = mask[:, :, -i-1]
            if (np.sum(mask_slice) == 0):
                continue
            contours = measure.find_contours(mask_slice, 0.5)
            initial = True
            for contour in contours:
                if initial:
                    plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color=color, label=name)
                    initial = False
                else:
                    plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color=color)
        plt.legend()
        outputFile = os.path.join(outputFolder, "{:03d}.png".format(i))
        plt.savefig(outputFile)
        plt.clf()
        print(outputFile)


def Liver_dicom_Examine():
    "Firstly, check if the folder contains rtstruct"
    dcmFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/Liver_dicom"
    files = os.listdir(dcmFolder)
    ctFiles = []
    rtFile = None
    for a in files:
        path = os.path.join(dcmFolder, a)
        dataset = pydicom.dcmread(path)
        if (dataset.Modality == "CT"):
            ctFiles.append(path)
        else:
            rtFile = path
    
    if False:
        rtList = ['Kidney_R', 'Kidney_L', 'Stomach', 'SmallBowel', 'LargeBowel',
                'Celiac', 'SMA_SMV', 'Liver', 'Heart', 'SpinalCord', 'DoseFalloff',
                'duodenum', 'Skin', 'PTV', 'cord+5mm', 'clip1', 'clip2', 'clip3', 'clips',
                'entrance', 'combinedKidney', 'CT Reference']
        outputFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/liver_view"
        dicomView(ctFiles, rtFile, outputFolder, rtList)
    
    if False:
        # seems that nothing contained in the rtstruct file
        rtList = ['Kidney_R', 'Kidney_L', 'Stomach', 'SmallBowel', 'LargeBowel',
                'Celiac', 'SMA_SMV', 'Liver', 'Heart', 'SpinalCord', 'DoseFalloff',
                'duodenum', 'Skin', 'PTV', 'cord+5mm', 'clip1', 'clip2', 'clip3', 'clips',
                'entrance', 'combinedKidney', 'CT Reference']
        rtstruct = RTStructBuilder.create_from(os.path.dirname(ctFiles[0]), rtFile)
        for name in rtList:
            mask = rtstruct.get_roi_mask_by_name(name)
            print(name, np.sum(mask))
        
    exampleDicomFile = "/data/qifan/projects/FastDoseWorkplace/PlanTune/Liver_dicom/anon3.dcm"
    dataset = pydicom.dcmread(exampleDicomFile)
    array_shape = dataset.pixel_array.shape
    print(array_shape)


def Liver_mask_Examine():
    shape = [168, 217, 217]
    image_shape = (shape[1], shape[2])
    maskFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/Liver_mask"
    roi_names = ['Celiac', 'DoseFalloff', 'Heart', 'KidneyL', 'KidneyR', 'LargeBowel',
                 'Liver', 'PTV', 'SMASMV', 'Skin', 'SmallBowel', 'SpinalCord',
                 'Stomach', 'duodenum']  #, 'entrance']
    roi_list = []
    for a in roi_names:
        path = os.path.join(maskFolder, a + '.bin')
        mask = np.fromfile(path, dtype=np.uint8)
        mask = np.reshape(mask, shape)
        mask = np.transpose(mask, axes=[0, 2, 1])
        roi_list.append((a, mask))

    dicomFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/Liver_dicom"
    dicomFiles = os.listdir(dicomFolder)
    dicom_rank = []
    
    init = False
    RescaleSlope = None
    RescaleIntercept = None
    for file in dicomFiles:
        path = os.path.join(dicomFolder, file)
        dataset = pydicom.dcmread(path)
        if (dataset.Modality == 'CT'):
            InstanceNumber = int(dataset.InstanceNumber)
            pixel_array = dataset.pixel_array
            pixel_array[pixel_array < 0] = 0
            pixel_array = transform.resize(pixel_array, image_shape)
            InstanceNumber = int(dataset.InstanceNumber)
            dicom_rank.append((InstanceNumber, pixel_array))
            if not init and False:
                RescaleSlope = float(dataset.RescaleSlope)
                RescaleIntercept = float(dataset.RescaleIntercept)
                print("(slope, intercept): ({}, {})".format(RescaleSlope, RescaleIntercept))
    dicom_rank.sort(key = lambda a: a[0])

    n_structures = len(roi_names)
    color_values = np.linspace(0, 1, n_structures)
    color_map = plt.get_cmap()
    colors = [color_map(a) for a in color_values]
    viewFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/Liver_view"
    if not os.path.isdir(viewFolder):
        os.mkdir(viewFolder)
    for i in range(shape[0]):
        plt.imshow(dicom_rank[i][1], cmap="gray")
        legend = []
        for j in range(n_structures):
            mask = roi_list[j][1]
            mask_slice = mask[i, :, :]
            if np.sum(mask_slice) == 0:
                continue
            contours = measure.find_contours(mask_slice, 0.5)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color=colors[j])
            legend.append(roi_list[j][0])
        plt.legend(legend)
        imageFile = os.path.join(viewFolder, "{:03d}.png".format(i))
        plt.savefig(imageFile)
        plt.clf()
        print(imageFile)


def convert_liverRT_to_dicom():
    shape = [168, 217, 217]
    image_shape = (shape[1], shape[2])
    maskFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/Liver_mask"
    roi_names = ['Celiac', 'DoseFalloff', 'Heart', 'KidneyL', 'KidneyR', 'LargeBowel',
                 'Liver', 'PTV', 'SMASMV', 'Skin', 'SmallBowel', 'SpinalCord',
                 'Stomach', 'duodenum']  #, 'entrance']
    roi_list = []
    for a in roi_names:
        path = os.path.join(maskFolder, a + '.bin')
        mask = np.fromfile(path, dtype=np.uint8)
        mask = np.reshape(mask, shape)
        mask = np.transpose(mask, axes=[0, 2, 1])
        roi_list.append((a, mask))

    dicomFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/Liver_dicom"
    dicomFiles = os.listdir(dicomFolder)
    dicomFiles.sort()
    exampleDicomFile = os.path.join(dicomFolder, dicomFiles[0])
    shape_org = pydicom.dcmread(exampleDicomFile).pixel_array.shape
    
    shape_new = (shape[0], shape_org[0], shape_org[1])
    roi_list_new = []
    for name, mask in roi_list:
        # To resize mask
        mask = 255.0 * mask
        mask_new = np.zeros(shape_new, dtype=mask.dtype)
        for j in range(shape_new[0]):
            mask_new[j, :, :] = transform.resize(mask[j, :, :], shape_org)
        mask[mask>0] = 1
        mask_new = mask_new > 128
        mask_new = mask_new.astype(bool)
        mask_new = np.transpose(mask_new, axes=(1, 2, 0))
        # to make it compatible with rtstruct, we have to flip it.
        mask_new = np.flip(mask_new, axis=2)
        roi_list_new.append((name, mask_new))
        print(name)
    

    # construct an rtstruct file
    rtstruct = RTStructBuilder.create_new(dicom_series_path=dicomFolder)
    for name, mask in roi_list_new:
        rtstruct.add_roi(mask=mask, name=name)
        print(name)
    rtFile = "/data/qifan/projects/FastDoseWorkplace/PlanTune/Liver_dicom/liverRT.dcm"
    rtstruct.save(rtFile)
    return

    dicom_rank = []    
    init = False
    RescaleSlope = None
    RescaleIntercept = None
    for file in dicomFiles:
        path = os.path.join(dicomFolder, file)
        dataset = pydicom.dcmread(path)
        if (dataset.Modality == 'CT'):
            InstanceNumber = int(dataset.InstanceNumber)
            pixel_array = dataset.pixel_array
            pixel_array[pixel_array < 0] = 0
            dicom_rank.append((InstanceNumber, pixel_array))
            if not init and False:
                RescaleSlope = float(dataset.RescaleSlope)
                RescaleIntercept = float(dataset.RescaleIntercept)
                print("(slope, intercept): ({}, {})".format(RescaleSlope, RescaleIntercept))
    dicom_rank.sort(key = lambda a: a[0])
    
    n_structures = len(roi_names)
    color_values = np.linspace(0, 1, n_structures)
    color_map = plt.get_cmap()
    colors = [color_map(a) for a in color_values]
    viewFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/Liver_view"
    if not os.path.isdir(viewFolder):
        os.mkdir(viewFolder)
    for i in range(shape[0]):
        plt.imshow(dicom_rank[i][1], cmap="gray")
        legend = []
        for j in range(n_structures):
            mask = roi_list_new[j][1]
            mask_slice = mask[i, :, :]
            if np.sum(mask_slice) == 0:
                continue
            contours = measure.find_contours(mask_slice, 0.5)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color=colors[j])
            legend.append(roi_list[j][0])
        plt.legend(legend)
        imageFile = os.path.join(viewFolder, "{:03d}.png".format(i))
        plt.savefig(imageFile)
        plt.clf()
        print(imageFile)


def verify_liverRT():
    folder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/Liver_dicom"
    rtFile = os.path.join(folder, "liverRT.dcm")
    ctFiles = os.listdir(folder)
    ctFiles = [os.path.join(folder, a) for a in ctFiles]
    ctFiles_filtered = []
    for file in ctFiles:
        dataset = pydicom.dcmread(file)
        if (dataset.Modality == "CT"):
            ctFiles_filtered.append(file)
    roi_names = ['Celiac', 'DoseFalloff', 'Heart', 'KidneyL', 'KidneyR', 'LargeBowel',
                 'Liver', 'PTV', 'SMASMV', 'Skin', 'SmallBowel', 'SpinalCord',
                 'Stomach', 'duodenum']
    outputFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/Liver_view_dcm"
    dicomView(ctFiles_filtered, rtFile, outputFolder, roi_names)


def PROSTATE_view():
    binaryDir = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_binary"
    shape = [90, 184, 184]
    files = os.listdir(binaryDir)
    density_name = "density"
    
    densityFile = os.path.join(binaryDir, density_name + ".bin")
    density = np.fromfile(densityFile, dtype=np.uint16)
    density = np.reshape(density, shape)
    density = np.transpose(density, axes=(0, 2, 1))
    viewDir = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_view"
    if not os.path.isdir(viewDir):
        os.mkdir(viewDir)
    
    voi_names = [a.split(".")[0] for a in files if density_name not in a]
    voi_names = ['BODY', 'Bladder', 'Lt_femoral_head', 'Lymph_Nodes', 'PTV_56', 'PTV_68',
                 'Penile_bulb', 'Rectum', 'Rt_femoral_head', 'prostate_bed']
    masks = []
    for name in voi_names:
        file = os.path.join(binaryDir, name + ".bin")
        mask = np.fromfile(file, dtype=np.uint8)
        mask = np.reshape(mask, shape)
        mask = np.transpose(mask, (0, 2, 1))
        # if name == 'Lymph_Nodes':
        #     name = 'PTV_68'
        # elif name == 'PTV_68':
        #     name = 'Lymph_Nodes'
        masks.append((name, mask))

    n_structures = len(voi_names)
    color_values = np.linspace(0, 1, n_structures)
    color_map = plt.get_cmap()
    colors = [color_map(a) for a in color_values]
    
    for i in range(shape[0]):
        density_slice = density[i, :, :]
        plt.imshow(density_slice, cmap="gray")
        for j in range(len(masks)):
            name = masks[j][0]
            mask = masks[j][1]
            mask_slice = mask[i, :, :]
            if (np.sum(mask_slice) == 0):
                continue
            contours = measure.find_contours(mask_slice, 0.5)
            initial = True
            for contour in contours:
                if (initial):
                    plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color=colors[j], label=name)
                    initial = False
                else:
                    plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color=colors[j])
        plt.legend()
        outputFile = os.path.join(viewDir, "{:03d}.png".format(i))
        plt.savefig(outputFile)
        plt.clf()
        print(outputFile)


def PROSTATE_to_dicom():
    target_shape = [90, 184, 184]
    targetFile = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_binary/density.bin"
    density = np.fromfile(targetFile, dtype=np.uint16)
    density = np.reshape(density, target_shape)

    outputFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_view"
    if not os.path.isdir(outputFolder):
        os.mkdir(outputFolder)

    templateFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/Liver_dicom"
    numSlices = 169
    dataOrder = []

    PixelSpacing = [3.0, 3.0]
    st_number = 3.0
    SliceThickness = '3.0'
    SpacingBetweenSlices = '0.0'
    ImagePositionPatient = [-325, -358.8, 117]
    for i in range(1, numSlices+1):
        file = os.path.join(templateFolder, "anon{}.dcm".format(i))
        dataset = pydicom.dcmread(file)
        if (dataset.Modality != "CT"):
            continue
        InstanceNumber = int(dataset.InstanceNumber)
        dataOrder.append((InstanceNumber, dataset))
    dataOrder.sort(key = lambda a: a[0])

    for idx in range(target_shape[0]):
        dataset = dataOrder[idx][1]
        dataset.PixelSpacing = PixelSpacing
        dataset.SliceThickness = SliceThickness
        dataset.SpacingBetweenSlices = SpacingBetweenSlices
        dataset.InstanceNumber = str(idx)
        dataset.ImagePositionPatient = ImagePositionPatient
        dataset.ImagePositionPatient[2] += idx * st_number
        dataset.Rows = target_shape[1]
        dataset.Columns = target_shape[2]

        slice = density[idx, :, :]
        slice = np.ascontiguousarray(slice)
        dataset.PixelData = slice
        file = os.path.join(outputFolder, "{:03d}.dcm".format(idx))
        dataset.save_as(file)
        print(file)


def PROSTATE_RT():
    binaryFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_binary"
    dicomFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_dicom"
    rtstruct = RTStructBuilder.create_new(dicomFolder)
    density_file = "density.bin"
    files = os.listdir(binaryFolder)
    files.remove(density_file)
    shape = [90, 184, 184]

    files = ["PTV_56.bin"]
    for file in files:
        name = file.split(".")[0]
        path = os.path.join(binaryFolder, file)
        mask = np.fromfile(path, dtype=np.uint8)
        mask = mask > 0
        mask = np.reshape(mask, shape)
        mask = np.transpose(mask, axes=(2, 1, 0))

        if True:
            # For debug purposes
            outputFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_sanity"
            if not os.path.isdir(outputFolder):
                os.mkdir(outputFolder)
            for i in range(shape[0]):
                filename = os.path.join(outputFolder, "{:03d}.png".format(i))
                plt.imsave(filename, mask[:, :, i])
                print(filename)
            return

        rtstruct.add_roi(mask=mask, name=name)

        if False:
            mask_new = rtstruct.get_roi_mask_by_name(name)
            outputFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_sanity"
            if not os.path.isdir(outputFolder):
                os.mkdir(outputFolder)
            for i in range(shape[0]):
                filename = os.path.join(outputFolder, "{:03d}.png".format(i))
                plt.imsave(filename, mask_new[:, :, i])
                print(filename)
            return

    destFile = os.path.join(dicomFolder, "prostRT.dcm")
    rtstruct.save(destFile)


def PROSTATE_dicom_sanity_check():
    dicomFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_dicom"
    rtFile = os.path.join(dicomFolder, "prostRT.dcm")
    rtstruct = RTStructBuilder.create_from(dicom_series_path=dicomFolder, rt_struct_path=rtFile)
    voi_list = rtstruct.get_roi_names()
    name = "PTV_56"
    assert name in voi_list, "{} not contained".format(name)
    obj_mask = rtstruct.get_roi_mask_by_name(name)
    outputFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_sanity"
    for i in range(obj_mask.shape[2]):
        file = os.path.join(outputFolder, "{:03d}.png".format(i))
        plt.imsave(file, obj_mask[:, :, i])
        print(file)

    # outputFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_sanity"
    # if not os.path.isdir(outputFolder):
    #     os.mkdir(outputFolder)
    # dicomFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_dicom"
    # numSlices = 90
    # for i in range(numSlices):
    #     file = os.path.join(dicomFolder, "{:03d}.dcm".format(i))
    #     dataset = pydicom.dcmread(file)
    #     if (dataset.Modality != "CT"):
    #         continue
    #     pixel_array = pydicom.dcmread(file).pixel_array
    #     outputFile = os.path.join(outputFolder, "{:03d}.png".format(i))
    #     plt.imsave(outputFile, pixel_array)
    #     print(outputFile)
        

def studyRTstruct():
    exampleFile = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_dicom/006.dcm"
    dataset = pydicom.dcmread(exampleFile)
    print("Great!")


def rtstrut_test():
    destFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_LIVER"

    if False:
        # process liver dicom files
        liverFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/Liver_dicom"
        files = os.listdir(liverFolder)
        files.sort()
        order = []
        for file in files:
            path = os.path.join(liverFolder, file)
            dataset = pydicom.dcmread(path)
            if dataset.Modality != "CT":
                continue
            InstanceNumber = int(dataset.InstanceNumber)
            order.append((InstanceNumber, path))
        order.sort(key=lambda a: a[0])
        
        numberWeWant = 90
        order = order[:numberWeWant]
        if not os.path.isdir(destFolder):
            os.mkdir(destFolder)
        for i in range(numberWeWant):
            command = "cp {} {}".format(order[i][1], os.path.join(destFolder, "{:03d}.dcm".format(i)))
            os.system(command)
            print(command)
    
    if False:
        # generate the rtstruct file using the dicoms generated above
        binaryFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_binary"
        maskFiles = os.listdir(binaryFolder)
        maskShape = [90, 184, 184]
        densityFile = "density.bin"
        maskFiles.remove(densityFile)

        dicomFiles = os.listdir(destFolder)
        exampleDicom = dicomFiles[0]
        exampleDicom = os.path.join(destFolder, exampleDicom)
        shape = pydicom.dcmread(exampleDicom).pixel_array.shape
        outputShape = [maskShape[0], shape[0], shape[1]]

        rtstruct = RTStructBuilder.create_new(dicom_series_path=destFolder)
        
        for file in maskFiles:
            name = file.split(".")[0]
            path = os.path.join(binaryFolder, file)
            mask = np.fromfile(path, dtype=np.uint8)
            mask = mask > 0
            mask = mask * 255
            mask = mask.astype(np.uint8)
            mask = np.reshape(mask, maskShape)
            mask = transform.resize(mask, outputShape)
            mask = np.transpose(mask, axes=(2, 1, 0))
            mask = np.flip(mask, axis=2)
            mask = mask > 0.5

            if False:
                # for debug purposes
                imageFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROLIVER_show"
                if not os.path.isdir(imageFolder):
                    os.mkdir(imageFolder)
                for i in range(mask.shape[2]):
                    file = os.path.join(imageFolder, "{:03d}.png".format(i))
                    plt.imsave(file, mask[:, :, i])
                    print(file)
                return

            rtstruct.add_roi(mask=mask, name=name)

            if False:
                # for debug purposes
                imageFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROLIVER_show"
                if not os.path.isdir(imageFolder):
                    os.mkdir(imageFolder)
                mask = rtstruct.get_roi_mask_by_name(name)
                for i in range(mask.shape[2]):
                    file = os.path.join(imageFolder, "{:03d}.png".format(i))
                    plt.imsave(file, mask[:, :, i])
                    print(file)
                return
        rtFile = os.path.join("/data/qifan/projects/FastDoseWorkplace/PlanTune/"
                              "PROSTATE_dicom_v2", "prostRT.dcm")
        rtstruct.save(rtFile)
        print(rtFile)
    
    if True:
        # try to combine the generated rtstruct with the CT dicom files
        dicomFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_dicom_v2"
        rtFile = os.path.join(dicomFolder, "prostRT.dcm")
        files = os.listdir(dicomFolder)
        ctFiles = []
        for file in files:
            path = os.path.join(dicomFolder, file)
            dataset = pydicom.dcmread(path)
            if dataset.Modality == "CT":
                ctFiles.append(path)
        outputFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_view"
        rtList = ['BODY', 'Bladder', 'Lt_femoral_head', 'Lymph_Nodes',
                  'PTV_56', 'PTV_68', 'Penile_bulb', 'Rectum', 'Rt_femoral_head', 'prostate_bed']
        dicomView(ctFiles, rtFile, outputFolder, rtList)

    if False:
        # Seems that the dicom generated is problematic. Here we just insert the arrays
        # into the correct template
        prostateShape = [90, 184, 184]
        prostateDensity = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_binary/density.bin"
        prostateDensity = np.fromfile(prostateDensity, dtype=np.uint16)
        prostateDensity = np.reshape(prostateDensity, prostateShape)
        prostateDensity = np.transpose(prostateDensity, axes=(0, 2, 1))
        
        targetFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_dicom_v2"
        if not os.path.isdir(targetFolder):
            os.mkdir(targetFolder)
        
        sourceFiles = os.listdir(destFolder)
        sourceFiles.sort()
        sourceOrder = []
        for file in sourceFiles:
            path = os.path.join(destFolder, file)
            dataset = pydicom.dcmread(path)
            if (dataset.Modality != "CT"):
                continue
            InstanceNumber = int(dataset.InstanceNumber)
            sourceOrder.append((InstanceNumber, dataset))
        sourceOrder.sort(key=lambda a: a[0])

        for i in range(len(sourceOrder)):
            dataset = sourceOrder[i][1]
            shape = dataset.pixel_array.shape
            densitySlice = prostateDensity[i, :, :]
            densitySlice = transform.resize(densitySlice, shape)
            densitySlice *= 65535
            densitySlice = densitySlice.astype(np.uint16)
            dataset.PixelData = densitySlice
            datasetFile = os.path.join(targetFolder, "{:03d}.dcm".format(i))
            dataset.save_as(datasetFile)
            print(datasetFile)

    if False:
        exampleFile = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_dicom_v2/004.dcm"
        dataset = pydicom.dcmread(exampleFile)
        print("Great!")


def generate_PROSTATE_dicom_v2():
    """
    After the experiments above, we successfully generated a series of CT dicom images
    that are compatible with the rtstruct. Here we generate the dicom files with the 
    correct geometry parameters
    """
    densityShape = [90, 184, 184]
    targetFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_dicom_v2"
    if not os.path.isdir(targetFolder):
        os.mkdir(targetFolder)
    
    if False:
        templateFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_LIVER"
        densityFile = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_binary/density.bin"
        density = np.fromfile(densityFile, dtype=np.uint16)
        density = np.reshape(density, densityShape)
        density = np.transpose(density, axes=[0, 2, 1])

        basePosition = [-325, -354.8, 117]
        sliceThickness = 3.0
        pixelSpacingScalar = 3.0 * densityShape[1] / 512

        for i in range(densityShape[0]):
            templateFile = os.path.join(templateFolder, "{:03d}.dcm".format(i))
            dataset = pydicom.dcmread(templateFile)
            dataset.ImagePositionPatient = basePosition
            dataset.ImagePositionPatient[2] = basePosition[2] - i * sliceThickness
            dataset.SliceLocation = str(dataset.ImagePositionPatient[2])
            dataset.SliceThickness = str(sliceThickness)
            dataset.PixelSpacing = [pixelSpacingScalar, pixelSpacingScalar]
            
            densitySlice = density[i, :, :]
            newShape = dataset.pixel_array.shape
            densitySlice = transform.resize(densitySlice, newShape)
            densitySlice *= 65535
            densitySlice = densitySlice.astype(np.uint16)
            dataset.PixelData = densitySlice
            dataFile = os.path.join(targetFolder, "{:03d}.dcm".format(i))
            dataset.save_as(dataFile)
            print(dataFile)
    
    if True:
        binaryFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/PROSTATE_binary"
        binaryFiles = os.listdir(binaryFolder)
        densityFile = "density.bin"
        binaryFiles.remove(densityFile)
        outputShape = (densityShape[0], 512, 512)

        rtstruct = RTStructBuilder.create_new(dicom_series_path=targetFolder)
        for file in binaryFiles:
            name = file.split(".")[0]
            path = os.path.join(binaryFolder, file)
            mask = np.fromfile(path, dtype=np.uint8)
            mask = mask > 0
            mask = mask * 255
            mask = mask.astype(np.uint8)
            mask = np.reshape(mask, densityShape)
            mask = transform.resize(mask, outputShape)
            mask = np.transpose(mask, axes=(2, 1, 0))
            mask = np.flip(mask, axis=2)
            mask = mask > 0.5
            rtstruct.add_roi(mask=mask, name=name)
            print(name)
        rtPath = os.path.join(targetFolder, "prostRT.dcm")
        rtstruct.save(rtPath)


def dataClean():
    if False:
        # clean head-and-neck patients
        sourceFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/HN_dicom/000000"
        files = os.listdir(sourceFolder)
        ctFiles = []
        rtFiles = []
        for file in files:
            path = os.path.join(sourceFolder, file)
            dataset = pydicom.dcmread(path)
            if (dataset.Modality == "CT"):
                InstanceNumber = int(dataset.InstanceNumber)
                ctFiles.append((InstanceNumber, path))
            else:
                rtFiles.append(path)
        ctFiles.sort(key = lambda a: a[0])
        rtFiles.sort()
        
        targetFolder = "/data/qifan/projects/FastDoseWorkplace/CORTclean"
        targetFolder = os.path.join(targetFolder, "HeadNeck")
        if not os.path.isdir(targetFolder):
            os.mkdir(targetFolder)
        for i, entry in enumerate(ctFiles):
            InstanceNumber, sourceFile = entry
            targetFile = os.path.join(targetFolder, "{:03d}.dcm".format(i))
            command = "cp {} {}".format(sourceFile, targetFile)
            os.system(command)
            print(command)
        
        # save rtstruct
        sourceFile = os.path.join(sourceFolder, rtFiles[0])
        targetFile = os.path.join(targetFolder, "RTstruct_HN.dcm")
        command = "cp {} {}".format(sourceFile, targetFile)
        os.system(command)
        print(command)
    
    if False:
        # check the generated folder
        dicomFolder = "/data/qifan/projects/FastDoseWorkplace/CORTclean/HeadNeck"
        imageFolder = "/data/qifan/projects/FastDoseWorkplace/CORTclean/HeadNeck_view"
        files = os.listdir(dicomFolder)
        ctFiles = []
        rtFile = None
        for file in files:
            path = os.path.join(dicomFolder, file)
            dataset = pydicom.dcmread(path)
            if dataset.Modality == "CT":
                ctFiles.append(path)
            else:
                rtFile = path
        rtList = ['BRAIN_STEM', 'BRAIN_STEM_PRV', 'CEREBELLUM', 'CHIASMA', 'LARYNX', 'LENS_LT',
                  'LENS_RT', 'LIPS', 'OPTIC_NRV_LT', 'OPTIC_NRV_RT', 'PAROTID_LT', 'PAROTID_RT', 
                  'SKIN', 'SPINAL_CORD', 'SPINL_CRD_PRV', 'TEMP_LOBE_LT', 'TEMP_LOBE_RT',
                  'TM_JOINT_LT', 'TM_JOINT_RT', 'PTV_crop']
        dicomView(ctFiles, rtFile, imageFolder, rtList)
    
    if False:
        # copy liver case
        sourceFolder = "/data/qifan/projects/FastDoseWorkplace/PlanTune/Liver_dicom"
        files = os.listdir(sourceFolder)
        ctFiles = []
        rtFile = None
        for file in files:
            path = os.path.join(sourceFolder, file)
            dataset = pydicom.dcmread(path)
            if (dataset.Modality == "CT"):
                InstanceNumber = int(dataset.InstanceNumber)
                ctFiles.append((InstanceNumber, path))
            else:
                rtFile = path
        ctFiles.sort(key = lambda a: a[0])

        targetFolder = "/data/qifan/projects/FastDoseWorkplace/CORTclean"
        targetFolder = os.path.join(targetFolder, "Liver")
        if not os.path.isdir(targetFolder):
            os.mkdir(targetFolder)
        for i, entry in enumerate(ctFiles):
            InstanceNumber, sourceFile = entry
            targetFile = os.path.join(targetFolder, "{:03d}.dcm".format(i))
            command = "cp {} {}".format(sourceFile, targetFile)
            os.system(command)
            print(command)
        
        # save rtstruct
        sourceFile = os.path.join(sourceFolder, rtFile)
        targetFile = os.path.join(targetFolder, "RTstruct_Liver.dcm")
        command = "cp {} {}".format(sourceFile, targetFile)
        os.system(command)
        print(command)
    
    if False:
        # view Liver images
        dicomFolder = "/data/qifan/projects/FastDoseWorkplace/CORTclean/Liver"
        imageFolder = "/data/qifan/projects/FastDoseWorkplace/CORTclean/Liver_view"
        files = os.listdir(dicomFolder)
        ctFiles = []
        rtFile = None
        for file in files:
            path = os.path.join(dicomFolder, file)
            dataset = pydicom.dcmread(path)
            if dataset.Modality == "CT":
                ctFiles.append(path)
            else:
                rtFile = path
        rtList = ['Celiac', 'DoseFalloff', 'Heart', 'KidneyL', 'KidneyR', 'LargeBowel',
                  'Liver', 'PTV', 'SMASMV', 'Skin', 'SmallBowel', 'SpinalCord', 'Stomach', 'duodenum']
        dicomView(ctFiles, rtFile, imageFolder, rtList)
    
    if True:
        # view prostate images
        dicomFolder = "/data/qifan/projects/FastDoseWorkplace/CORTclean/Prostate"
        imageFolder = "/data/qifan/projects/FastDoseWorkplace/CORTclean/Prostate_view"
        files = os.listdir(dicomFolder)
        ctFiles = []
        rtFile = None
        for file in files:
            path = os.path.join(dicomFolder, file)
            dataset = pydicom.dcmread(path)
            if dataset.Modality == "CT":
                ctFiles.append(path)
            else:
                rtFile = path
        rtList = ['BODY', 'Bladder', 'Lt_femoral_head', 'Lymph_Nodes', 'PTV_56',
                  'PTV_68', 'Penile_bulb', 'Rectum', 'Rt_femoral_head', 'prostate_bed']
        dicomView(ctFiles, rtFile, imageFolder, rtList)


if __name__ == '__main__':
    # HN_dicom_Examine()
    # Liver_dicom_Examine()
    # Liver_mask_Examine()
    # convert_liverRT_to_dicom()
    # verify_liverRT()
    # PROSTATE_view()
    # PROSTATE_to_dicom()
    # PROSTATE_RT()
    # PROSTATE_dicom_sanity_check()
    # studyRTstruct()
    # rtstrut_test()
    # generate_PROSTATE_dicom_v2()
    dataClean()