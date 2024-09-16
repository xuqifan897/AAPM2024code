import os
import numpy as np
import h5py
import json
from typing import List
import functools

def readKernel():
    kernelFile = "/data/qifan/projects/FastDoseWorkplace/Pancreas" \
        "/plansSIB/Patient001/QihuiRyan/preprocess/cumulative_kernel.h5"
    with h5py.File(kernelFile, "r") as hdf:
        keys = hdf.attrs.keys()
        for key in keys:
            value = hdf.attrs[key]
            print(key, value)


def print_hdf5_structure(name, obj):
    """Callback function to print the structure of the HDF5 file."""
    print(f"{name}: {'Group' if isinstance(obj, h5py.Group) else 'Dataset'}")

    # If the object has attributes, print them
    if obj.attrs:
        print(f"  Attributes of {name}:")
        for attr_name, attr_value in obj.attrs.items():
            print(f"    {attr_name}: {attr_value}")


def show_hdf5_structure(kernelFile):
    with h5py.File(kernelFile, "r") as hdf:
        hdf.visititems(print_hdf5_structure)


class kernelLaunchEntry:
    def __init__(self):
        self.start = 0  # ns
        self.end = 0  # ns
        self.deviceId = 0
        self.contextId = 0
        self.streamId = 0
        self.correlationId = 0
        self.globalPid = 0
        self.demangleName = 0
        self.shortName = 0
        self.mangleName = 0
        self.launchType = 0
        self.launchConfig = 0
        self.registersPerThread = 0
        self.gridX = 0
        self.gridY = 0
        self.gridZ = 0
        self.blockX = 0
        self.blockY = 0
        self.blockZ = 0
        self.staticSharedMemory = 0
        self.dynamicSharedMemory = 0
        self.localMemoryPerThread = 0
        self.localMemoryTotal = 0
        self.gridIdx = 0
        self.sharedMemoryExecuted = 0
        self.graphNodeId = 0
        self.sharedMemoryLimitConfig = 0

    staticmethod
    def convertToStructList(inputList):
        outputList = []
        for entry in inputList:
            newEntry = kernelLaunchEntry()
            [newEntry.start, newEntry.end, newEntry.deviceId, newEntry.contextId, newEntry.streamId, newEntry.correlationId,
            newEntry.globalPid, newEntry.demangleName, newEntry.shortName, newEntry.mangleName, newEntry.launchType,
            newEntry.launchConfig, newEntry.registersPerThread, newEntry.gridX, newEntry.gridY,
            newEntry.gridZ, newEntry.blockX, newEntry.blockY, newEntry.blockZ, newEntry.staticSharedMemory,
            newEntry.dynamicSharedMemory, newEntry.localMemoryPerThread, newEntry.localMemoryTotal,
            newEntry.gridIdx, newEntry.sharedMemoryExecuted, newEntry.graphNodeId,
            newEntry.sharedMemoryLimitConfig] = entry
            outputList.append(newEntry)
        return outputList
    
    def print(self):
        currentVars = vars(self)
        currentVars = ["  {}: {}".format(key, value) for key, value in currentVars.items()]
        currentVars = ",".join(currentVars)
        print(currentVars)


def analyzeProfilingRyan():
    resultFolder = "/data/qifan/projects/FastDoseWorkplace/profiling"
    resultFile = "Ryan_dosecalc_nsys.h5"
    resultFile = os.path.join(resultFolder, resultFile)
    hdf5Dict = {}
    def loadH5toDict(name, obj):
        hdf5Dict[name] = np.array(obj)
    with h5py.File(resultFile, "r") as hdf:
        hdf.visititems(loadH5toDict)
    keys = ['ANALYSIS_DETAILS', 'COMPOSITE_EVENTS', 'CUPTI_ACTIVITY_KIND_KERNEL',
            'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET', 'CUPTI_ACTIVITY_KIND_RUNTIME',
            'CUPTI_ACTIVITY_KIND_SYNCHRONIZATION', 'ENUM_CUDA_DEV_MEM_EVENT_OPER',
            'ENUM_CUDA_FUNC_CACHE_CONFIG', 'ENUM_CUDA_KERNEL_LAUNCH_TYPE', 'ENUM_CUDA_MEMCPY_OPER',
            'ENUM_CUDA_MEMPOOL_OPER', 'ENUM_CUDA_MEMPOOL_TYPE', 'ENUM_CUDA_MEM_KIND',
            'ENUM_CUDA_SHARED_MEM_LIMIT_CONFIG', 'ENUM_CUDA_UNIF_MEM_ACCESS_TYPE', 'ENUM_CUDA_UNIF_MEM_MIGRATION',
            'ENUM_CUPTI_STREAM_TYPE', 'ENUM_CUPTI_SYNC_TYPE', 'ENUM_D3D12_CMD_LIST_TYPE', 'ENUM_D3D12_HEAP_FLAGS',
            'ENUM_D3D12_HEAP_TYPE', 'ENUM_D3D12_PAGE_PROPERTY', 'ENUM_DXGI_FORMAT', 'ENUM_ETW_MEMORY_TRANSFER_TYPE',
            'ENUM_GPU_CTX_SWITCH', 'ENUM_NET_DEVICE_ID', 'ENUM_NET_LINK_TYPE', 'ENUM_NET_VENDOR_ID',
            'ENUM_NSYS_EVENT_CLASS', 'ENUM_NSYS_EVENT_TYPE', 'ENUM_NVDRIVER_EVENT_ID', 'ENUM_OPENACC_DEVICE',
            'ENUM_OPENACC_EVENT_KIND', 'ENUM_OPENGL_DEBUG_SEVERITY', 'ENUM_OPENGL_DEBUG_SOURCE',
            'ENUM_OPENGL_DEBUG_TYPE', 'ENUM_OPENMP_DISPATCH', 'ENUM_OPENMP_EVENT_KIND', 'ENUM_OPENMP_MUTEX',
            'ENUM_OPENMP_SYNC_REGION', 'ENUM_OPENMP_TASK_FLAG', 'ENUM_OPENMP_TASK_STATUS', 'ENUM_OPENMP_THREAD',
            'ENUM_OPENMP_WORK', 'ENUM_SAMPLING_THREAD_STATE', 'ENUM_SLI_TRANSER', 'ENUM_STACK_UNWIND_METHOD',
            'ENUM_VULKAN_PIPELINE_CREATION_FLAGS', 'ENUM_WDDM_ENGINE_TYPE', 'ENUM_WDDM_INTERRUPT_TYPE',
            'ENUM_WDDM_PACKET_TYPE', 'ENUM_WDDM_PAGING_QUEUE_TYPE', 'ENUM_WDDM_VIDMM_OP_TYPE', 'EXPORT_META_DATA',
            'OSRT_API', 'OSRT_CALLCHAINS', 'PROCESSES', 'PROFILER_OVERHEAD', 'ProcessStreams', 'SAMPLING_CALLCHAINS',
            'SCHED_EVENTS', 'StringIds', 'TARGET_INFO_CUDA_NULL_STREAM', 'TARGET_INFO_CUDA_STREAM',
            'TARGET_INFO_GPU', 'TARGET_INFO_SESSION_START_TIME', 'TARGET_INFO_SYSTEM_ENV', 'ThreadNames',
            'UnwindMethodType']
    
    kernelLaunchInfo = hdf5Dict['CUPTI_ACTIVITY_KIND_KERNEL']
    kernelLaunchInfo = kernelLaunchEntry.convertToStructList(kernelLaunchInfo)
    beamletRayTrace = ("beamletRayTrace", 1592)
    packRowConvolve = ("packRowConvolve", 1595)
    revToBev = ("revToBev", 1597)
    unpackBevDosePillar = ("unpackBevDosePillar", 1600)
    statDict = {beamletRayTrace: [], packRowConvolve: [], revToBev: [], unpackBevDosePillar: []}
    for entry in kernelLaunchInfo:
        for key, collection in statDict.items():
            if entry.shortName == key[1]:
                collection.append(entry)
    
    averageTimeDict = {beamletRayTrace: 0, packRowConvolve: 0, revToBev: 0, unpackBevDosePillar: 0}
    print("Kernel Launch Info")
    for key in averageTimeDict:
        collection = statDict[key]
        timeTotal = 0
        for entry in collection:
            timeTotal += entry.end - entry.start
        timeTotal *= 1e-6
        timeAvg = timeTotal / len(collection)
        averageTimeDict[key] = timeAvg
        print("| {} | {} | {:.3f} | {:.3f} |".format(key[0], len(collection), timeAvg, timeTotal))
    computationTime = statDict[revToBev][-1].end - statDict[beamletRayTrace][0].start
    computationTime *= 1e-6  # ns to ms
    print("Total computation time: {:.4f}".format(computationTime))
    print("\n")

    print("Memcpy Info")
    MemcpyInfo = hdf5Dict["CUPTI_ACTIVITY_KIND_MEMCPY"]
    if False:
        streamIdxList = [a[4] for a in MemcpyInfo]
        print(streamIdxList)
        return
    relevantStreamList = [17, 18, 19, 20]
    MemcpyInfo = [a for a in MemcpyInfo if a[4] in relevantStreamList]
    timeList = []
    for line in MemcpyInfo:
        duration = line[1] - line[0]
        duration *= 1e-6  # ns to ms
        timeList.append(duration)
    print(len(MemcpyInfo), np.mean(timeList), np.sum(timeList))
    print("\n")

    print("Sparsification")
    sparseList = []
    for i in range(len(MemcpyInfo) - 1):
        timeGap = MemcpyInfo[i+1][0] - MemcpyInfo[i][0]
        timeGap *= 1e-6  # ns to ms
        sparseList.append(timeGap)
    print(len(sparseList), np.mean(sparseList), np.sum(sparseList))


def analyzeProfilingFastDose():
    resultFolder = "/data/qifan/projects/FastDoseWorkplace/profiling"
    resultFile = os.path.join(resultFolder, "IMRT_dosecalc_nsys.h5")
    hdf5Dict = {}
    def loadH5toDict(name, obj):
        hdf5Dict[name] = np.array(obj)
    with h5py.File(resultFile, "r") as hdf:
        hdf.visititems(loadH5toDict)
    
    kernelLaunchInfo = hdf5Dict['CUPTI_ACTIVITY_KIND_KERNEL']
    kernelLaunchInfo = kernelLaunchEntry.convertToStructList(kernelLaunchInfo)
    if False:
        for a in kernelLaunchInfo:
            a.print()
            print("\n")
        return

    termaCompute = ("termaCompute", 1824)
    doseCompute = ("doseCompute", 1826)
    interpArrayPrep = ("interpArrayPrep", 1830)
    superVoxelInterp = ("superVoxelInterp", 1832)
    voxelInterp = ("voxelInterp", 1834)
    cusparseCountNz = ("cusparseCountNz", 1836)
    deviceScanInit = ("deviceScanInit", 1838)
    deviceScan = ("deviceScan", 1840)
    cusparseGatherNz = ("cusparseGatherNz", 1842)

    statDict = {termaCompute: [], doseCompute: [], interpArrayPrep: [],
        superVoxelInterp: [], voxelInterp: [], cusparseCountNz: [],
        deviceScanInit: [], deviceScan: [], cusparseGatherNz: []}
    for entry in kernelLaunchInfo:
        for key in statDict:
            if entry.shortName == key[1]:
                statDict[key].append(entry)
    
    print("Kernel Launch Info")
    for key in statDict:
        collection = statDict[key]
        timeTotal = 0
        for entry in collection:
            timeTotal += entry.end - entry.start
        timeTotal *= 1e-6  # ns to ms
        timeAvg = timeTotal / len(collection)
        print("| {} | {} | {:.3f} | {:.3f} |".format(
            key[0], len(collection), timeAvg, timeTotal))
    computationTime = statDict[cusparseGatherNz][-1].end - statDict[termaCompute][0].start
    computationTime *= 1e-6  # ns to ms
    print("Total computation time: {:.4f}".format(computationTime))
    print("\n")


class ncuOneLine:
    def __init__(self):
        self.ID = 0
        self.ProcessID = 0
        self.ProcessName = 0
        self.HostName = 0
        self.KernelName = 0
        self.Context = 0
        self.Stream = 0
        self.BlockSize = 0
        self.GridSize = 0
        self.Device = 0
        self.CC = 0
        self.SectionName = 0
        self.MetricName = 0
        self.MetricUnit = 0
        self.MetricValue = 0

    def parse(self, line):
        line = eval("[{}]".format(line))
        line = line[:15]
        line_parsed = []
        for a in line:
            if is_number(a):
                line_parsed.append(eval(a))
            else:
                line_parsed.append(a)
        self.ID, self.ProcessID, self.ProcessName, self.HostName, self.KernelName, \
        self.Context, self.Stream, self.BlockSize, self.GridSize, self.Device, self.CC, \
        self.SectionName, self.MetricName, self.MetricUnit, self.MetricValue = line_parsed
    
    def print(self):
        print(vars(self))


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def ncuResultAnalysis():
    # sourceFile = "/data/qifan/projects/FastDoseWorkplace/profiling/Ryan_dosecalc.csv"
    sourceFile = "/data/qifan/projects/FastDoseWorkplace/profiling/IMRT_dosecalc.csv"
    with open(sourceFile, "r") as f:
        sourceLines = f.readlines()
    if False:
        title = eval("[{}]".format(sourceLines[0]))
        title = title[:-4]
        content = []
        for entry in title:
            line = "self.{} = 0".format(entry.replace(" ", ""))
            content.append(line)
        content = "\n".join(content)
        print(content)

        content = []
        for entry in title:
            line = "self.{}".format(entry.replace(" ", ""))
            content.append(line)
        content = ", ".join(content)
        print(content)
        return
    
    sourceLines = sourceLines[1: ]  # remove the title
    sourceLines = [a for a in sourceLines if a != "\n"]
    destLines: List[ncuOneLine] = []
    for i, line in enumerate(sourceLines):
        entry = ncuOneLine()
        entry.parse(line)
        destLines.append(entry)
    
    stats = {}  # structure: {kernelName: {ID: {metricName: value}}}
    metricSet = set()
    for entry in destLines:
        ID = entry.ID
        name = entry.KernelName
        if name not in stats:
            stats[name] = {}
        if ID not in stats[name]:
            stats[name][ID] = {}
        metricName = entry.MetricName
        if metricName == '':
            continue
        if metricName not in metricSet:
            metricSet.add(metricName)
        value = entry.MetricValue
        stats[name][ID][metricName] = value
    
    metricSetList = ["Active Warps Per Scheduler", "Eligible Warps Per Scheduler",
        "One or More Eligible", "No Eligible", "Issued Warp Per Scheduler", 
        "Avg. Active Threads Per Warp", "Avg. Not Predicated Off Threads Per Warp",
        "Theoretical Occupancy", "Achieved Occupancy", "Theoretical Active Warps per SM",
        "Achieved Active Warps Per SM", "Warp Cycles Per Issued Instruction",
        "Warp Cycles Per Executed Instruction", ]
    
    firstLine = ["Kernel Name \\ Metrics"] + metricSetList
    secondLine = [" - " for entry in firstLine]
    firstLine = "| " + " | ".join(firstLine) + " |"
    secondLine = "| " + " | ".join(secondLine) + " |"
    content = [firstLine, secondLine]
    for kernelName in stats:
        kernelSub = stats[kernelName]
        statList = {a: [] for a in metricSetList}
        for id, idDict in kernelSub.items():
            for key, value in idDict.items():
                if key in statList:
                    statList[key].append(value)
        for key in statList:
            keyList = statList[key]
            mean, std = np.mean(keyList), np.std(keyList)
            statList[key] = (mean, std)
        
        kernelNameShort = kernelName.split("(")[0]
        currentLine = [kernelNameShort]
        for key in metricSetList:
            currentLine.append("{:.4f} $\pm$ {:.4f}".format(*statList[key]))
        currentLine = "|" + " | ".join(currentLine) + "|"
        content.append(currentLine)
    content = "\n".join(content)
    print(content)


def analyzeProfilingOptimize():
    resultFolder = "/data/qifan/projects/FastDoseWorkplace/profiling"
    resultFile = os.path.join(resultFolder, "IMRTOpt_nsys.h5")
    hdf5Dict = {}
    def loadH5toDict(name, obj):
        hdf5Dict[name] = np.array(obj)
    with h5py.File(resultFile, "r") as hdf:
        hdf.visititems(loadH5toDict)
    stringIds = hdf5Dict["StringIds"]
    stringIds = {a: b for a, b in stringIds}
    kernelLaunchInfo = hdf5Dict['CUPTI_ACTIVITY_KIND_KERNEL']
    kernelLaunchInfo: List[kernelLaunchEntry] = kernelLaunchEntry.convertToStructList(kernelLaunchInfo)
    if False:
        # name stat
        nameStat = {}
        for entry in kernelLaunchInfo:
            shortName = entry.shortName
            if shortName in nameStat:
                nameStat[shortName] += 1
            else:
                nameStat[shortName] = 1
        for shortNameId in nameStat:
            shortName = stringIds[shortNameId]
            print(shortNameId, shortName)

    # find the main kernel for sparse matrix multiplication
    csrmv_v3_kernel_binary = "csrmv_v3_kernel"
    csrmv_v3_kernel_binary = csrmv_v3_kernel_binary.encode('utf-8')
    csrmv_v3_kernel_shortName = None
    for key, value in stringIds.items():
        if value == csrmv_v3_kernel_binary:
            csrmv_v3_kernel_shortName = key
            break
    assert csrmv_v3_kernel_shortName is not None
    
    # get the statistics of all kernels
    statDict = {}
    for entry in kernelLaunchInfo:
        shortNameId = entry.shortName
        if shortNameId in statDict:
            statDict[shortNameId].append(entry)
        else:
            statDict[shortNameId] = [entry]

    def customComp(str1, str2):
        str1 = str1[0]
        str2 = str2[0]
        keyWord = "d_"
        if (keyWord in str1) and (keyWord not in str2):
            return -1
        elif (keyWord not in str1) and (keyWord in str2):
            return 1
        elif str1 > str2:
            return -1
        elif str1 < str2:
            return 1
        else:
            return 0
    
    if False:
        # total statistics
        lineList = []
        for key in statDict:
            collection = statDict[key]
            timeTotal = 0
            for entry in collection:
                timeTotal += entry.end - entry.start
            timeTotal *= 1e-6
            timeAvg = timeTotal / len(collection)

            shortName = stringIds[key].decode('utf-8')
            lineList.append((shortName, len(collection), timeAvg, timeTotal))
        lineList.sort(key=functools.cmp_to_key(customComp), reverse=True)
        for entry in lineList:
            line = "| {} | {} | {:.3f} | {:.3f} |".format(*entry)
            print(line)
    
    if True:
        # block-wise statistics
        septa = [172.5, 187.9, 200.0, 217.5, 240.0, 255.0, 265.5, 275.0, 279.5, 285.0]  # s
        septa = np.array(septa)
        septa *= 1e9  # convert s to ns
        groups = [[] for i in septa]
        csrmv_v3_kernel_collection: List[kernelLaunchEntry] = statDict[csrmv_v3_kernel_shortName]
        for entry in csrmv_v3_kernel_collection:
            timeStart = entry.start
            groupIdx = None
            for k in range(len(septa)):
                if septa[k] > timeStart:
                    groupIdx = k
                    break
            assert groupIdx is not None
            groups[groupIdx].append(entry)

        nMatrices = [402, 177, 117, 94, 68, 43, 27, 23, 22, 20]
        assert len(nMatrices) == len(septa)
        for i in range(len(groups)):
            localGroup = groups[i]
            A_and_ATrans: List[kernelLaunchEntry] = [localGroup[j] for j in range(len(localGroup)) if j%2==0]
            D_and_DTrans: List[kernelLaunchEntry] = [localGroup[j] for j in range(len(localGroup)) if j%2==1]
            if False:
                # To test if A_and_ATrans and D_and_DTrans are well separated
                A_and_ATrans_Avg = calcAvg(A_and_ATrans)
                D_and_DTrans_Avg = calcAvg(D_and_DTrans)
                print(A_and_ATrans_Avg, D_and_DTrans_Avg)
            
            if True:
                # To test if A and ATrans are well separated
                A_List = [A_and_ATrans[i] for i in range(len(A_and_ATrans)) if i % 3 in [0, 2]]
                ATrans_List = [A_and_ATrans[i] for i in range(len(A_and_ATrans)) if i % 3 == 1]
                A_List_Avg = calcAvg(A_List) * 1e-6
                ATrans_List_Avg = calcAvg(ATrans_List) * 1e-6
                line = "| {} | {:.4f} | {:.4f} |".format(nMatrices[i], A_List_Avg, ATrans_List_Avg)
                print(line)

            if False:
                A_and_ATrans_Avg = calcAvg(A_and_ATrans) * 1e-6  # ns to ms
                time_per_mat = A_and_ATrans_Avg / nMatrices[i]
                line = "| {} | {:.4f} | {:.4f} |".format(nMatrices[i], A_and_ATrans_Avg, time_per_mat)
                print(line)
            


def calcAvg(inputList: List[kernelLaunchEntry]):

    timeTotal = 0
    for entry in inputList:
        timeTotal += entry.end - entry.start
    timeTotal /= len(inputList)
    return timeTotal


if __name__ == "__main__":
    # readKernel()
    # show_hdf5_structure()
    # analyzeProfilingRyan()
    # analyzeProfilingFastDose()
    # ncuResultAnalysis()
    analyzeProfilingOptimize()