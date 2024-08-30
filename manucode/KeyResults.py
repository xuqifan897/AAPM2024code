import os
import numpy as np

currentDir = None

def currentDirInit():
    current_file_path = os.path.abspath(__file__)
    global currentDir
    currentDir = os.path.dirname(current_file_path)
    print(currentDir)

def doseCalcSpeedup():
    KeyResults = os.path.join(currentDir, "KeyResults.md")
    lineStart = 41
    nPatients = 5
    with open(KeyResults, "r") as f:
        KeyResults = f.readlines()
    KeyResults = KeyResults[lineStart: lineStart + nPatients]
    speedup = np.zeros(nPatients)
    ourTimeList = np.zeros(nPatients)
    baselineTimeList = np.zeros(nPatients)
    for idx, line in enumerate(KeyResults):
        line = line.split("|")
        ourTime = line[2]
        baselineTime = line[3]
        ourTime = minutes2Seconds(ourTime)
        baselineTime = minutes2Seconds(baselineTime)
        speedup[idx] = baselineTime / ourTime
        ourTimeList[idx] = ourTime
        baselineTimeList[idx] = baselineTime
    ourTimeAvg = np.mean(ourTimeList)
    baselineTimeAvg = np.mean(baselineTimeList)
    speedupAvg = np.mean(speedup)
    print(speedup)
    print(seconds2Minutes(ourTimeAvg),
        seconds2Minutes(baselineTimeAvg), speedupAvg)
    

def minutes2Seconds(input):
    input = input.split(":")
    minutes = input[0]
    seconds = input[1]
    result = 60 * eval(minutes) + eval(seconds)
    return result

def seconds2Minutes(input):
    minutes = int(input // 60)
    seconds = input % 60
    result = "{}:{:.3f}".format(minutes, seconds)
    return result

def optimizeSpeedup():
    KeyResults = os.path.join(currentDir, "KeyResults.md")
    lineStart = 51
    nPatients = 5
    with open(KeyResults, "r") as f:
        KeyResults = f.readlines()
    KeyResults = KeyResults[lineStart: lineStart + nPatients]
    speedup = np.zeros(nPatients)
    ourTimeList = np.zeros(nPatients)
    baselineTimeList = np.zeros(nPatients)
    for idx, line in enumerate(KeyResults):
        line = line.split("|")
        ourTime = eval(line[2])
        baselineTime = eval(line[3])
        speedup[idx] = baselineTime / ourTime
        ourTimeList[idx] = ourTime
        baselineTimeList[idx] = baselineTime
    ourTimeAvg = np.mean(ourTimeList)
    baselineTimeAvg = np.mean(baselineTimeList)
    speedupAvg = np.mean(speedup)
    print(speedup)
    print(ourTimeAvg, baselineTimeAvg, speedupAvg)


def optimizeSpeedupMinutes():
    KeyResults = os.path.join(currentDir, "KeyResults.md")
    lineStart = 51
    nPatients = 5
    with open(KeyResults, "r") as f:
        KeyResults = f.readlines()
    KeyResults = KeyResults[lineStart: lineStart + nPatients]
    ourTimeArray = np.zeros(nPatients)
    baselineArray = np.zeros(nPatients)
    speedupArray = np.zeros(nPatients)
    for i in range(nPatients):
        line = KeyResults[i]
        line = line.split("|")
        ourTime = eval(line[2])
        baseline = eval(line[3])
        speedup = eval(line[4])
        ourTimeArray[i] = ourTime
        baselineArray[i] = baseline
        speedupArray[i] = speedup
    ourTimeAvg = np.mean(ourTimeArray)
    baselineAvg = np.mean(baselineArray)
    speedupAvg = np.mean(speedupArray)

    # convert seconds to minutes
    content = "| Case \ Group | Ours | Baseline | Speedup |\n|-|-|-|-|"
    for i in range(nPatients):
        name = "Patient{:03d}".format(i+1)
        ourTime = seconds2Minutes(ourTimeArray[i])
        baseline = seconds2Minutes(baselineArray[i])
        line = "| {} | {} | {} | {} |".format(
            name, ourTime, baseline, speedupArray[i])
        content = content + "\n" + line
    lastLine = "| {} | {} | {} | {} |".format("Average",
        seconds2Minutes(ourTimeAvg), seconds2Minutes(baselineAvg), speedupAvg)
    content = content + "\n" + lastLine
    print(content)


if __name__ == "__main__":
    currentDirInit()
    # doseCalcSpeedup()
    # optimizeSpeedup()
    optimizeSpeedupMinutes()