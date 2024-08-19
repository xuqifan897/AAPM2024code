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
    for idx, line in enumerate(KeyResults):
        line = line.split("|")
        ourTime = line[2]
        baselineTime = line[3]
        ourTime = minutes2Seconds(ourTime)
        baselineTime = minutes2Seconds(baselineTime)
        speedup[idx] = baselineTime / ourTime
    print(speedup, np.mean(speedup))
    

def minutes2Seconds(input):
    input = input.split(":")
    minutes = input[0]
    seconds = input[1]
    result = 60 * eval(minutes) + eval(seconds)
    return result

def optimizeSpeedup():
    KeyResults = os.path.join(currentDir, "KeyResults.md")
    lineStart = 51
    nPatients = 5
    with open(KeyResults, "r") as f:
        KeyResults = f.readlines()
    KeyResults = KeyResults[lineStart: lineStart + nPatients]
    speedup = np.zeros(nPatients)
    for idx, line in enumerate(KeyResults):
        line = line.split("|")
        ourTime = eval(line[2])
        baselineTime = eval(line[3])
        speedup[idx] = baselineTime / ourTime
    print(speedup, np.mean(speedup))


if __name__ == "__main__":
    currentDirInit()
    # doseCalcSpeedup()
    optimizeSpeedup()