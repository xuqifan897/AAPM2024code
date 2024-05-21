## Dose calculation and Optimization
For all scripts in this program, please modify the variables `patientName` (`HGJ_001` or `HN_002`) and `groupName` (`benchmark` or `our_model`) before running

1. PreProcess
    * Preview

        In the python file `preprocess.py`, the function `dataView()` is to visualize the input `*.nii` files and to draw the anatomy annotations above the CT slices.
    * Conversion

        After confirming that the input data is correct, the function `writeArray()` converts the `*.nii` files into linear binary arrays.

        The script `PreProcess.sh` then converts the binary matrices into structured data
    
    * Metadata generation

        Back in the python file `preprocess.py`, the functions `structureGen()` and `StructureInfoGen()` generate `structures*.json` and `StructureInfo*.csv`, respectively

2. To run the dose calculation
    * Run the script `dosecalc.sh`. This script runs the dose calculation in 4 iterations, each processing a subset of the beamlist. The results are stored in the folder specified by the argument `${outputFolder}`

    * Then back in the python file `preprocess.py`, the function `doseMatMerge()` coalesces individual dose matrix components into a single dose matrix, which is stored in the folder specified by the variable `${targetFolder}`

3. Run BOO
    * Run the script `imrtOpt.sh`. Please note that structure weights are specified in the files `StructureInfo*.csv`

4. Evaluation
    * Draw DVH by calling the function `drawDose_opt()` (for one patient and one group) or `patient_HGJ_001_DVH_comp()` (for comparison between two groups)