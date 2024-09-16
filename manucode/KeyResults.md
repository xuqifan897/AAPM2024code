## Slab dose accuracy
2mm/2% passing rate:
| width | passing rate |
|-|-|
| 0.5cm | 98.83% |
| 1.0cm | 98.83% |
| 2.0cm | 98.83% |
## Water dose accuracy
2mm/2% passing rate:
| with | passing rate |
|-|-|
| 0.5cm | 99.61% |
| 1.0cm | 100.00% |
| 2.0cm | 98.83% |

## Dose calculation time
| Case \ Group | Ours | Baseline | Speedup |
|-|-|-|-|
| Patient001 | 1:10.53 | 28:2.56 | 23.86 |
| Patient002 | 1:17.70 | 15:32.63 | 12.00 |
| Patient003 | 1:15.25 | 20:17.60 | 16.18 |
| Patient004 | 1:13.70 | 18:26.65 | 15.02 |
| Patient005 | 48.78 | 13:1.24 | 16.01 |

Average speedup: 16.61

## Optimization time
| Case \ Group | Ours | Baseline | Speedup |
|-|-|-|-|
| Patient001 | 1:29.40 | 20:6.25 | 13.49 |
| Patient002 | 1:16.70 | 13:6.82 | 10.26 |
| Patient003 | 1:25.80 | 17:16.13 | 12.08 |
| Patient004 | 2:44.00 | 28:18.79 | 10.36 |
| Patient005 | 1:42.00 | 20:36.09 | 12.12 |
Average speedup: 11.66


# New results on Aug 19
## dose calculation time
| Case \ Group | Ours | Baseline | Speedup |
|-|-|-|-|
| Patient001 | 1:11.7 | 21:31.9 | 18.017 |
| Patient002 | 0:52.0 | 9:52.8 | 11.408 |
| Patient003 | 1:13.5 | 13:58.2 | 11.398 |
| Patient004 | 1:14.6 | 16:34.2 | 13.326 |
| Patient005 | 0:52.2 | 12:11.1 | 14.013 |
| Average | 1:04.8 | 14:49.6 | 13.632 |

## Optimization time
| Case \ Group | Ours | Baseline | Speedup |
|-|-|-|-|
| Patient001 | 9.36e+01 | 891.7584 | 9.527 |
| Patient002 | 7.87e+01 | 837.3996 | 10.640 |
| Patient003 | 6.28e+01 | 678.3028 | 10.801 |
| Patient004 | 1.69e+02 | 1.7982e+03 | 10.640 |
| Patient005 | 1.00e+02 | 1.1421e+03 | 11.421 |
| Average | 1:40.820 | 17:49.552 | 10.606 |

## Optimization time convert
| Case \ Group | Ours | Baseline | Speedup |
|-|-|-|-|
| Patient001 | 1:33.6 | 14:51.8 | 9.527 |
| Patient002 | 1:18.7 | 13:57.4 | 10.640 |
| Patient003 | 1:02.8 | 11:18.3 | 10.801 |
| Patient004 | 2:49.0 | 29:58.2 | 10.640 |
| Patient005 | 1:40.0 | 19:02.1 | 11.421 |
| Average | 1:40.8 | 17:49.6 | 10.606 |

## Pancreas SIB results
### R50
| Patient | Ours | Baseline | Clinical |
| - | - | - | - |
| 001 | 1.828 | 2.790 | 4.746 |
| 002 | 3.781 | 7.617 | 9.913 |
| 003 | 4.143 | 4.778 | 6.593 |
| 004 | 2.509 | 3.997 | 4.806 |
| 005 | 2.503 | 3.454 | 4.853 |

### Dose calculation
| Patient | Ours | Baseline | Speedup |
| - | - | - | - |
| 001 | 71.838 | 1352.117 | 18.822 |
| 002 | 57.993 | 825.821 | 14.240 |
| 003 | 75.191 | 998.836 | 13.284 |
| 004 | 74.643 | 1015.347 | 13.603 |
| 005 | 54.973 | 781.929 | 14.224 |
| Avg | 66.928 | 994.810 | 14.834 |

### Optimization
| Patient | Ours | Baseline | Speedup |
| - | - | - | - |
| 001 | 89.400 | 1008.496 | 11.281 |
| 002 | 76.700 | 980.211 | 12.780 |
| 003 | 85.800 | 1575.840 | 18.366 |
| 004 | 164.000 | 1583.301 | 9.654 |
| 005 | 102.000 | 1442.012 | 14.137 |
| Avg | 103.580 | 1317.972 | 13.244 |

## TCIA results
### R50
| Patient | Ours | Baseline | Clinical |
| - | - | - | - |
| 002 | 1.611 | 1.618 | 2.592 |
| 003 | 1.891 | 1.859 | 3.767 |
| 009 | 2.010 | 1.888 | 4.824 |
| 013 | 1.801 | 1.816 | 4.218 |
| 070 | 1.981 | 1.820 | 3.141 |
| 125 | 1.697 | 1.720 | 3.082 |
| 132 | 1.951 | 1.889 | 4.972 |
| 190 | 1.986 | 1.869 | 3.675 |

### Dose calculation
| Patient | Ours | Baseline | Speedup |
| - | - | - | - |
| 002 | 68.383 | 1200.426 | 17.554 |
| 003 | 59.455 | 809.928 | 13.623 |
| 009 | 70.075 | 1102.948 | 15.740 |
| 013 | 72.054 | 1071.515 | 14.871 |
| 070 | 65.940 | 922.584 | 13.991 |
| 125 | 33.248 | 435.220 | 13.090 |
| 132 | 57.133 | 634.242 | 11.101 |
| 190 | 69.177 | 962.148 | 13.908 |
| Avg | 61.933 | 892.376 | 14.235 |

### Optimization
| Patient | Ours | Baseline | Speedup |
| - | - | - | - |
| 002 | 88.300 | 910.532 | 10.312 |
| 003 | 41.500 | 358.909 | 8.648 |
| 009 | 51.700 | 528.370 | 10.220 |
| 013 | 95.700 | 733.953 | 7.669 |
| 070 | 58.700 | 640.590 | 10.913 |
| 125 | 45.000 | 599.408 | 13.320 |
| 132 | 53.500 | 524.601 | 9.806 |
| 190 | 37.000 | 489.516 | 13.230 |
| Avg | 58.925 | 598.235 | 10.515 |

### Ryan dose calculation profiling
Kernel Execution time
| Kernel | Invocations | Time Avg | Time total |
| - | - | - | - |
| beamletRayTrace | 32 | 21.412 | 685.180 |
| packRowConvolve | 32 | 9.408 | 301.065 |
| revToBev | 32 | 0.801 | 25.619 |
| unpackBevDosePillar | 385 | 0.175 | 67.366 |
Total computation time: 1011.9376 ms

Memcpy operations
* Invocations: 385
* Average time: 0.6367
* Total time: 245.12

Sparsification
* Counted: 384
* Average time: 6.5977
* Total time: 2533.5223

### FastDose dose calculation profiling
Kernel Execution time
| Kernel Name | Invocations | Time Avg | Time total |
| - | - | - | - |
| termaCompute | 10 | 0.830 | 8.302 |
| doseCompute | 10 | 61.206 | 612.065 |
| interpArrayPrep | 10 | 0.263 | 2.626 |
| superVoxelInterp | 10 | 0.146 | 1.457 |
| voxelInterp | 10 | 30.355 | 303.548 |
| cusparseCountNz | 10 | 21.921 | 219.207 |
| deviceScanInit | 10 | 0.001 | 0.014 |
| deviceScan | 10 | 0.002 | 0.022 |
| cusparseGatherNz | 10 | 25.934 | 259.340 |
Total computation time: 1588.2034

### Ryan dose calculation kernel statistics
| Kernel Name \ Metrics | Active Warps Per Scheduler | Eligible Warps Per Scheduler | One or More Eligible | No Eligible | Issued Warp Per Scheduler | Avg. Active Threads Per Warp | Avg. Not Predicated Off Threads Per Warp | Theoretical Occupancy | Achieved Occupancy | Theoretical Active Warps per SM | Achieved Active Warps Per SM | Warp Cycles Per Issued Instruction | Warp Cycles Per Executed Instruction |
|  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |
|cudaBeamletRaytrace | 6.9788 $\pm$ 0.0201 | 0.4325 $\pm$ 0.0927 | 26.8597 $\pm$ 4.3265 | 73.1403 $\pm$ 4.3265 | 0.2675 $\pm$ 0.0432 | 15.0981 $\pm$ 3.3675 | 14.4809 $\pm$ 3.2259 | 62.5000 $\pm$ 0.0000 | 58.2191 $\pm$ 0.1721 | 30.0000 $\pm$ 0.0000 | 27.9453 $\pm$ 0.0827 | 26.7775 $\pm$ 4.9686 | 26.8087 $\pm$ 4.9767|
|PackRowConvolve | 6.1644 $\pm$ 1.6122 | 1.0175 $\pm$ 0.3892 | 54.7044 $\pm$ 12.4916 | 45.2956 $\pm$ 12.4916 | 0.5469 $\pm$ 0.1252 | 12.8812 $\pm$ 1.7771 | 11.8584 $\pm$ 1.5947 | 89.0625 $\pm$ 10.9639 | 51.0613 $\pm$ 13.6604 | 42.7500 $\pm$ 5.2619 | 24.5088 $\pm$ 6.5562 | 11.1791 $\pm$ 0.4584 | 11.1866 $\pm$ 0.4592|
|PackedREVtoBEVdose | 10.2731 $\pm$ 0.0601 | 0.5425 $\pm$ 0.1189 | 34.5138 $\pm$ 5.8570 | 65.4862 $\pm$ 5.8570 | 0.3444 $\pm$ 0.0584 | 28.9572 $\pm$ 0.0057 | 28.7000 $\pm$ 0.0071 | 100.0000 $\pm$ 0.0000 | 86.1337 $\pm$ 0.3767 | 48.0000 $\pm$ 0.0000 | 41.3428 $\pm$ 0.1812 | 30.7278 $\pm$ 5.6799 | 30.7300 $\pm$ 5.6809|
|UnpackBEVDosePillar | 9.7052 $\pm$ 0.2391 | 0.8275 $\pm$ 0.0144 | 23.9896 $\pm$ 0.3903 | 76.0104 $\pm$ 0.3903 | 0.2398 $\pm$ 0.0045 | 31.9283 $\pm$ 0.0068 | 31.7574 $\pm$ 0.0068 | 100.0000 $\pm$ 0.0000 | 80.9794 $\pm$ 0.4598 | 48.0000 $\pm$ 0.0000 | 38.8698 $\pm$ 0.2205 | 40.4581 $\pm$ 0.8140 | 40.4969 $\pm$ 0.8149|

### FastDose dose calculation kernel statistics
| Kernel Name \ Metrics | Active Warps Per Scheduler | Eligible Warps Per Scheduler | One or More Eligible | No Eligible | Issued Warp Per Scheduler | Avg. Active Threads Per Warp | Avg. Not Predicated Off Threads Per Warp | Theoretical Occupancy | Achieved Occupancy | Theoretical Active Warps per SM | Achieved Active Warps Per SM | Warp Cycles Per Issued Instruction | Warp Cycles Per Executed Instruction |
|  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |  -  |
|d_TermaComputeCollective | 7.0700 $\pm$ 0.0540 | 3.2000 $\pm$ 0.0535 | 80.4030 $\pm$ 0.5454 | 19.5970 $\pm$ 0.5454 | 0.8040 $\pm$ 0.0049 | 32.0000 $\pm$ 0.0000 | 29.8300 $\pm$ 0.0000 | 100.0000 $\pm$ 0.0000 | 58.8090 $\pm$ 0.4351 | 48.0000 $\pm$ 0.0000 | 28.2280 $\pm$ 0.2076 | 8.7930 $\pm$ 0.0405 | 8.7940 $\pm$ 0.0390|
|d_DoseComputeCollective | 6.1580 $\pm$ 0.0368 | 2.0660 $\pm$ 0.0162 | 68.5670 $\pm$ 0.2188 | 31.4330 $\pm$ 0.2188 | 0.6850 $\pm$ 0.0050 | 31.5990 $\pm$ 0.0104 | 27.8090 $\pm$ 0.0104 | 66.6700 $\pm$ 0.0000 | 51.3160 $\pm$ 0.3057 | 32.0000 $\pm$ 0.0000 | 24.6310 $\pm$ 0.1464 | 8.9800 $\pm$ 0.0253 | 9.0440 $\pm$ 0.0242|
|d_InterpArrayPrep | 8.7740 $\pm$ 0.0297 | 1.8820 $\pm$ 0.1123 | 55.5420 $\pm$ 1.5417 | 44.4580 $\pm$ 1.5417 | 0.5560 $\pm$ 0.0162 | 32.0000 $\pm$ 0.0000 | 28.1840 $\pm$ 0.0459 | 100.0000 $\pm$ 0.0000 | 74.0370 $\pm$ 0.1892 | 48.0000 $\pm$ 0.0000 | 35.5390 $\pm$ 0.0909 | 15.8110 $\pm$ 0.4964 | 15.8200 $\pm$ 0.4966|
|d_superVoxelInterp | 1.4580 $\pm$ 0.0060 | 0.1700 $\pm$ 0.0000 | 14.8160 $\pm$ 0.0400 | 85.1840 $\pm$ 0.0400 | 0.1500 $\pm$ 0.0000 | 31.8290 $\pm$ 0.0083 | 30.0990 $\pm$ 0.0070 | 66.6700 $\pm$ 0.0000 | 12.1500 $\pm$ 0.0000 | 32.0000 $\pm$ 0.0000 | 5.8300 $\pm$ 0.0000 | 9.8410 $\pm$ 0.0359 | 9.8460 $\pm$ 0.0347|
|d_voxelInterp | 7.1160 $\pm$ 0.0514 | 3.5080 $\pm$ 0.0340 | 77.7000 $\pm$ 0.3032 | 22.3000 $\pm$ 0.3032 | 0.7770 $\pm$ 0.0046 | 21.2010 $\pm$ 0.1495 | 19.5880 $\pm$ 0.1343 | 66.6700 $\pm$ 0.0000 | 58.5440 $\pm$ 0.5736 | 32.0000 $\pm$ 0.0000 | 28.1000 $\pm$ 0.2755 | 9.1590 $\pm$ 0.0740 | 9.1590 $\pm$ 0.0740|
|count_nz_kernel | 4.6780 $\pm$ 0.0371 | 0.1600 $\pm$ 0.0000 | 14.8230 $\pm$ 0.1013 | 85.1770 $\pm$ 0.1013 | 0.1500 $\pm$ 0.0000 | 32.0000 $\pm$ 0.0000 | 32.0000 $\pm$ 0.0000 | 100.0000 $\pm$ 0.0000 | 38.9700 $\pm$ 0.3001 | 48.0000 $\pm$ 0.0000 | 18.7060 $\pm$ 0.1440 | 31.5530 $\pm$ 0.0347 | 31.5540 $\pm$ 0.0338|
|DeviceScanInitKernel | 0.9810 $\pm$ 0.0373 | 0.0300 $\pm$ 0.0000 | 2.7270 $\pm$ 0.1131 | 97.2730 $\pm$ 0.1131 | 0.0300 $\pm$ 0.0000 | 32.0000 $\pm$ 0.0000 | 20.3200 $\pm$ 0.0000 | 100.0000 $\pm$ 0.0000 | 8.0600 $\pm$ 0.0089 | 48.0000 $\pm$ 0.0000 | 3.8660 $\pm$ 0.0049 | 36.0240 $\pm$ 1.0817 | 52.9750 $\pm$ 1.5909|
|DeviceScanKernel | 0.9940 $\pm$ 0.0294 | 0.0790 $\pm$ 0.0030 | 8.0040 $\pm$ 0.2456 | 91.9960 $\pm$ 0.2456 | 0.0790 $\pm$ 0.0030 | 32.0000 $\pm$ 0.0000 | 29.5830 $\pm$ 0.0064 | 83.3300 $\pm$ 0.0000 | 8.2900 $\pm$ 0.0000 | 40.0000 $\pm$ 0.0000 | 3.9800 $\pm$ 0.0000 | 12.4140 $\pm$ 0.2422 | 13.0330 $\pm$ 0.2544|
|gather_nz_kernel | 4.6770 $\pm$ 0.0366 | 0.2300 $\pm$ 0.0000 | 21.0790 $\pm$ 0.1409 | 78.9210 $\pm$ 0.1409 | 0.2100 $\pm$ 0.0000 | 31.7820 $\pm$ 0.0189 | 30.7430 $\pm$ 0.0185 | 100.0000 $\pm$ 0.0000 | 38.9430 $\pm$ 0.3037 | 48.0000 $\pm$ 0.0000 | 18.6930 $\pm$ 0.1441 | 22.1820 $\pm$ 0.0252 | 22.1820 $\pm$ 0.0252|


### FastDose Optimization kernel statistics
| Kernel Name | Invocations | Time Avg | Time total |
| - | - | - | - |
| asum_kernel | 166504 | 0.003 | 540.609 |
| csrmv_v3_kernel | 56588 | 1.617 | 91510.322 |
| csrmv_v3_partition_kernel | 56588 | 0.032 | 1817.900 |
| dot_kernel | 14474 | 0.002 | 26.913 |
| reduce_1Block_kernel | 14474 | 0.002 | 25.984 |
| d_ATimesBSquare | 43422 | 0.013 | 543.074 |
| d_calcProx4Term4 | 14474 | 0.003 | 46.163 |
| d_calcSumProx4 | 14474 | 0.004 | 62.737 |
| d_calc_grad_term1_input | 7237 | 0.026 | 189.631 |
| d_calc_grad_term2_input | 7237 | 0.004 | 28.015 |
| d_calc_prox1 | 14474 | 0.002 | 34.472 |
| d_calc_prox2 | 14474 | 0.014 | 209.348 |
| d_calc_prox_tHat_g0 | 6583 | 0.074 | 484.069 |
| d_elementWiseAdd | 7237 | 0.002 | 10.873 |
| d_elementWiseGreater | 5441 | 0.001 | 6.378 |
| d_elementWiseMax | 7237 | 0.001 | 10.172 |
| d_elementWiseMul | 5441 | 0.001 | 7.726 |
| d_elementWiseScale | 6588 | 0.001 | 8.795 |
| d_elementWiseSqrt | 12024 | 0.001 | 15.050 |
| d_elementWiseSquare | 13166 | 0.001 | 18.730 |
| d_linearComb | 27612 | 0.002 | 44.320 |
| d_prox1Norm | 14474 | 0.006 | 85.582 |
| d_proxL2Onehalf_calc_tHat | 6583 | 0.001 | 9.348 |

#### Efficiency measure w.r.t. the number of beams
| # beams | Time per scrmv | Time per csrmv per beam |
| - | - | - |
| 402 | 21.1837 | 0.0527 |
| 177 | 9.7496 | 0.0551 |
| 117 | 6.4736 | 0.0553 |
| 94 | 5.1630 | 0.0549 |
| 68 | 3.6794 | 0.0541 |
| 43 | 2.3847 | 0.0555 |
| 27 | 1.5453 | 0.0572 |
| 23 | 1.3139 | 0.0571 |
| 22 | 1.2457 | 0.0566 |

#### A and ATrans efficiency measure
| # beams | Time per scrmv (A) | Time per csrmv (ATrans) |
| - | - | - |
| 402 | 21.1836 | 21.1838 |
| 177 | 9.7345 | 9.7796 |
| 117 | 6.4698 | 6.4813 |
| 94 | 5.1601 | 5.1690 |
| 68 | 3.6794 | 3.6792 |
| 43 | 2.3847 | 2.3847 |
| 27 | 1.5449 | 1.5463 |
| 23 | 1.3139 | 1.3139 |
| 22 | 1.2457 | 1.2457 |