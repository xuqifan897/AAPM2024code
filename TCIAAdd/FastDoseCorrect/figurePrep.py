import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import h5py
from skimage import measure

colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.XKCD_COLORS.values())
RootFolder = "/data/qifan/FastDoseWorkplace/TCIAAdd"
patientList = ["002", "003", "009", "013", "070", "125", "132", "190"]

def drawDVH():
    pass