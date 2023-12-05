#==============================================================
# Script for calculating T and R of a 1d grating where the 
# substrate is Al2O3 and the grating is created by one dimensional
# groves of GST 
#==============================================================
# The primary purpose was to recreate the result of the article
# 'Artificial neural network discovery of a switchable metasurface 
# reflector.' 
# Optics Express, 28(17), 24629-24656
# (Mainly Fig. 6a and so on)
#==============================================================
#==============================================================
# copyright @ Apex Mds.
# Created by Piero Paialunga on June 16 2023
# last modified on : July 10 2023 by Piero
#==============================================================
#The base unit is kept in micrometer
# about unit :
#S4 solves the linear Maxwell’s equations, which are scale-invariant.
# Therefore, S4 uses normalized units, so the conversion between 
#the numbers in S4 and physically relevant numbers can sometimes 
#be confusing. Here we show how to perform these conversions and 
#provide some examples.
#==============================================================
#==============================================================
#The speed of light and the vacuum permittivity and vacuum permeability 
#are all normalized to unity in S4. Because of this, time is measured 
#in units of length and frequency is measured in units of inverse length. 
#Due to the scale invariant nature of the linear Maxwell’s equations, 
#there is no intrinsic length unit. Instead, one can imagine that all 
#lengths specified in S4 are multiples of a common reference length unit.
#When the lattice vectors are set, this determines a length scale. 
#Suppose we have a square lattice with a periodicity of 680nm. 
#It might be logical then to choose 1 micron as the base length unit, 
#and to specify all lengths in microns. The lattice would then be set to 
#the vectors (0.68, 0) and (0, 0.68). Since frequency is in units of 
#inverse length, then SetFrequency(1) corresponds to a wavelength of 
#1um or a physical frequency of (c/1um) = 300 THz, and SetFrequency(1.1) 
#corresponds to a wavelength of (1/1.1) = 909nm, etc.
#==============================================================
#https://web.stanford.edu/group/fan/S4/units.html
#==============================================================
import numpy as np
#from util import * 
#from s4_util import * 
#from plot_util import * 

#==============================================================


num_X,num_H,num_L = 100,100,100
n_al2o3= 1.725 #Material Parameter
dir_path = '/home/apexmds/Desktop/pcm_phc_tr/Version 0.5.1/Data'
output_path = '/home/apexmds/Desktop/pcm_phc_tr/Version 0.5.1/Data/output'
al2o3_file= '/materials/al2o3.csv'
a_gst_file = '/materials/a_gst.csv'
c_gst_file = '/materials/c_gst.csv'
IncludeLoss= True # if set True is includes loss parameter k
dim = 1 #Dimensions
nbasis_int = 15 #Number of basis for Fourier Transform
lambdaMin= 1.0 # Minimum wavelength
lambdaMax= 3.0 # Maximum wavelength
numPoints = 101 # Number of wavelengths
wavelength_list = np.linspace(lambdaMin,lambdaMax, numPoints) #List of wavelengths
show_TRplot = True # if TRUE shows T, R vs Wavelength
show_TplusR= False # if show_TRplot = TRUE and if show_TplusR = TRUE shows T+R on the same plots
show_material_interpolation_plot = False #Plotting parameter
#Checking for any file_not_found error
a_gst = np.loadtxt(dir_path+a_gst_file, delimiter=",") #Extracting a state properties
c_gst = np.loadtxt(dir_path+c_gst_file,delimiter=',') #Extracting c state properties
#Xmin,Xmax,LambdaMin,LambdaMax
states = ['a','c'] #Number of possible states
Xmin = 0.005 #Minimum X
Xmax = 0.5 #Maximum X
Hmin = 0.05 #Minimum Height
Hmax = 1 #Maximum Height
Lambdamin = 0.05 #Minimum Lambda
Lambdamax = 2 #Maximum Lambda
thetas = [0,45] #theta parameter
polarizations = ['s','p'] #polarization parameters
N=20 
polarization_dict = {'s':[1.0 + 0.0j,0.0 + 0.0j],'p':[0.0 + 0.0j,1.0 + 0.0j]} #Converting them into complex numbers
#Middle point used to plot
middle_point = int(numPoints/2)
#Dimensions for the subplot
dim_plot = (4,4)
#Output image path 
OUTPUT_IMAGE_PATH = '/home/apexmds/Desktop/pcm_phc_tr/Version 0.5.1/Data/output/'

