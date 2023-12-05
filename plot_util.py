
import S4 
import numpy as np
import matplotlib.pyplot as plt 
import csv 
import os.path
from scipy.interpolate import CubicSpline
from input_data import *
import seaborn as sns
import pandas as pd
from util import * 
from MeDiANN_02.data_generator import *
from input_data import dim_plot

def color_sets(j):
    #Defining the colors to get the image as close as possible to the paper 
    if (j%2) == 0:
        c = ['tomato','deepskyblue']
    else:
        c = ['orchid','palegreen']
    return c 


def build_plot_test_data():
    #upper left
    test_data_0 = [0.91,0.31,0.2366,0,'s','a']
    test_data_1 = [0.91,0.31,0.2366,0,'s','c']
    test_data_2 = [0.89,0.31,0.2403,0,'s','a']
    test_data_3 = [0.89,0.31,0.2403,0,'s','c']
    #upper right
    test_data_4 = [1.00,0.60,0.1450,0,'p','a']
    test_data_5 = [1.00,0.60,0.1450,0,'p','c']
    test_data_6 = [1.09,0.61,0.1417,0,'p','a']
    test_data_7 = [1.09,0.61,0.1417,0,'p','c']
    #lower left
    test_data_8 = [0.81, 0.15, 0.1863, 45, 's','a']
    test_data_9 = [0.81, 0.15, 0.1863, 45, 's','c']
    test_data_10 = [0.72, 0.35, 0.1296, 45, 's','a']
    test_data_11 = [0.72, 0.60, 0.1296, 45, 's','c']
    #lower right
    test_data_12 = [0.41, 0.30, 0.1640, 45, 'p','a']
    test_data_13 = [0.41, 0.30, 0.1640, 45, 'p','c']
    test_data_14 = [0.40, 0.31, 0.1600, 45, 'p','a']
    test_data_15 = [0.40, 0.31, 0.1600, 45, 'p','c']
    test_data = [
        test_data_0,test_data_2,test_data_4,test_data_6,
        test_data_1,test_data_3,test_data_5,test_data_7,
        test_data_8,test_data_10,test_data_12,test_data_14,
        test_data_9,test_data_11,test_data_13,test_data_15,
    ]
    return test_data


def plot_test_case():
    """This function is used to reproduce the same plots of the paper. 
    """
    test_data = build_plot_test_data()
    plt.figure(figsize=(20,20)) #Dimension of the image 
    for i in range(1,dim_plot[0]*dim_plot[1]+1): #Dim plot is the dimensions of the plot (4x4)
        colors = color_sets(i) #Pickin the color set 
        plt.subplot(dim_plot[0],dim_plot[1],i) #Building the subplot structure
        param = test_data[i-1] #Calling the parameter list
        if i not in [5,6,7,8,13,14,15,16]: #Writing the title list (avoiding redundancy)
            plt.title('{%.4f,%.2f,%.2f}'%(param[2],param[1],param[0]),fontweight='bold')
        res = calculate_power_loop(param) #Computing R and T for those parameter space 
        xs = np.linspace(1,3,numPoints) #Building the x axis 
        if i in [1,2]: #Writing R and T in the label and plotting wavelength vs R/T
            plt.plot(xs,res[0],color=colors[0],label='R')
            plt.plot(xs,res[1],color=colors[1],label='T')
        plt.plot(xs,res[0],color=colors[0]) #Plotting wavelength vs R
        plt.plot(xs,res[1],color=colors[1]) #Plotting wavelength vs T
        plt.ylim(0,1) #Setting limits for the y axis 
        plt.annotate(param[-1]+'GST',xy=(1,0.90),xytext=(1,0.90),size=10,weight='bold') #Adding the title (a or c)
        if i in [13,14,15,16]: #Writing the x label (avoiding redundancy)
            plt.xlabel(r'$\lambda (\mu m)$')
        if i in [1,5,9,13]: #Writing the y label (avoiding redundancy)
            plt.ylabel('Normed power flux')
    plt.subplots_adjust(hspace=0.3) #Plotting the subplot
    plt.figlegend(loc='lower center', ncol=2,fontsize=16)  # Adjust the legend location and columns as needed
    plt.text(0.35, 0.95, 's-polarization', ha='center', va='top', fontsize=14, transform=plt.gcf().transFigure,weight='bold') #Annotating text for s polarization
    plt.text(0.35+0.35, 0.95, 'p-polarization', ha='center', va='top', fontsize=14, transform=plt.gcf().transFigure,weight='bold') #Annotating text for p polarization
    plt.text(0.08,0.75, r'$\theta=0°$', ha='center', va='top', fontsize=14, transform=plt.gcf().transFigure,weight='bold',rotation=90) #Annotating theta = 0
    plt.text(0.08,0.35, r'$\theta=45°$', ha='center', va='top', fontsize=14, transform=plt.gcf().transFigure,weight='bold',rotation=90) #Annotating theta = 45 
    plt.savefig(OUTPUT_IMAGE_PATH+'Paper_replicate.svg') #Saving the image


def heatmap_builder(param_list,reflection_matrix,transmission_matrix,class_p):
    """This function builds the heatmaps to plot X vs Lambda. The color is the Reflectance at mid frequency
    """
    X_plot = param_list[:,2].astype(float) #Defining the X of our plot
    Y_plot = param_list[:,0].astype(float) #Defining the Y of our plot
    hue_plots = [reflection_matrix[:,middle_point],transmission_matrix[:,middle_point]] #Reflection and transmission at middle point
    k=1 #Iteration starts
    names = [r'Reflection $\lambda = 2 \mu m$ value',r'Transmission $\lambda = 2 \mu m$ value'] #Name of the plot
    plt.figure(figsize=(10,5)) #Building image
    for hue_plot in hue_plots: #Running on reflection and transmission
        plt.subplot(1,2,k)
        scatter = plt.scatter(X_plot, Y_plot, c=hue_plot, cmap='hot', s=10) #scatterplot (heatmap)
        plt.title(names[k-1]) #Writing the title
        plt.xlabel(r'$X(\mu m)$') #X label name 
        plt.ylabel(r'$\Lambda(\mu m$)') #Y label name 
        cbar = plt.colorbar(scatter) #Showing the color bar
        k=k+1
    plt.savefig(OUTPUT_IMAGE_PATH+class_p+'_heatmap.svg') #Saving the image