import numpy as np
import os
import matplotlib.pyplot as plt

def cfdPlotRes(path,equations):
    # Read data from file
    data = np.loadtxt(path+os.sep+'convergence'+os.sep+'convergenceUp.out', skiprows=1)
    
    # Return if no data yet is stored
    if data.size == 0:
        return
    
    # Extract data columns
    iters = data[:, 0]
    currentTime = data[:, 1]
    UxResInit = data[:, 2]
    UyResInit = data[:, 3]
    UzResInit = data[:, 4]
    pResInit = data[:, 5]
    TResInit = data[:, 9]
    
    # Clear previous plot
    plt.clf()
    
    # Plot residuals for each equation
    legendStringArray = []
    # theEquationNames = model.equations
    
    for iEquation in equations:
        if iEquation.name == 'U':
            legendStringArray.append('Ux')
            legendStringArray.append('Uy')
            legendStringArray.append('Uz')
            
            plt.semilogy(iters, UxResInit, '-xr')
            plt.hold(True)
            plt.semilogy(iters, UyResInit, '-og')
            plt.hold(True)
            plt.semilogy(iters, UzResInit, '-+b')
            plt.hold(True)
        elif iEquation.name == 'p':
            legendStringArray.append('p-mass')
            plt.semilogy(iters, pResInit, '-<k')
        elif iEquation.name == 'T':
            legendStringArray.append('T')
            plt.semilogy(iters, TResInit, '->c')
    
    # Set plot labels and legend
    plt.xlabel('Global Iterations')
    plt.ylabel('Scaled RMS Residuals')
    plt.legend(legendStringArray)
    
    # Set plot grid and axis limits
    plt.grid(True)
    plt.axis('tight')
    plt.ylim(1e-6, 1e2)
    
    # Pause to show the plot
    plt.pause(0.01)