import os
import numpy as np
import matplotlib.pyplot as plt

# SciPy modules for statistical analysis
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import linregress

import re

# This script lists the files in a folder.
def list_files(folder_path):
    try:
        return os.listdir(folder_path)
    except FileNotFoundError:
        print("The folder does not exist.")
        return []
    except PermissionError:
        print("Permission denied to access the folder.")
        return []

def read_datasets(folder_path, datasets):
    data_all = {}

    for file_name in datasets:
        # Try to open and read the file
        try:
            # Read and print the file content
            data = np.loadtxt(folder_path + "/" + file_name, skiprows=4)  # skiprows to skip the header line if present
            print(f"{file_name}")
            key = file_name.split('.')[0]
            data_all[key] = data
        except Exception as e:
            print(f"Failed to read {file_name}: {e}")

    return data_all

# Represents a linear function f(x) = a*x + b.
def f(x, a, b):
    return a*x + b

def calculate_bandgap(x, y, method_name, neighbor_size, image_folder, plot=False):

    peaks, _ = find_peaks(y, height=0)

    if len(peaks) > 1:
        # Pick the index erlated to highest values for the peaks.
        maxindex = peaks[np.argmax(y[peaks])]
    else:
        maxindex = peaks[0]

    #print("Peaks: ",peaks)
    #print("maxindex",maxindex)

    # Index of steepest change.
    i = np.argmax(np.gradient(y[neighbor_size:maxindex],x[neighbor_size:maxindex]))

    best_r2 = 0  # Initialize the best R² value as 0
    best_region = None  # Variable to store the best linear region

    # Iterate over the range
    for i in (range(i-neighbor_size, maxindex-neighbor_size)):
        slope, intercept, r_value, _, _ = linregress(x[i-neighbor_size:i+neighbor_size], y[i-neighbor_size:i+neighbor_size])

        if r_value**2 > best_r2:  # Eğer R² değeri daha iyi ise güncelle
            best_r2 = r_value**2
            best_region = i
 

    #best_region = i 

    slope, intercept, r_value, _, _ = linregress(x[best_region-neighbor_size:best_region+neighbor_size+1], y[best_region-neighbor_size:best_region+neighbor_size+1])
    E_bandgap = -intercept/slope

    x_linear = x[best_region - neighbor_size: best_region + neighbor_size + 1]
    y_linear = y[best_region - neighbor_size: best_region+ neighbor_size + 1]

    #a, b, r_value, p_value, stderr = linregress(x_linear, y_linear)
    #E_bandgap = -b/a

    visualization_x = np.linspace(E_bandgap-0.2, x[best_region+neighbor_size]+0.2, 2)

    if plot:
        plt.figure(figsize=(6, 4))
        plt.rcParams['lines.linewidth'] = 3  # Default line width for all plots

        plt.plot(x, y, label = "Data")
        plt.plot(x_linear, y_linear, 'o', markersize=8, color='green', label = "Selected points for\nlinear regression") 
        plt.plot(visualization_x, f(visualization_x, slope, intercept), '--', color='red')
        plt.scatter(E_bandgap, 0, marker='x', color='k', label="Bandgap = " + str(round(E_bandgap,3)))
        
        # Get current tick locations and generate new labels divided by 10^4
        yticks = plt.gca().get_yticks()
        new_labels = ['{:.1f}'.format(y/10000) for y in yticks]
        # Set fixed tick locations and new labels
        plt.gca().set_yticks(yticks)
        plt.gca().set_yticklabels(new_labels)

        plt.xlabel("Energy (eV)", fontsize=14)
        plt.ylabel('Absorbance ($\\times 10^4$)', fontsize=14)
        plt.title(re.sub(r'60', r'$_{60}$', "Tauc plot for dataset " + method_name))
        plt.legend(loc='upper right',fontsize=11)
        plt.grid()

        plt.savefig(image_folder+"/"+method_name+'_bandgap.png', bbox_inches="tight", dpi=300)
        plt.show()

    return E_bandgap, best_region

def prepare_train_test_dataset(X_full, y_full, step_length, plot=False):
    """
    Splits data into training and testing datasets based on a step length, 
    converts energy units (nm to eV), sorts data, reshapes arrays for compatibility, 
    and optionally plots the data.

    Parameters:
        X_full (array-like): Full array of x values (e.g., energy in nm).
        y_full (array-like): Full array of y values (e.g., epsilon).
        step_length (int): Step size for selecting training data points.
        plot (bool): Whether to plot the graphs of training and testing data points. Default is False.

    Returns:
        X_train (numpy.ndarray): Training x values reshaped to (-1, 1).
        y_train (numpy.ndarray): Training y values reshaped to (-1, 1).
        X_test (numpy.ndarray): Testing x values reshaped to (-1, 1).
        y_test (numpy.ndarray): Testing y values reshaped to (-1, 1).
    """

    # Validate inputs
    if not isinstance(step_length, int) or step_length <= 0:
        raise ValueError("step_length must be a positive integer.")
    if len(X_full) != len(y_full):
        raise ValueError("x_full and y_full must have the same length.")
    if len(X_full) == 0:
        raise ValueError("x_full and y_full cannot be empty.")

    # Ensure x_full and y_full are numpy arrays
    X_full = np.array(X_full)
    y_full = np.array(y_full)

    # Select indices for training data at intervals defined by step_length
    train_data_indices = list(range(0, len(X_full), step_length))

    # Split the data into training and testing sets
    X_train = X_full[train_data_indices]
    y_train = y_full[train_data_indices]

    # Use full simulation data (real function) to test the performance of the approach.
    X_test = X_full
    y_test = y_full

    # Optionally plot the training and testing data in nm
    if plot:      
        plt.figure(figsize=(6, 4))
        plt.rcParams['lines.linewidth'] = 3  # Default line width for all plots
        
        plt.plot(X_test, y_test, 'b.', label='Full simulation data')
        plt.plot(X_train, y_train, 'ro', label='Training data points')

        # Get current tick locations and generate new labels divided by 10^4
        yticks = plt.gca().get_yticks()
        new_labels = ['{:.1f}'.format(y/10000) for y in yticks]
        # Set fixed tick locations and new labels
        plt.gca().set_yticks(yticks)
        plt.gca().set_yticklabels(new_labels)

        plt.xlabel('Energy (nm)', fontsize=14)
        plt.ylabel('Absorbance ($\\times 10^4$)', fontsize=14)

        plt.legend(fontsize=11)
        plt.show()

    # Convert energy units from nm (nanometers) to eV (electron volts)
    X_train = 1240 / X_train
    X_test = 1240 / X_test

    # Reverse arrays using sorting.
    # Sort training data by x values and rearrange corresponding y values
    x_train_sorted_indices = X_train.argsort()
    X_train = X_train[x_train_sorted_indices]
    y_train = y_train[x_train_sorted_indices]

    # Reshape training data for compatibility with ML models or plotting
    X_train = X_train.reshape(-1,1)
    y_train = y_train.reshape(-1,1)

    # Reverse arrays using sorting.
    # Sort testing data by x values and rearrange corresponding y values
    x_test_sorted_indices = X_test.argsort()
    y_test = y_test[x_test_sorted_indices]
    X_test = X_test[x_test_sorted_indices]

    # Reshape testing data for compatibility with ML models or plotting
    X_test = X_test.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    # Optionally plot the training and testing data in eV
    if plot:
        plt.figure(figsize=(6, 4))
        plt.rcParams['lines.linewidth'] = 3  # Default line width for all plots
        
        plt.plot(X_test, y_test, 'b.',  label='Full simulation data')
        plt.plot(X_train, y_train, 'ro', label='Training data points')

        # Get current tick locations and generate new labels divided by 10^4
        yticks = plt.gca().get_yticks()
        new_labels = ['{:.1f}'.format(y/10000) for y in yticks]
        # Set fixed tick locations and new labels
        plt.gca().set_yticks(yticks)
        plt.gca().set_yticklabels(new_labels)

        plt.xlabel('Energy (eV)', fontsize=14)
        plt.ylabel('Absorbance ($\\times 10^4$)', fontsize=14)

        plt.legend(fontsize=11)
        plt.show()

    return X_train, y_train, X_test, y_test
