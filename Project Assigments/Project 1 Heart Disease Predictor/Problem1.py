################################################################################
# Created on Sun Sept 25, 2025                                                 #
#                                                                              #
# @author: Carson Wagner                                                       #
#                                                                              #
# Project 1, Problem 1: Machine Learning to Predict Risk of Heart Disease      #
################################################################################

# Import Libraries
from pandas import DataFrame, read_csv       # Read CSV File and Convert to Data Frame
import pandas as pd                          # Panda Library
import numpy as np                           # Numpy Library
import seaborn as sns                        # Creating Pairplot
import matplotlib.pyplot as plt              # Create and Display Pair Plot

# Get the Correlation Matrix and and Sort the Values to Analyze
def correlation(data):
    
    # Create Correlation Matrix (Use absolute value because the large negatives are as useful as large positive)
    corr = data.corr().abs()
    print('\nCorrelation Matrix:\n', corr)

    # Clear Redundant Values by Element Multiplication and Transposing
    corr *= np.tri(*corr.values.shape, k=-1).T
    
    # Unstack the Correlation Matrix for Sorting
    corr_unstack = corr.unstack()
    
    # Sort the Values in Descending Order to get highest correlation values
    corr_unstack = corr_unstack.copy()
    corr_unstack.sort_values(inplace=True, ascending=False)
    
    # Print only the Top Values and the Target Variable, a1p1
    return corr_unstack.head(5), corr_unstack['a1p2'].head(10)
    
# Create Covariance Matrix and Analyze Data
def covariance(data):
        
    # Create Covariance Matrix (Use absolute value because the large negatives are as useful as large positive)
    cov = data.cov().abs()
    print('\nCovariance Matrix:\n', cov)
    
    # Clear Redundant Values by Element Multiplication and Transposing
    cov *= np.tri(*cov.values.shape, k=-1).T
    
    # Unstack the Covariance Matrix for Sorting
    cov_unstack = cov.unstack()

    # Sort the Values in Descending Order to get highest correlation values
    cov_unstack = cov_unstack.copy()
    cov_unstack.sort_values(inplace=True, ascending=False)
    
    # Print only the Top Values and the Target Variable, a1p1
    return cov_unstack.head(5), cov_unstack['a1p2'].head(10)

# Create Pair Plot
def pairplot(data):
    
    # Set up and Create Pair Plot
    sns.set(style='whitegrid', context='notebook')
    sns.pairplot(get_data,height=3)
    plt.show()

##########################################
#               MAIN                     #
##########################################

# Read the CSV File to gather data
get_data = pd.read_csv('heart1.csv')

# Get Correlation and Covariance
top_corr, predict_corr = correlation(get_data)
top_cov, predict_cov = covariance(get_data)

# Print the Top Correlated Variables
print('Highest Correlated Variables')
print(top_corr)

# Print Top Correlation Variables for a1p2
print('\nHighest Correlation with a1p2 variable')
print(predict_corr)

# Print the Top Covariance Variables
print('\nHighest Covariance Variables')
print(top_cov)

# Print Top Covariance Variables for a1p2
print('\nHighest Covariance with a1p2 variable')
print(predict_cov)


# Create Pair Plot and Print
pairplot(get_data)