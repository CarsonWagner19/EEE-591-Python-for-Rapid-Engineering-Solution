# -*- coding: utf-8 -*-
#############################################################
# Title: Homework 2
# Author: Carson Wagner
# Date: 9/4/2025
#############################################################

import numpy as np
from scipy.integrate import quad

############################################################################
# Problem 1: Estimating the accuracy of quad Package for Finite Integrals
############################################################################


# Test Values for now, CHANGE TO USER INPUT (should be between -10 and 10)
#user_input = input("Input a set of 3 numbers between -10 and 10: ")

# Split each value by Spaces and Assign them to Variables
#a, b, c = map(int, user_input.split())


while True:
    user_input = input("Input a set of 3 numbers between -10 and 10: ")
    
    try:
        # Split each value by Spaces
        input_list = user_input.split()
        
        if len(input_list) != 3:
            print("Error: You must enter exactly 3 numbers. Please Try Again")
            continue
        
        # Assign the 3 numbers to Variables
        a, b, c = map(int, input_list)
        
        if not(-10 <= a <= 10 and -10 <= b <= 10 and -10 <= c <= 10):
            print("Error: All Values must be between -10 and 10. Please Try Again")
            continue
        
        break # if values are in range, exit the loop
    
    except ValueError:
        print("Error: Invalid Input. All values must be a number. Please Try Again")
        
        
        


# Setting Lower and Upper Limit of Integral
lower_limit = 0
upper_limit = 5

# Create Function, I(r)
def my_func(x, a, b, c):
    y = a * (x ** 2) + b * x + c
    return y

####################################
# Method 1: Quad Integration
####################################
result, err = quad(my_func, lower_limit, upper_limit, args = (a,b,c)) # Integrate Function

#Format result to display up to 4 decimal values
result_quad = round(result, 4)

# Print Method 1 Result after Formatting
print("Method 1: I1 =", result_quad)

####################################
# Method 2: Numerical Integration
####################################

# Setting value of n to calculate step size
n = 5000

# Calculate Total Area using Trapezoid Rule
def trapezoid_rule(f, a, b, c):
    
    # Calculate the width of the trapezoid
    h = (upper_limit - lower_limit) / n
    
    # Calculate the values of I(0) and I(n) and add them together
    total_area = (my_func(lower_limit, a, b, c) + my_func(upper_limit, a, b, c))  # y(0) and y(n) part of Trapezoidal Rule
    
    # Calculate sum for y(1) to y(n-1)
    for i in range (1, n):
        x = lower_limit + i * h             # Calculate x value based on change of h
        
        mid_sum = 2 * my_func(x, a, b, c)    # Calculate value of function at value of x and mulitply by 2
        
        total_area += mid_sum               # Add value found at x to total area value
        
    total_area = (h / 2.0) * total_area       # Sum of all f(x) values multiplied by h/2
    
    return total_area
    
    
# Create Function, I(r)
def my_func(x, a, b, c):
    y = a * (x ** 2) + b * x + c
    return y

# Call Trapezoid Rule function with User_Inputs
result_trapezoid = trapezoid_rule(my_func, a, b, c)

#Format result to display up to 4 decimal values
result_format = round(result_trapezoid, 4)

# Print Result of Trapezoid Rule Calculation
print("Method 2: I1 =", result_format)


# Calculate Percentage Error (Accurate is Method 1, Inaccurate is Method 2)
percent_error = abs(((result - result_trapezoid) / result)) * 100

percent_error_format = round(percent_error,4)

# Pring Percent Error
print(f"Percentage Error = {percent_error_format}% \n")


############################################################################
# Problem 2: Estimating the Accuracy of the Substitution Method and Numerical Method
#            for Infinite Integrals
############################################################################

####################################
# Method 1: Substitution Method, substituted using x = z / (1 - z)
####################################

# Create Function after substituion is done
def my_func(z):
  y = 1 / (np.sqrt(z) * np.sqrt(1 - z))
  return y

# Integrate Substituted Function in range of 0 to 1
integrate, err = quad(my_func, 0, 1)

# Format result to display up to 8 decimal points
integrate_format = round(integrate, 8)

# Print Formatted Result
print(integrate_format, "          # I2 Method 1")

####################################
# Method 2: Numerical Integration
####################################

# Define the Original Function Given
def integrand(z):
    y = 1 / (np.sqrt(z) * np.sqrt(1 - z))
    return y

# Declaring number of steps and bounds
n = 500000
lower_limit = 0.0000001  # Solves singularity at z = 0
upper_limit = 0.9999999  # Solves singularity at z = 1

def trapezoid_rule2(x):
    
    # Calculating Step Size and f(0) and f(n) in trapezoidal rule equation
    h = (upper_limit - lower_limit) / n
    total_area = integrand(lower_limit) + integrand(upper_limit)

    for i in range(1, n):
      x = lower_limit + i * h             # Calculate x value based on change of h
      
      mid_sum = 2 * integrand(x)          # Calculate value of function at value of x and mulitply by 2
      
      total_area += mid_sum               # Add value found at x to total area value
      
    total_area = (h / 2.0) * total_area   # Sum of all f(x) values multiplied by h/2
  
    return total_area
   
# Calling Trapezoid Rule Method and Formatting to 8 Decimal Places
numerical_calc = trapezoid_rule2(integrand)
numerical_format = round(numerical_calc, 8)

# Print Formatted Result
print(numerical_format,"          # I2 Method 2")


# Calculate Differences between methods 
difference_method_1 = abs(integrate_format - np.pi)       # Method 1 Difference

difference_method_2 = abs(numerical_format - np.pi)       # Method 2 Difference

# Format to not be beyond 16 decimal places
#difference_format_1 = round(difference_method_1, 16) # Method 1
#difference_format_2 = round(difference_method_2, 16) # Method 2

# Print Differences
print(f"{difference_method_1:.16f}", "  # Difference Method 1")   # Method 1
print(f"{difference_method_2:.16f}", "  # Difference Method 2")   # Method 2

    
    
    
    
    








