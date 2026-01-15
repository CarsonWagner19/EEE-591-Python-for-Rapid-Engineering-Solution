################################################################################
# Created on Mon Oct 27, 2025                                                  #
#                                                                              #
# @author: Carson Wagner                                                       #
#                                                                              #
# Project 3: Diodes                                                            #
################################################################################

# Import Libraries
import numpy as np                           # Used for logs, arrange, and loadtxt
from scipy.optimize import fsolve, leastsq   # Used to solve Voltage across Diode
import matplotlib.pyplot as plt              # For plotting
import warnings                              # Ignore Warnings

# Global Constants
K = 1.380648e-23      # Boltzmann's Constant
Q = 1.6021766208e-19  # Coulomb's Constant
NORMAL = 1e-15        # Normalized Error Constant
MAX_TOL = 1e-4        # Max Tolerance Allowed
MAX_ITER = 100        # Max Iterations Allowed


#########################
# Problem 1 Parameters  #
#########################
IS1 = 1e-9   # Source Current Value
N1 = 1.7     # Ideality Value
R1 = 11000   # Resistance of 11k Ohms
T1 = 350     # Temperature Value
STEP = 0.1   # Step size of 0.1 V

# Voltage value Starting at 0.1 V
v1 = 0.1     # Applied voltage between 0.1 to 2.5 V


#########################################
#             Problem 1                 #
#########################################

# Equation of Current in Diode (Equation Given in PDF Document)
def diode1_current(Vd):
    return IS1 * (np.exp((Vd * Q) / (N1 * K * T1)) - 1)

# Nodal Analysis Equation of Diode (Equation to solve Err, also in PDF)
def nodal_diode1(Vd, V):
    return ((Vd - V) / R1) + diode1_current(Vd)

# Plot 2 Curves for log(Diode Current) vs Source Voltage; log(Diode Current) vs Diode Voltage
def plot_diode1(log_diode_curr, source_voltage, diode_voltage):
    
    plt.title("Problem 1 Plot")
    plt.plot(source_voltage, log_diode_curr, label = 'log(Diode Current) vs Source Voltage')
    plt.plot(diode_voltage, log_diode_curr, label = 'log(Diode Current) vs Diode Voltage')
    plt.xlabel('Voltage(V)')
    plt.ylabel('log(Diode Current)')
    plt.legend(loc = 'lower right')
    plt.grid()
    plt.show()
    
##########################
# START OF MAIN PROGRAM
##########################

# Print Header for Problem 1 Display
print('\n---------------------------')
print('      Problem 1 Start      ')
print('---------------------------\n')

# Ignore Warnings
warnings.simplefilter("ignore")

# Problem 1
diode_voltage = []   # Diode Voltage Values 
diode_current = []   # Diode Current Values

# Create array of Source Voltages (Starts at 0.1, Ends at 2.5 with Step Size of 0.1)
source_voltage = np.arange(0.1, 2.6, STEP, dtype = float)

for v in source_voltage:
    
    # Calculate Diode Voltage for each Step (Step Size = 0.1)
    v1 = fsolve(nodal_diode1, v1, args=(v,))[0]
    
    # Append Voltage value to Voltage Array
    diode_voltage.append(v1)
    
    # Calculate Diode Current
    current_value = diode1_current(v1)
    diode_current.append(current_value)

# Print Diode Voltage and Current Values
print('Diode Voltage for V between 0.1V and 2.5V:')
print(diode_voltage)
print('\nDiode Current for V between 0.1V and 2.5V:')
print(diode_current)

# Plot the Plot for log(Diode Current) vs Source Voltage and Diode Voltage (Problem 1)
plot_diode1(np.log10(diode_current), source_voltage, diode_voltage)



#########################################
#             Problem 2                 #
#########################################

# Function for Diode in Problem 2 (Given in PDF)
def DiodeI(Vd, A, phi, n, T):
    k = 1.380648e-23
    q = 1.6021766208e-19
    Vt = n*k*T/q
    Is = A*T*T*np.exp(-phi*q/(k*T))
    return Is*(np.exp(Vd/Vt)-1)

# Nodal Analysis Equation of Diode in Problem 2
def nodal_diode2(Vd, Vs, A, phi, r, n, T):
    return ((Vd - Vs) / r) + DiodeI(Vd, A, phi, n, T)

# Calculate the Current of the Diode at each given voltage
def diode2_current(A, phi, r, n, T, src_volt):
    
    # Create Arrays for Diode Voltage and Current
    diode_voltage = np.zeros_like(src_volt)
    
    # Initial Guess
    volt_guess = 0.1
    
    # Calculate the Diode Current by using fsolve to find Diode Voltage
    for i in range(len(src_volt)):
        volt_guess = fsolve(nodal_diode2, volt_guess,
            args=(src_volt[i], A, phi, r, n, T),
            xtol=1e-12)[0]
        
        # Calculate Diode Voltage
        diode_voltage[i] = volt_guess
        
    # Calculate Diode Current and Return Value
    return DiodeI(diode_voltage, A, phi, n, T)


# Residual Function for Optimizing R (Resistor Value)
def optimize_r(r, phi, n, Vs, i, A, T):
    
    # Calculate Diode Current for Optimization
    curr_pred = diode2_current(A, phi, r, n, T, Vs)
    
    # Return Absolute Error
    return (curr_pred - i) / (curr_pred + i + NORMAL)

# Residual Function for Optimizing Phi (Barrier Height)
def optimize_phi(phi, n, r, Vs, i, A, T):
    
    # Calculate Diode Current for Optimization
    curr_pred = diode2_current(A, phi, r, n, T, Vs)
    
    # Return Absolute Error
    return (curr_pred - i) / (curr_pred + i + NORMAL)

# Residual Function for Optimizing n (Ideality)
def optimize_n(n, phi, r, Vs, i, A, T):
    
    # Calculate Diode Current for Optimization
    curr_pred = diode2_current(A, phi, r, n, T, Vs)
    
    # Return Absolute Error
    return (curr_pred - i) / (curr_pred + i + NORMAL)


# Plot Problem 2 Diode Graph
def plot_diode2(Vs, i, curr_pred):
    
    plt.title("Problem 2 Plot: Measured vs Predicted Diode Current")
    plt.plot(Vs, i, marker="o", linestyle="", label="Measured Data")
    plt.plot(Vs, curr_pred, linestyle="-", marker="x", label="Predicted Curve")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Diode Current (A, log scale)")
    plt.yscale("log")  # log scale for current
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


##########################
# START OF MAIN PROGRAM
##########################

# Print Header for Problem 2 Display
print('\n---------------------------')
print('      Problem 2 Start      ')
print('---------------------------\n')

# Problem 2 Parameters (Given in PDF)
A = 1e-8                  # Area
T2 = 375                  # Temperature

# Unknown Parameters, Initial Guesses (Given in PDF)
n2 = 1.5                 # Ideality Guess
r2 = 10000               # Resistance Guess
phi = 0.8                # Optimal Barrier Height Guess

# Load DiodeIV.txt File
data = np.loadtxt("DiodeIV.txt", dtype=np.float64)
Vs_measured = data[:, 0]
I_measured = data[:, 1]

# Initial iteration value
iteration = 0

# Calculate Diodude Current with Intial Guess Values
curr_pred = diode2_current(A, phi, r2, n2, T2, Vs_measured)

# Calculate Normalized Error
normal_err = np.linalg.norm((curr_pred - I_measured) / (curr_pred + I_measured + NORMAL), ord=1)

# Print Iterations and Values Headers
print("----------------------------------------------------------------------------")
print(" Iteration |       Phi       |        R           |     N     |  Residual Error ")
print("----------------------------------------------------------------------------")

# Iterating for Optimization
while iteration < MAX_ITER and normal_err > MAX_TOL:

    # Optimize phi
    phi_opt = leastsq(optimize_phi, phi, args=(n2, r2, Vs_measured, I_measured, A, T2))
    phi = phi_opt[0][0]

    # Optimize R
    r_opt = leastsq(optimize_r, r2, args=(phi, n2, Vs_measured, I_measured, A, T2))
    r2 = r_opt[0][0]

    # Optimize n
    n_opt = leastsq(optimize_n, n2, args=(phi, r2, Vs_measured, I_measured, A, T2))
    n2 = n_opt[0][0]

    # Recompute predicted current and normalized error
    curr_pred = diode2_current(A, phi, r2, n2, T2, Vs_measured)
    normal_err = np.linalg.norm((curr_pred - I_measured) / (curr_pred + I_measured + NORMAL), ord=1)

    # Print iteration results
    print(
        f" {iteration:9d} | {phi:<15.6f} | {r2:<15.3f} | {n2:<8.4f} | {normal_err:<17.6e}"
    )

    iteration += 1
    
# Print the Optimized Values of Phi, R, N
print("----------------------------------------------------------------------------")
print("\nOptimized Values")
print(f"Optimized Phi = {round(phi, 4): < {15}}")
print(f"Optimized R = {round(r2, 4): < {15}}")
print(f"Optimized N = {round(n2, 4): < {15}}")


# Plot the Graph with Optimized Values (Problem 2)
plot_diode2(Vs_measured, I_measured, curr_pred)
    

    
    

