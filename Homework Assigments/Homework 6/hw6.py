################################################################################
# Created on Sun Nov 9, 2025                                                   #
#                                                                              #
# @author: Carson Wagner                                                       #
#                                                                              #
# Solve Differential Equations using the Odeint Function                       #
################################################################################

# Import Libraries
import numpy as np                 # import packages
import matplotlib.pyplot as plt    # Plotting Graphs
from scipy.integrate import odeint # this is the ODE solver


# Set up Time Range to be between 0 to 7 for 700 points
def time_range():
    return np.linspace(0, 7, 700)

# Plot Y values vs T values
def plot(title, solution, time):
    plt.plot(time, solution)
    plt.ylabel('y value')
    plt.xlabel('t value')
    plt.title(title)
    plt.grid(True)
    plt.show()


#####################################
#  Problem 1: y' = cos(t), y(0) = 1
#####################################
def problem_1(y, t):
    return np.cos(t)

####################################################
#  Problem 2: y' = -y + t^2*e(-2t) + 10, y(0) = 0
####################################################
def problem_2(y, t):
    return -y + t**2 * np.exp(-2*t) + 10


###########################################################################
#  Problem 3: y'' + 4y' + 4y = 25cos(t) + 25sin(t), y(0) = 1, y'(0) = 1
###########################################################################
def problem_3(y, t):
    # y' value
    dy = y[1]
    
    # y'' value
    dyy = 25 * np.cos(t) + 25 * np.sin(t) - 4 * y[1] - 4 * y[0]
    
    return [dy, dyy]


#####################################
#       Problem 1 SOLUTION
#####################################
def problem_1_solution():
    
    # 700 points between range of 0 to 7
    t = time_range()
    
    # y(0) = 1
    y_initial = 1
    
    # Solve the Differential Equation
    sol = odeint(problem_1, y_initial, t)
    
    # Title for Plot of Y vs T
    title = "Solution of y' = cos(t),  y(0) = 1"
    
    # Plot the Solution of y vs t
    plot(title, sol, t)
  
    
#####################################
#       Problem 2 SOLUTION
#####################################
def problem_2_solution():
    
    # 700 points between range of 0 to 7
    t = time_range()
    
    # y(0) = 1
    y_initial = 0
    
    # Solve Differential Equation
    sol = odeint(problem_2, y_initial, t)
    
    # Set up Title for the Plot
    title = "Solution of y' = -y + t^2*e(-2t) + 10, y(0) = 0"
    
    # Plot the Solution of y vs t
    plot(title, sol, t)


#####################################
#       Problem 3 SOLUTION
#####################################
def problem_3_solution():
    
    # 700 points between range of 0 to 7
    t = time_range()
    
    # y(0) = 1, y'(0) = 1
    y_initial = [1,1]
    
    # Solve Differential Equation
    sol = odeint(problem_3, y_initial, t)
    
    # Set up Title for the Plot
    title = "Solution of y'' + 4y' + 4y = 25cos(t) + 25sin(t), y(0) = 1, y'(0) = 1"
    
    # Plot the Solution of y vs t
    plt.plot(t, sol[:,0], label = 'y(t)')
    plt.plot(t, sol[:,1], label = "y'(t)")
    plt.title(title)
    plt.xlabel('t value')
    plt.ylabel('y value')
    plt.legend()
    plt.grid(True)
    plt.show()
    


###################################
#             MAIN                #
###################################
def main():
    
    # Solve Problem 1
    problem_1_solution()
    
    # Solve Problem 2
    problem_2_solution()

    # Solve Problem 3
    problem_3_solution()
    
    
# Run Main Method
main()