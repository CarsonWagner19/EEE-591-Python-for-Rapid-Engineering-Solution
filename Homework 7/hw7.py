################################################################################
# Created on Wed Nov 16, 2025                                                  #
#                                                                              #
# @author: Carson Wagner                                                       #
#                                                                              #
# Homework 7: Monte Carlo Simulation                                           #
################################################################################
import numpy as np    # Used for computing distance from origin
import random         # Used to Generate random values


TRIAL_VALUE = 100     # Number of Trials to Simulate
MAX_POINTS = 10000    # Maximum Points to Use

# Generate the Random Value of R
def random_r():
    
    # Generate Random Point for x and y values
    x = np.random.rand()
    y = np.random.rand()
    
    # Return the Value of R
    return np.sqrt(x**2 + y**2)

# Print the Results
def print_results(precision, success, pi_value):
    
    if success > 0:
        # Calculate the Averge Pi value
        avg_pi = pi_value / success
        print(f'{precision} success {success} times {avg_pi}')
    else:
        print(f'{precision} no success')
        
# Main Method
def main():
    
    precisions = [10**(-i) for i in range(1,8)]   # Precision Values from 10^-1 to 10^-7
    
    for p in precisions:
        
        # Set Initial Values
        success = 0    # Initial number of successful attempts
        pi_value = 0   # Initial Value of Pi
        
        # Iterate Through Trials (100 Attemps)
        for t in range(TRIAL_VALUE):
        
            inside = 0   # Initial number of points inside circle
            
            # Iterate trials until max points is reached
            for trials in range(1, MAX_POINTS + 1):
                
                # Generate Random value of R
                r = random_r()
                
                # Check if R value is inside the circle
                if(r <= 1):
                    inside += 1   # Add number of inside points
                
                # Stop simulation is pi estimate is within precision value
                if abs(4 * inside/trials - np.pi) <= p:
                    
                    success += 1                    # Add Successful Attempt
                    pi_value += 4 * inside/trials   # Calculate Pi Value
                    break
         
        # Print Results
        print_results(p, success, pi_value)  
                
# Run Main Method
main()            
                
                
            
                
                
            
            
        