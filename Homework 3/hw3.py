################################################################################
# Created on Sun Sept 14, 2025                                                 #
#                                                                              #
# @author: Carson Wagner                                                       #
#                                                                              #
# Program to solve resister network with voltage and/or current sources        #
################################################################################

import numpy as np                     # needed for arrays
from numpy.linalg import solve         # needed for matrices
from read_netlist import read_netlist  # supplied function to read the netlist
import comp_constants as COMP          # needed for the common constants

# this is the list structure that we'll use to hold components:
# [ Type, Name, i, j, Value ]

################################################################################
# How large a matrix is needed for netlist? This could have been calculated    #
# at the same time as the netlist was read in but we'll do it here             #
# Input:                                                                       #
#   netlist: list of component lists                                           #
# Outputs:                                                                     #
#   node_cnt: number of nodes in the netlist                                   #
#   volt_cnt: number of voltage sources in the netlist                         #
################################################################################

def get_dimensions(netlist):           # pass in the netlist

    node_cnt = 0                       # Initialize node_cnt to 0
    volt_cnt = 0                       # Initialize volt_cnt to 0
   
    # Find the amount of nodes in netlist by finding the max node value
    for comp in netlist:
        
        # Compare I, J, and current node_cnt and take the highest value
        node_cnt = max(node_cnt, comp[COMP.I], comp[COMP.J])
        
        # Add 1 to volt_cnt for each value in netlist that is equal to VS
        if comp[COMP.TYPE] == COMP.VS:
            volt_cnt += 1
            
    return node_cnt, volt_cnt
            
    
    #print(' Nodes ', node_cnt, ' Voltage sources ', volt_cnt)
    return node_cnt,volt_cnt

################################################################################
# Function to stamp the components into the netlist                            #
# Input:                                                                       #
#   y_add:    the admittance matrix                                            #
#   netlist:  list of component lists                                          #
#   currents: the matrix of currents                                           #
#   node_cnt: the number of nodes in the netlist                               #
# Outputs:                                                                     #
#   node_cnt: the number of rows in the admittance matrix                      #
################################################################################

def stamper(y_add,netlist,currents,node_cnt):
    # return the total number of rows in the matrix for
    # error checking purposes
    # add 1 for each voltage source...
    
    volt_src_cnt = 0                      # Count # of Voltage Sources
    
    for comp in netlist:                  # for each component...
        #print(' comp ', comp)            # which one are we handling...

        # extract the i,j and fill in the matrix...
        # subtract 1 since node 0 is GND and it isn't included in the matrix
        i = comp[COMP.I] - 1
        j = comp[COMP.J] - 1

        # Check if it's a Restior
        if (comp[COMP.TYPE] == COMP.R ):
            if (i >= 0):                            
                y_add[i,i] += 1.0/comp[COMP.VAL]    # Add 1/VAL for entries at [i,i] 
            
            if (j >= 0):
                y_add[j,j] += 1.0/comp[COMP.VAL]    # Add 1/VAL for entries at [j,j]
            
            if (i >= 0 and j >= 0):
                y_add[i,j] -= 1.0/comp[COMP.VAL]    # Add 1/VAL for entries at [i,j]
                y_add[j,i] -= 1.0/comp[COMP.VAL]    # Add 1/VAL for entries at [j,i]
        
        # Check if it's a current source
        elif (comp[COMP.TYPE] == COMP.IS):          
            if (i >= 0):
                currents[i] -= 1.0 * comp[COMP.VAL]  # Add I from entry i
            if (j >= 0):
                currents[j] += 1.0 * comp[COMP.VAL]  # Subtract I from entry j
        
        # Check if it's a Voltage Source
        elif (comp[COMP.TYPE] == COMP.VS):
            
            # Initialize number of rows/cols in Voltage Admittance Matrix
            M = node_cnt + volt_src_cnt                            
            
            if(i >= 0):
                y_add[M, i] = 1.0
                y_add[i, M] = 1.0
            if(j >= 0):
                y_add[M, j] = -1.0
                y_add[j, M] = -1.0
            
            # Stamping the Currents Vector
            currents[M] = comp[COMP.VAL]
            
            # Increment Voltage Source Counter by 1
            volt_src_cnt += 1

    return node_cnt # should be same as number of rows!

################################################################################
# Start the main program now...                                                #
################################################################################

# Read the netlist!
netlist = read_netlist()

# Find and Print Count of Nodes and Voltage Sources
node_count, voltage_src_cnt = get_dimensions(netlist)

# Find total Number of Nodes
total_nodes = node_count + voltage_src_cnt

# Initialize the admittance, voltage, and current matrices
admittance_matrix = np.zeros((total_nodes, total_nodes), dtype = float)
current_matrix = np.zeros(total_nodes, dtype = float)
voltage_matrix = np.zeros(total_nodes, dtype = float)

# Call stamper function to find number of nodes
node_count = stamper(admittance_matrix, netlist, current_matrix, node_count)

# Solve Matrix and Print
voltage_values = solve(admittance_matrix, current_matrix)
print('Vector is',voltage_values)

# Gather only the voltage values and take the average
node_voltages = voltage_values[:node_count]
voltage_avg = np.mean(node_voltages)

# Round voltage average to 4 decimal places and print
volt_avg_format = round(voltage_avg, 4)
print('Voltages average is', volt_avg_format)

# Print the netlist so we can verify we've read it correctly
#for index in range(len(netlist)):
#    print(netlist[index])
#print("\n")

#EXTRA STUFF HERE!

