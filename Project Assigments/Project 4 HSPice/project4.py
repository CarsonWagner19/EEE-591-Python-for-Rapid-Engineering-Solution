################################################################################
# Created on Thur Nov 6, 2025                                                  #
#                                                                              #
# @author: Carson Wagner                                                       #
#                                                                              #
# Project 4: Driving a Tool with HSPICE                                        #
################################################################################

import numpy as np      # package needed to read the results file
import subprocess       # package needed to lauch hspice
import os               #


MAX_FAN_VALUE = 5                    # Max number of fans
MAX_INVERTER_VALUE = 8              # Max number of inverters
INPUT_FILE_NAME = "header.sp"        # Header Input File name
NETLIST_FILE = "InvChain.sp"         # Netlist File namm
OUTPUT_FILE = "InvChain.mt0.csv"     # Output File

# Run hspice simulation for fan and the amount of inverters (Given in Canvas)
def run_hspice():
    # launch hspice. Note that both stdout and stderr are captured so
    # they do NOT go to the terminal!
    proc = subprocess.Popen(["hspice","InvChain.sp"],
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    output, err = proc.communicate()
 
    
# Extract tphl (Propagation Delay)
def get_tphl():
    # extract tphl from the output file
    data = np.recfromcsv(OUTPUT_FILE,comments="$",skip_header=3)
    tphl = data["tphl_inv"]
    return tphl



# Generate a netlist for the fan and inverter combinations
def generate_netlist(netlist, fan, inverter):
    
    # Add .param fan line to hspice file 
    netlist += f'\n\n.param fan={fan}\n' 
    
    # Condition if there is 1 inverter, add A to Z 
    if(inverter == 1): 
        netlist += 'Xinv1 a z inv M=1\n' 
    else: 
        #Starting Node 
        start_node = ord('a') 
        
        # Add First Inverter to the Netlist 
        netlist += 'Xinv1 a b inv M=1\n' 
        
        # Iterate through inverters and add values to Netlist 
        for i in range(2, inverter): 
            
            # Add the next inverter to Netlist 
            netlist += f'Xinv{i} {chr(start_node + 1)} {chr(start_node + 2)} inv M=fan**{i-1}\n' 
            
            # Increase the ASCII value of Starting Node (A -> B and so on) 
            start_node += 1 
            
        # Final Inverter added to Netlist 
        netlist += f'Xinv{inverter} {chr(start_node + 1)} z inv M=fan**{inverter-1}\n' 
            
    # End Netlist Generation 
    netlist += '.end\n' 
        
    # Write Netlist to a File then Close File 
    file = open(NETLIST_FILE, 'w') 
    file.write(netlist) 
    file.close()
    
# Main Method
def main():
    
    # Open and Read Header.sp file to create netlist
    file = open(INPUT_FILE_NAME)
    netlist = file.read()
    
    
    # Set up Initial Values
    min_delay = float('inf')   # Minimum Delay Initial Value
    optimal_fan = 0            # Initial Optimal Fan Value
    optimal_inv = 0            # Initial Optimal Inverter Value
    
    # Create list for Fan and append to list
    fan_group = []
    for fan in range(2, MAX_FAN_VALUE):
        fan_group.append(fan)
    
    # Create list for Inverters and Append to the List
    inv_group = []
    for inv in range(0, MAX_INVERTER_VALUE):
        # The 'if' condition from the comprehension is placed here
        if inv % 2 != 0:
            inv_group.append(inv)
    
    # Sweep through Fan and Inverters to find tphl value for each simulation
    for fan in fan_group:
        for inv in inv_group:
            
            # Generate netlist
            generate_netlist(netlist, fan, inv)
            
            # Run Hspice Simulation
            run_hspice()
            
            # Check if Output File is ready before moving on
            ready = False
            while(not ready):
                ready = os.path.isfile(OUTPUT_FILE)
            
            # Find tphl value (Propagation Delay)
            tphl = get_tphl()
            
            # Print Results
            print(f'N {inv: < {2}} fan {fan: < {1}} tphl {tphl: <{10}}')
            
            # Find the Optimal Fan, # of Inverters, and Min Propagation Delay
            if(tphl < min_delay):
                min_delay = tphl
                optimal_fan = fan
                optimal_inv = inv
    
    # Print Optimal Min Delay, Fan, and Number of Inverters
    print('\nOptimal Values:')
    print(' Fan:', optimal_fan)
    print(' Number of Inverters:', optimal_inv)
    print(' tphl:', min_delay)
    
# Run Main
main()
        
    
    
    
    
    
    
    
    
    
    


