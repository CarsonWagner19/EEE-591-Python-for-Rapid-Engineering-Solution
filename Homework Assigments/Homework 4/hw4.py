################################################################################
# Created on Sun Sept 17, 2025                                                 #
#                                                                              #
# @author: Carson Wagner                                                       #
#                                                                              #
# Program to create wealth calculator for each of the 70 years after           #
# you start work                                                               #
################################################################################

# Import Libraries
from tkinter import *            # Import GUI Package
import numpy as np               # Import Numpy 
import matplotlib.pyplot as plt  # Import Matplot for Plotting

####################################################
#  Create Constants for the Wealth Calculator
####################################################

# define the field names and their indices
FIELD_NAMES = ['Mean Return (%)', 'Std Dev Return (%)', 'Yearly Contribution ($)',
'No. Years of Contribution', 'No. Years to Retirement','Annual Retirement Spend']
F_MEAN_RETURN = 0             # index for Mean Return (%
F_STD_DEV = 1                 # index for Std Dev Return (%)
F_YEARS_CONT = 2              # index for Yearly Contribution
F_YEARS_NUM = 3               # index for No. Years of Contribution
F_YEAR_RETIRE = 4             # index for No. Years to Retirement
F_RETIRE_SPEND = 5            # index for Annual Retirement Spend
NUM_FIELDS = 6                # how many fields there are

MAX_YEARS = 70     # Max amount of years set to 70
TOTAL_POINTS = 10  # Total Points Plotted (10 total values)

# Calculte Average Wealth at Retirement
def wealth_at_retirement(entries):
    r = float(entries[F_MEAN_RETURN].get())             # Getting value of Mean (Rate entered into GUI)
    sigma = float(entries[F_STD_DEV].get())             # Getting value of STD Deviation
    Y = float(entries[F_YEARS_CONT].get())              # Getting Yearly Contribution (Start until Contributions Stop)
    years_retire = int(entries[F_YEAR_RETIRE].get())    # Getting Years to Retirement
    years_contr = int(entries[F_YEARS_NUM].get())       # Getting Years of Contribution
    S = float(entries[F_RETIRE_SPEND].get())            # Getting Retirement Spending 
    
    # Create Plot
    plt.figure()
    
    # Store wealth at retirement for each of the 10 runs taken
    final_wealths = []  
    
    # Matrix with Max Years and Total Points on Plot
    wealth_matrix = np.zeros((MAX_YEARS, TOTAL_POINTS), dtype = float)
    
    # Calculate the Average Wealth for the 10 Runs
    for n in range(TOTAL_POINTS):
        current = 0                # Start with $0
        last = 0                   # Track last year with wealth > 0
        
        # Generate Random Noise
        noise = (sigma/100)*np.random.randn(MAX_YEARS)
        
        # Iterate through 70 years and calculate retirement wealth
        for i in range(MAX_YEARS):
            
            # From Start until Contributions End
            if i < years_contr:
                current = current * (1 + (r/100) + noise[i]) + Y
            
            # End of Contributions until Retirement
            elif i < years_retire:
                current = current * (1 + (r/100) + noise[i])
            
            # Retirement to End
            else:
                current = current * (1 + (r/100) + noise[i]) - S
            
            # Stop if wealth goes negative
            if current >= 0:
                wealth_matrix[i, n] = current
                last = i
            else:
                break
        
        # Plot each of the runs
        plt.plot(range(last + 1), wealth_matrix[0: last + 1,n], '-x', label=f'Run {n+1}')
        
        # Record wealth at retirement year (or final year if broke early)
        if years_retire <= last:
            final_wealths.append(wealth_matrix[years_retire, n])
        else:
            # If wealth goes below $0
            final_wealths.append(0)
    
    # Plot the Graph
    plt.title('Wealth Over 70 Years')
    plt.xlabel('years')
    plt.ylabel('wealth')
    plt.legend()
    plt.show(block=False)
    
    # Average wealth at retirement
    wealth_average = np.mean(final_wealths)
    wealth_var.set(f"Wealth at retirement: ${wealth_average:,.2f}")

# Create the fields for user to input values (Taken from 4.3 Calculator Slides)
def makeform(root):
    entries = []                             # create an empty list
    for index in range(NUM_FIELDS):          # for each of the fields to create
        row = Frame(root)                     # get the row and create the label
        lab = Label(row, width=22, text=FIELD_NAMES[index]+": ", anchor='w')

        ent = Entry(row)                      # create the entry and init to 0
        ent.insert(0,"0")

        # fill allows the widget to take extra space: X, Y, BOTH, default=NONE
        # expand allows the widget to use up sapce in the parent widget
        row.pack(side=TOP, fill=X, padx=5, pady=5)   # place it in the GUI
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
    
        entries.append(ent)                   # add it to the list
      
    return entries  


##########################
# START OF MAIN PROGRAM
##########################

root = Tk()                                 # Create GUI
root.title("Retirement Wealth Calcultor")   # Name the window
ents = makeform(root)                       # Make the Entry Fields

# Set up label to display Average Wealth at Retirement
wealth_var = StringVar()
wealth_var.set("Wealth at retirement:")
w = Label(root, textvariable = wealth_var, anchor = 'w', justify = "left")
w.pack(anchor = 'w', padx = 10, pady = 10)

# Create Frame for Buttons
button_frame = Frame(root)
button_frame.pack(pady = 5)

# Create Quit Button
b1 = Button(button_frame, text='Quit', command=root.destroy)
b1.pack(side=LEFT, padx=5, pady=5)

# Create Calculate Button
b2 = Button(button_frame, text='Calculate',                # add balance button
          command=(lambda e=ents: wealth_at_retirement(e)))
b2.pack(side=LEFT, padx=200, pady=5)

# Run the Retirement Wealth Calculator GUI
root.mainloop()     