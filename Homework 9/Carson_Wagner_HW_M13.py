################################################################################
# Created on Wed Nov 26, 2025                                                  #
#                                                                              #
# @author: Carson Wagner                                                       #
#                                                                              #
# Homework 9: Calculators                                                      #
################################################################################

import math                         # Used for calculator functions
import numpy as np                  # Used for calculator functions
import matplotlib.pyplot as plt     # Graphing Calculator Function
import tkinter as tk                # Calculator GUI
from tkinter import messagebox      # Calculator Message Box



#####################################
#       Basic Calculator
#####################################
class basic_calculator:
    
    # Initialize name of calulator
    def __init__ (self, name):
        self.name = name
    
    # Adding Function
    def add(self, a, b):
        return a + b
    
    # Subtract Function
    def subtract(self, a, b):
        return a - b
    
    # Multiply Function
    def multiply(self, a, b):
        return a * b
    
    # Divide Function (Include Divide by Zero Error Catch)
    def divide(self, a, b):
        # Check is b is value of 0
        if b == 0:
            raise ValueError("Error: Division by Zero is not allowed")
        
        return a / b
    

#####################################
#       Scientific Calculator
#####################################
class scientific_calculator(basic_calculator):
    
    LOG_BASE = 10.0
    
    # Initialize Name of Calculator
    def __init__(self, name):
        super().__init__(name)
        
    # Logarithmic Function
    def log(self, a):
        # Check if a is positive number
        if a <= 0:
            raise ValueError("Error: Logarithmic Input must be positive value")
        
        return math.log(a, self.LOG_BASE)
    
    # Exponent Function
    def exponent(self, a):
        return self.LOG_BASE ** a

    # Sine Function
    def sine(self, a):
        return math.sin(a)
    
    # Cosine Function
    def cosine(self, a):
        return math.cos(a)
    
#####################################
#       Graphing Calculator
#####################################
class graphing_calculator(scientific_calculator):

    # Initialize Name of Calculator
    def __init__(self, name):
        super().__init__(name)
        
    # Graph Values
    def plot(self, y_values):
        # Check if y_values array is not empty
        if not y_values:
            print("Error: Y-Value List is Empty. Cannot Plot")
            return
        
        # Start at x-value and increment by 1 for each y_value
        unique_x_start = 22
        x_values = np.arange(unique_x_start, unique_x_start + len(y_values))
        y_values_arr = np.array(y_values)
        
        print(f"\n--- {self.name} Plotting ---")
        print(f"Y-Values (Input Array): {y_values}")
        print(f"X-Values (Unique Domain): Starts at {unique_x_start} and increments.")
        
        # Plot Graph
        plt.figure()
        plt.plot(x_values, y_values_arr, marker = 'o', linestyle='-')
        plt.title(f'{self.name} Curve Plot')
        plt.xlabel(f'Unique X-Axis Domain (Starting at {unique_x_start} and increments)')
        plt.ylabel('Y-Axis (User Input)')
        plt.grid(True)
        plt.show()
        
        print('Plot Finished')



#####################################
#       Calculator GUi
#####################################
def calculator_GUI():
    
    # Create Root for GUI (Set up Light Blue Background)
    root = tk.Tk()
    root.title("Basic Calculator")
    root.configure(bg='#add8e6')
    
    # Create Basic Calculator
    basic_calc = basic_calculator('Calculator GUI')

    # Calculator Logic
    def calculate(operation):
        
        # Read inputs from User Input
        try:
            
            # Read inputs from user
            num1 = float(entry1.get())
            num2 = float(entry2.get())

            result = None
            #operating_symbol = ""
            
            if operation == 'add':
                result = basic_calc.add(num1, num2)
                operating_symbol = "+"
                
            elif operation == "subtract":
                result = basic_calc.subtract(num1, num2)
                operating_symbol = "-"
                    
            elif operation == "multiply":
                result = basic_calc.multiply(num1, num2)
                operating_symbol = "x"
            
            elif operation == "divide":
                result = basic_calc.divide(num1, num2)
                operating_symbol = "/"
                
            # Print Result Label
            result_label.config(text=f'Result: {result}')
            
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            result_label.config(text="Result: Error")
               
    ##########################
    #       GUI Setup        #
    ##########################
        
    # Input Frame
    input_frame = tk.Frame(root, pady=10, bg='#add8e6')
    input_frame.pack(pady=10)
    
    # Text Box 1 (Num 1)
    tk.Label(input_frame, text='Number 1:',fg='black', bg='#add8e6').pack(side=tk.LEFT, padx=5)
    entry1 = tk.Entry(input_frame, width=10)
    entry1.pack(side=tk.LEFT, padx=10)
    entry1.insert(0, "15.7") # Default Value
        
    # Text Box 2 (Num 2)
    tk.Label(input_frame, text='Number 2:',fg='black', bg='#add8e6').pack(side=tk.LEFT, padx=5)
    entry2= tk.Entry(input_frame, width=10)
    entry2.pack(side=tk.LEFT, padx=10)
    entry2.insert(0, "3.1")  # Default Value 2
        
    # Button Frame
    button_frame = tk.Frame(root, pady=10, bg='#add8e6')
    button_frame.pack(pady=10)
        
    # Buttons (Add, Sub, Mult, Div)
        
    tk.Button(button_frame, text='+', command=lambda: calculate('add')).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text='-', command=lambda: calculate('subtract')).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text='/', command=lambda: calculate('divide')).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text='x', command=lambda: calculate('multiply')).pack(side=tk.LEFT, padx=5)

    # Result Label
    result_label = tk.Label(root, text='Result', fg='white', bg='black', pady=15)
    result_label.pack()
        
    # Run the Calculator GUI
    root.mainloop()
        
###################################
#            MAIN                 #
###################################

# Basic Calculator Demonstration using Addition
basic_calc_obj = basic_calculator('Basic Calculator')
a, b = 40.5, 30.2
result_basic = basic_calc_obj.add(a, b)
        
# Print Addition Result
print(f'\n--- {basic_calc_obj.name} Demonstration ---')
print('Function: Addition')
print(f'Calculation: {a} + {b}')
print(f'Result: {result_basic}')
print('-' * 40)
    

# Scientific Calculator Demonstration
scientific_calc = scientific_calculator('Scientific Calculator')
log_input = 225.0

try:
    result_scientific = scientific_calc.log(log_input)
    
    print(f'\n--- {scientific_calc.name} Demonstration ---')
    print(f'Function: Logarithmic (Base {scientific_calc.LOG_BASE})')
    print(f'Calculation: log_{scientific_calc.LOG_BASE}({log_input})')
    print(f'Result: {result_scientific:.4f}')
    print('-' * 40)

except ValueError as e:
    print(f'Scientific Error: {e}')

# Graphical Calculator Demonstration
graphical_calc = graphing_calculator('Graphical Calculator')
y_data_arr = [15, 5, 12, 8, 15, 6, 11]      # Input of Y-Values
graphical_calc.plot(y_data_arr)
print('-' * 40)

# Basic Calculator GUI (Run after Plot is Closed)
print('\n--- Launching Basic Calculator GUI ---')
calculator_GUI()




    
