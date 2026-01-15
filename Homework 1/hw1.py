# -*- coding: utf-8 -*-
#############################################################
# Title: Homework 1
# Author: Carson Wagner
# Date: 8/24/2025
#############################################################

from numpy import sqrt



#############################################################
# Problem 1: Greatest Common Divisor
#############################################################

# Get user input and check if it is a valid number value
while True:
    try:
        user_input = list(map(int, input("Input a list of integers: ").split())) # Get User Input and convert the Strings into Integer List

        break
    except ValueError:
        print("Invalid input. Please enter whole number") # If not an integer, ask for whole number


#Calculate GCD using Euclidean Algorithm
def euclidean(a, b):
  while b != 0:
    a, b = b, a % b

    gcd = a # Set the value found from algorithm to GCD
  return gcd


#Iterate through List and find the GCD (Need to find how to compare each value and get correct one)
def gcd_for_list(numbers):
    
    result = numbers[0]
    
    #Iterate through the List of Numbers and use Euclidean's Algorithm
    for i in range(1, len(numbers)):
        result = euclidean(result,numbers[i])
    return result


gcd_answer = gcd_for_list(user_input)
print("The GCD is:",gcd_answer,"\n") #Included "\n" to seperate lines between Problems 1 and 2


#############################################################
# Problem 2: Prime Number Checker
#############################################################

def prime_checker(number):
    
    #Check is value is less than or equal to 1 as the value of 1 or less is NOT a prime number
    if (number <= 1):
        return print(number,"is a not a prime")
    
    # Trial Division: Iterate values in range 2 to sqrt(number) and check if divisible by these values
    for i in range(2, int(sqrt(number)) + 1):
        if number % i == 0:
            return print(number,"is not a prime number")
            
    return print(number,"is a prime number")
    

# Get user input and check if it is a valid number value
while True:
    try:
        user_input = list(map(int, input("Input a list of integers: ").split()))
        break
    except ValueError:
        print("Invalid input. Please enter whole number") # If not an integer, ask for whole number

# Iterate through the List to Check if values are Prime Numbers
for i in range (len(user_input)):
    prime_checker(user_input[i])

print() #Used to Seperate lines between Problem 2 and Problem 3

#############################################################
# Problem 3: Riemann-Zeta Function
#############################################################

def riemann_function(number, terms):

    riemann = 0
    
    # Iterate Riemann Sum from range of 2 to value of terms (10,000)
    for n in range(1, terms + 1):
        
        value = 1 / (n ** number)
        
        riemann += value
    return riemann
        

# Get user input and check if it is a positive number
while True:
    try:
        user_input = int(input("Input an Integer: "))
        
        # If input is negative, ask for new value
        if user_input < 0:
            print("That is a negative number. Please try again.")
        else:
            break
    
    except ValueError:
        print("Invalid input. Please enter whole number") # If not an integer, ask for whole number


terms = 1000000

# Calculate Riemann-Zeta Function Calulation
riemann_value = riemann_function(user_input, terms)

# Format the Riemann-Zeta Sum to the thousandths place
riemann_formatted = round(riemann_value,4)

# Print the Riemann-Zeta Sum and the Number of Terms Used
print(f"Zeta({user_input}) = {riemann_formatted} based on {terms} terms")



