################################################################################
# Created on Sun Sept 25, 2025                                                 #
#                                                                              #
# @author: Carson Wagner                                                       #
#                                                                              #
# Project 2, Problem 2: Machine Learning to Predict Risk of Heart Disease      #
################################################################################

from pandas import DataFrame, read_csv                # Read the CSV and convert to Data Frame
import pandas as pd                                   # Panda Library
from sklearn import datasets                          # Read the data sets
import numpy as np                                    # Needed for arrays
from sklearn.model_selection import train_test_split  # Splits database
from sklearn.preprocessing import StandardScaler      # Standardize data
from sklearn.linear_model import Perceptron           # Perceptron Algorithm
from sklearn.metrics import accuracy_score            # Grade the results
from sklearn.linear_model import LogisticRegression   # Logical Regression Algorithm
from sklearn.svm import SVC                           # SVM Algorithm
from sklearn.tree import DecisionTreeClassifier       # Decision Tree Algorithm
from sklearn.ensemble import RandomForestClassifier   # Random Forest Algorithm
from sklearn.neighbors import KNeighborsClassifier    # K-Nearest Neighbor Algorithm

# CONSTANT VARIABLES (Used to feed into algorithms)
CSV_FILE = 'heart1.csv'                 # CSV File Name
TARGET_VARIABLE = 'a1p2'                # Target Variable
MAX_ITERATIONS_PERCEPTRON = 6           # Perceptron Max Iterations
C_VAL = 1                               # Logical Regresssion Value
SVM_C_VAL = 0.25                        # SVM C Value
MAX_DEPTH_TREE = 2                      # Max Depth of Decision Tree
TREES = 500                             # Tree Value for Random Forest
K_NEIGHBORS = 25                        # K Neighbors Value

####################################
#     Print Results of Methods     #
####################################

def print_result(test_method, test_accuracy, combined_acc):
    
    # Print Test Method Name
    print(f'\n{test_method} Results:')                   # Method Name
    print(f'    Accuracy: {test_accuracy:.2f}')          # Test Accuracy
    print(f'    Combined Accuracy: {combined_acc:.2f}\n')  # Combined Accuracy


####################################
#      Perceptron Method           #
####################################
    
def perceptron(X_train_std, X_test_std, y_train, y_test, max_iterations):
    
    test_method = 'Perceptron'
    
    ppn = Perceptron(max_iter=max_iterations, tol=1e-3, eta0=0.001,
                     fit_intercept=True, random_state=0, verbose=True)
    ppn.fit(X_train_std, y_train)              # do the training

    test_sample = len(y_test)                  # Number of Test Samples
    y_pred = ppn.predict(X_test_std)           # now try with the test data

    # Note that this only counts the samples where the predicted value was wrong
    test_miss = (y_test != y_pred).sum()  # how'd we do?
    test_accuracy = accuracy_score(y_test, y_pred)

    # vstack puts first array above the second in a vertical stack
    # hstack puts first array to left of the second in a horizontal stack
    # NOTE the double parens!
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    
    # Get combined number of samples
    combined_sample = len(y_combined)

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = ppn.predict(X_combined_std)
    combined_miss = (y_combined != y_combined_pred).sum()
    combined_acc = accuracy_score(y_combined, y_combined_pred)
    
    # Print Results of Perceptron Method
    print_result(test_method, test_accuracy,combined_acc)
    

####################################
#    Logical Regression Method     #
####################################
    
def logical_regression(X_train_std, X_test_std, y_train, y_test, c_val):
    
    # Method Name for Printing
    test_method = 'Logical Regression'
    
    # Logical Regression Method in Action
    lr = LogisticRegression(C=c_val, solver='liblinear', \
                            multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train) # apply the algorithm to training data
        
    # Test Accuracy
    y_pred = lr.predict(X_test_std)
    test_accuracy = accuracy_score(y_test, y_pred)
        
    # combine the train and test data
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
        
    # Combined Accuracy
    y_combined_pred = lr.predict(X_combined_std)
    combined_acc = accuracy_score(y_combined, y_combined_pred)
    
    # Test Accuracy
    y_pred = lr.predict(X_test_std)
    test_accuracy = accuracy_score(y_test, y_pred)
        
    #Print Results
    print_result(test_method, test_accuracy, combined_acc)


####################################
#   Support Vector Machine Method  #
####################################

def support_vector_machine(X_train_std, X_test_std, y_train, y_test, c_val):
    
    # Test Method Name
    test_method = 'Support Vector Machine'
    
    # Support Vector Machine
    # kernel - specify the kernel type to use
    # C - the penalty parameter - it controls the desired margin size
    # Larger C, larger penalty
    
    svm = SVC(kernel='linear', C=c_val, random_state=0)
    svm.fit(X_train_std, y_train)                        # do the training
    y_pred = svm.predict(X_test_std)                     # work on the test data
    
    # Test Accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Combine the train and test sets
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    
    # Analyze the Combined Data Sets
    y_combined_pred = svm.predict(X_combined_std)
    
    # Combined Accuracy
    combined_acc = accuracy_score(y_combined, y_combined_pred)
    
    # Print Result
    print_result(test_method, test_accuracy, combined_acc)

####################################
#      Decision Tree Learning      #
####################################

def decision_tree_learning(X_train_std, X_test_std, y_train, y_test, max_depth):
    
    # Test Method Name
    test_method = 'Decision Tree Learning'
    
    # Create the classifier and train it
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=5 ,random_state=0)
    tree.fit(X_train,y_train)

    # Work on the test data
    y_pred = tree.predict(X_test_std)         
    
    # Test Accuracy
    test_accuracy = accuracy_score(y_test, y_pred)

    # Combine the train and test data
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    
    # See how we do on the combined data
    y_combined_pred = tree.predict(X_combined)
    combined_acc = accuracy_score(y_combined, y_combined_pred)
    
    # Graph the Decision Tree (Feature Names is Columns of heart1.csv)
    #export_graphviz(tree,out_file='tree.dot',
    #                feature_names=[features])
    
    # Print Result
    print_result(test_method, test_accuracy, combined_acc)


####################################
#         Random Forest            #
####################################

def random_forest(X_train_std, X_test_std, y_train, y_test, trees):
    
    # Test Method Name
    test_method = 'Random Forest'

    # create the classifier and train it
    # n_estimators is the number of trees in the forest
    # the entropy choice grades based on information gained
    # n_jobs allows multiple processors to be used
    forest = RandomForestClassifier(criterion='entropy', n_estimators=trees, \
                                    random_state=1, n_jobs=4)
    
    # Create the classifier and train it
    forest.fit(X_train,y_train)
    y_pred = forest.predict(X_test) # see how we do on the test data
    
    # Test Accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Combine the training and test data
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    
    # Predict the combined data
    y_combined_pred = forest.predict(X_combined)
  
    # Combined Accuracy
    combined_acc = accuracy_score(y_combined, y_combined_pred)

    # Print Results
    print_result(test_method, test_accuracy, combined_acc)
    

####################################
#       K-Nearest Neighbor         #
####################################
    
def k_nearest_neighbor(X_train_std, X_test_std, y_train, y_test, neighbors):
    
    test_method = 'K-Nearest Neighbor'

    knn = KNeighborsClassifier(n_neighbors=neighbors,p=2,metric='minkowski')
    knn.fit(X_train_std,y_train)
    
    # Run test data and get test accuracy
    y_pred = knn.predict(X_test_std)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Combine the training and test data
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    
    # Calculate the Combined Test Accuracy
    y_combined_pred = knn.predict(X_combined_std)
    combined_acc = accuracy_score(y_combined, y_combined_pred)
    
    # Print Results
    print_result(test_method, test_accuracy, combined_acc)


##########################
# START OF MAIN PROGRAM
##########################
    
# Read the CSV File to gather data
get_data = pd.read_csv(CSV_FILE)

# Turn Data into Array
data_array = get_data.to_numpy()

# Split the Data into Training and Testing Data
X = data_array[:, :13]                               # Seperate all the Features
y = data_array[:, 13].ravel()                        # Get all Classifications

# Split the Data into Training and Test: 70% Training, 30% Test
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.3,random_state=0)

# Mean and Standard Deviation can be overridden
sc = StandardScaler()                     # Create the standard scalar
sc.fit(X_train)                           # Compute the required transformation
X_train_std = sc.transform(X_train)       # Apply to the training data
X_test_std = sc.transform(X_test)         # Same transformation of test data


# Call Perceptron Method
perceptron(X_train_std, X_test_std, y_train, y_test, MAX_ITERATIONS_PERCEPTRON)

# Call Logical Regression Method
logical_regression(X_train_std, X_test_std, y_train, y_test, C_VAL)

# Call Support Vector Machine Method
support_vector_machine(X_train_std, X_test_std, y_train, y_test, SVM_C_VAL)

# Call Decision Tree Learning Method
decision_tree_learning(X_train_std, X_test_std, y_train, y_test, MAX_DEPTH_TREE)

# Call Random Forest Method
random_forest(X_train_std, X_test_std, y_train, y_test, TREES)

# Call K-Nearest Neighbor Method
k_nearest_neighbor(X_train_std, X_test_std, y_train, y_test, K_NEIGHBORS)

