################################################################################
# Created on Wed Nov 12, 2025                                                  #
#                                                                              #
# @author: Carson Wagner                                                       #
#                                                                              #
# Additional 591 Assignment: Ensemble Learning                                 #
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
    
    return test_accuracy, y_pred

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

    return test_accuracy, y_pred

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

    return test_accuracy, y_pred


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
    
    return test_accuracy, y_pred
    
    

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
    
    return test_accuracy, y_pred
    

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
    
    return test_accuracy, y_pred

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
percep_acc, percep_pred = perceptron(X_train_std, X_test_std, y_train, y_test, MAX_ITERATIONS_PERCEPTRON)

# Call Logical Regression Method
log_reg_acc, log_reg_pred = logical_regression(X_train_std, X_test_std, y_train, y_test, C_VAL)

# Call Support Vector Machine Method
svm_acc, svm_pred = support_vector_machine(X_train_std, X_test_std, y_train, y_test, SVM_C_VAL)

# Call Decision Tree Learning Method
dec_tree_acc, dec_tree_pred = decision_tree_learning(X_train_std, X_test_std, y_train, y_test, MAX_DEPTH_TREE)

# Call Random Forest Method
forest_acc, forest_pred = random_forest(X_train_std, X_test_std, y_train, y_test, TREES)

# Call K-Nearest Neighbor Method
k_nearest_acc, k_nearest_pred = k_nearest_neighbor(X_train_std, X_test_std, y_train, y_test, K_NEIGHBORS)



###############################################
#       Ensemble Learning Assignment          #
###############################################

# Print accuracy results of all Machine Learning Algorithms
print('\n---------------------------------------------')
print('  Machine Learning Method Accuracy Results')
print('---------------------------------------------')

print('Perceptron Accuracy:', round(percep_acc, 2))
print('Logical Regression Accuracy:', round(log_reg_acc, 2))
print('Support Vector Machine Accuracy:', round(svm_acc, 2))
print('Decision Tree Accuracy:', round(dec_tree_acc, 2))
print('Random Forest Accuracy:', round(forest_acc, 2))
print('K-Nearest Neighbor Accuracy:', round(k_nearest_acc, 2))

# Create tuple of Machine Learning algorithms test accuracy and prediction
ml_algorithms = [('Perceptron Accuracy:', percep_acc, percep_pred),
                 ('Logical Regression Accuracy:', log_reg_acc, log_reg_pred),
                 ('Support Vector Machine Accuracy:', svm_acc, svm_pred),
                 ('Decision Tree Accuracy:', dec_tree_acc, dec_tree_pred),
                 ('Random Forest Accuracy:', forest_acc, forest_pred), 
                 ('K-Nearest Neighbor Accuracy:', k_nearest_acc, k_nearest_pred)]

# Sort the Machine Learning Algorithms Tuple from Best to Worst Accuracy
ml_algorithms.sort(key=lambda x: x[1], reverse = True)

print('\n---------------------------------------------')
print('   Sorted by Accuracy (Best to Worst)')
print('---------------------------------------------')

# Print out the Algorithm's Accuracy by Best to Worst
for name, acc, _ in ml_algorithms:
    print(name, round(acc, 2)) 


#Print
print('\n--------------------------------')
print('   Ensemble Learning Accuracy')
print('--------------------------------')

################################################
#   STEP 1: Ensemble Learning with 3 Methods   #
################################################

# Setting Threshold and Taking Sum of Top 3 Most Accurate Methods
threshold = 4.5
arrays_pred = [x[2] for x in ml_algorithms]    # Create Array of Predictions

# Ensemble Learning using the Top 3 Accurate Methods
sum_3 = np.sum(arrays_pred[:3], axis = 0)

# Ensemble Learning Prediction
ensemble_3_pred = np.where(sum_3 > threshold, 2, 1)

# Ensemble Learning to Calculate Accuracy
ensemble_3_acc = accuracy_score(y_test, ensemble_3_pred)

# Print Result of Ensemble Learning of the Top 3 Most Accurate Methods
print(f'Ensemble with 3 methods: {ensemble_3_acc:.2f}')


################################################
#   STEP 2: Ensemble Learning with 4 Methods   #
################################################

# Setting threshold and Taking 3 Most Accurate Methods
threshold = 6

# Ensemble Learning using the Top 4 Accurate Methods
sum_4 = np.sum(arrays_pred[:4], axis=0)

# Ensemble Learning Prediction (Ties are Counted as Yes due to Medical Saftey Reasoning)
ensemble_4_pred = np.where(sum_4 >= threshold, 2, 1)

# Ensemble Learnign to Caculate Accuracy
ensemble_4_acc = accuracy_score(y_test, ensemble_4_pred)

# Print Result of Ensemble Learning of the Top 4 Most Accurate Methods 
print(f'Ensemble with 4 methods: {ensemble_4_acc:.2f} (Ties Counted as Yes due to Medical Saftey Caution)')


################################################
#   STEP 3: Ensemble Learning with 4 Methods   #
################################################

# Setting threshold and Taking 3 Most Accurate Methods 
threshold = 7.5

# Ensemble Learning using the Top 5 Accurate Methods
sum_5 = np.sum(arrays_pred[:5], axis=0)

# Ensemble Learning Prediction (Ties are Counted as Yes due to Medical Saftey Reasoning)
ensemble_5_pred = np.where(sum_5 >= threshold, 2, 1)

# Ensemble Learnign to Caculate Accuracy
ensemble_5_acc = accuracy_score(y_test, ensemble_5_pred)

# Print Result of Ensemble Learning of the Top 4 Most Accurate Methods 
print(f'Ensemble with 5 methods: {ensemble_5_acc:.2f}')



