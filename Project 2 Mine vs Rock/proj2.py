################################################################################
# Created on Sun Oct 8, 2025                                                   #
#                                                                              #
# @author: Carson Wagner                                                       #
#                                                                              #
# Project 2: Mine vs Rock                                                      #
################################################################################

# Import Libraries
import numpy as np                                     # needed for arrays
import pandas as pd                                    # data frame
import matplotlib.pyplot as plt                        # modifying plot
from sklearn.model_selection import train_test_split   # Test Split
from sklearn.preprocessing import StandardScaler       # scaling data
from sklearn.linear_model import LogisticRegression    # learning algorithm
from sklearn.decomposition import PCA                  # PCA package
from sklearn.metrics import accuracy_score             # grading
from sklearn.metrics import confusion_matrix           # generate the matrix
from sklearn.neural_network import MLPClassifier       # Multilayer Perceptron
from sklearn.metrics import confusion_matrix           # Confusion Matrix
from warnings import filterwarnings                    # Filter Warning Messages

RANDOM_SEED = 42     # Set constant random state for analysis that will be used for train/test split


# Run the PCA and Calculate the Confusion Matrix
def PCA_Algorithm(X_train_std, x_test_std, y_train, y_test):
    
    # Create Lists for Accuracy and Confusion Matrix
    accur = []
    confus_matrix = []
    
    # Run through all 60 components in the Data Set
    for n_comp in range (1, 61):
    
        # PCA for 62 Data Points (Includes Rock/Mine value)
        pca = PCA(n_components = n_comp)
        X_train_pca = pca.fit_transform(X_train_std)
        X_test_pca = pca.transform(X_test_std)
        
        # Create Perceptron with multiple layers and train it
        model = MLPClassifier(hidden_layer_sizes=(120, 110),
                              activation='relu',
                              max_iter=2000,
                              alpha=0.0001,
                              solver='adam',
                              tol=0.0001,
                              learning_rate='constant',
                              random_state=1)
        model.fit(X_train_pca,y_train)  # Do the actual training
        
        y_pred = model.predict(X_test_pca)
    
        # Get Acurracy and add it to the Accuracy List
        acc = accuracy_score(y_test, y_pred)
        accur.append(acc)
        
        # Calculate Confusion Matrix and Add it to the Confusion Matrix List
        confus_matrix.append(confusion_matrix(y_test, y_pred))
        
        
        # Print Accuracy for All Components
        print(f'| Components: {n_comp:<2} | Accuracy: {acc:.4f} |')
        
    return accur, confus_matrix


# Plot Accuracy vs Number of Components
def plot_accuracy(accur):
    
    plt.plot(range(1, 61), accur, marker='o', linestyle='-', markersize=4)
    plt.xlabel('Number of Components')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy vs Number of Components')
    plt.grid(True)
    plt.legend()
    plt.show()
    

# Print Accuracy Results and Confusion Matrix
def print_results(accur, confusion_matrix):
    
    # Find the index with the largest value (Best Accuracy)
    max_accuracy = np.argmax(accur)
    
    # Print Best Accuracy and Confusion Matrix
    print(f'\nMax Accuracy: {accur[max_accuracy]:.3f} with {max_accuracy + 1} components')
    print('\nConfusion Matrix: ')
    print(confusion_matrix[max_accuracy])
                


##########################
# START OF MAIN PROGRAM
##########################

# Read the database and add headers because it lacks them
df_wine = pd.read_csv('sonar_all_data_2.csv', header=None)

X = df_wine.iloc[:, :60].values   # features are in columns 0:59
y = df_wine.iloc[:, 60].values    # classes are in column 60

# now split the data into 70% Training and 30% Test
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

stdsc = StandardScaler() # apply standardization
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# Call PCA Function then Print Accuracy Results and Confusion Matrix
accuracy_list, confusion_matrix = PCA_Algorithm(X_train_std, X_test_std, y_train, y_test)
print_results(accuracy_list, confusion_matrix)

# Plot Accuracy vs Number of Components
n_components = range(1, 61)
plot_accuracy(accuracy_list)

filterwarnings('ignore')






