################################################################################
# Created on Thur Dec 4, 2025                                                  #
#                                                                              #
# @author: Carson Wagner                                                       #
#                                                                              #
# Final Project: Cats vs Dogs                                                  #
################################################################################

import torch                         # import the various PyTorch packages
import torch.nn            as nn
import torch.optim         as optim
import torch.nn.functional as F
import numpy as np
import time
from torchvision import datasets     # the data repository
from torchvision import transforms   # tranforming the data
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

################################################
# Setting up constants for training
################################################
BATCH_SIZE  = 64            # number of samples per gradient update
NUM_CLASSES = 2              # how many classes to classify (2 classes, Dogs and Cats)
EPOCHS      = 10             # how many epochs to run trying to improve (CIFAR10 Needs more for Accuracy)
LEARNING_RATE = 0.001        # For Accuracy/Runtime Trade-off
ROOT_DIR = '~/CIFAR10_data'  # Root Directory of Data
MY_NAME = 'Carson Wagner'    # My Name to be Printed for Academic Integrity Purposes


################################################
# Create the network
################################################
class CIFAR10_NET( nn.Module ):

    ################################################
    # Initializing the network
    ################################################
    def __init__( self, num_classes, input_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(3,3))  # Updated to be modular (RGB Model: in_channels=3, Grayscale: in_channels=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.mpool = nn.MaxPool2d( kernel_size=2 )
        self.drop1 = nn.Dropout( p=0.25 )
        self.flat  = nn.Flatten()
        self.fc1   = nn.Linear( in_features=64 * 14 * 14, out_features=128 )  # Updated to be (64 * 14 * 14 = 12544)
        self.drop2 = nn.Dropout( p=0.5 )
        self.fc2   = nn.Linear( in_features=128, out_features=num_classes )

    ################################################
    # Forward pass of the network
    ################################################
    def forward( self, x ):
        x = F.relu( self.conv1( x ) )
        x = F.relu( self.conv2( x ) )
        x = self.mpool( x )
        x = self.drop1( x )
        x = self.flat(  x )
        x = F.relu( self.fc1( x ) )
        x = self.drop2( x )
        x = self.fc2(   x )
        return x


################################################
# Data Loading and Filtering Function
################################################
# This function applies the required transforms and then filters the data.
def load_and_filter_data(transform, is_train):
    
    # **REQUIRED COMMAND USED HERE**
    dataset = datasets.CIFAR10(root=ROOT_DIR, train=is_train,
                               download=True, transform=transform)
    
    # Find indices for Cats (3) and Dogs(5)
    filter_mask = (np.array(dataset.targets) == 3) | \
                  (np.array(dataset.targets) == 5)
    
    # Filter Data and targets
    filtered_data = dataset.data[filter_mask]
    
    # Remap Target: Cats (3 -> 0), Dogs (5 -> 1)
    filtered_targets = np.array(dataset.targets)[filter_mask]
    filtered_targets[filtered_targets == 3] = 0  # Cat -> 0
    filtered_targets[filtered_targets == 5] = 1  # Dog -> 1
    
    # This ensures both the data and the targets are ready for the DataLoader
    final_data = []
    final_targets = torch.tensor(filtered_targets, dtype=torch.long)
    
    # Re-Apply the transformations to the filtered data arrays
    for img_array in filtered_data:
        # Convert NumPy array back to PIL Image, apply transform, and collect
        img_pil = Image.fromarray(img_array)
        img_tensor = transform(img_pil)
        final_data.append(img_tensor)
        
    final_data_tensor = torch.stack(final_data)

    # Create a TensorDataset from the filtered, transformed data
    return TensorDataset(final_data_tensor, final_targets)
    
################################################
#          Main Training and Testing
################################################
def train_and_test(is_grayscale):
    
    # Check if Grayscale/RGB and create transformation and setup model
    if is_grayscale:
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1), 
            transforms.Normalize([0.5], [0.5])])
        input_channels = 1
        model_name = "Grayscale"
    else:
        # RGB: 3 channels, 3 values for mean/std
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        input_channels = 3
        model_name = "RGB"

    # Load and Filter the Data
    print(f"\n--- Preparing {model_name} Data ---")
    trainset = load_and_filter_data(data_transforms, is_train=True)
    testset  = load_and_filter_data(data_transforms, is_train=False)
    
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Start time for runtime calculation
    start_time = time.time()
    
    model = CIFAR10_NET(NUM_CLASSES, input_channels) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 

    # Training Loop
    model.train()
    num_batches = len(trainloader)
    
    # Print which training model is being ran and the EPOCH amount
    print(f"--- Starting {model_name} Training (Epochs: {EPOCHS}) ---")
    for epoch in range(EPOCHS):
        for batch_idx, (images, labels) in enumerate(trainloader):
            
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            if (batch_idx % (num_batches // 5) == 0) and (batch_idx != 0):
                 print(f"Epoch {epoch+1}/{EPOCHS}\tBatch {batch_idx}/{num_batches}\tLoss: {loss.mean().item():.4f}")

    # Test Loop
    model.eval()
    num_correct = 0
    num_attempts = 0

    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            guesses = torch.argmax(outputs, 1)

            num_correct += torch.sum(labels == guesses).item()
            num_attempts += len(guesses)
    
    # Final Calculation
    end_time = time.time()
    runtime = end_time - start_time
    accuracy = 100 * num_correct / num_attempts

    print(f"--- Finished {model_name}. Accuracy: {accuracy:.1f}% | Runtime: {runtime:.2f} seconds ---\n")
    return accuracy, runtime
    
################################
#       MAIN METHOD 
################################
# Run RGB Algorithm
rgb_acc, rgb_time = train_and_test(is_grayscale=False)
  
# Run Grayscale Algorithm
gray_acc, gray_time = train_and_test(is_grayscale=True)
  
# 3. Determine Recommendation
# If RGB is > 2% more accurate, recommend RGB. Otherwise, prioritize runtime savings.
if rgb_acc > gray_acc + 2.0:
    recommended_algo = "RGB" 
elif gray_acc > rgb_acc + 2.0:
    recommended_algo = "Grayscale"
elif gray_time < rgb_time: 
    recommended_algo = "Grayscale"
else:
    recommended_algo = "RGB" 
  
  
# Print Output
print("\n--- Final Results ---")
print(f"RGB Accuracy: {rgb_acc:.1f}%")
print(f"Grayscale Accuracy: {gray_acc:.1f}%")
print(f"RGB Runtime: {rgb_time:.5f} seconds")
print(f"Grayscale Runtime: {gray_time:.5f} seconds")
print(f"{MY_NAME} recommends {recommended_algo} algorithm")
