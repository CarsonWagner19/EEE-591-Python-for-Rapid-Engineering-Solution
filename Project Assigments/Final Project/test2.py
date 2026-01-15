import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset # Import TensorDataset for filtered data
from PIL import Image

################################################
# Setting up constants for training
################################################
# **TUNING VALUES - ADJUST THESE TO MAXIMIZE ACCURACY**
BATCH_SIZE  = 64    # Recommended batch size
NUM_CLASSES = 2     # Cat (0) and Dog (1)
EPOCHS      = 15    # Tune this for accuracy/runtime trade-off
LEARNING_RATE = 0.001
ROOT_DIR = '~/CIFAR10_data' 
MY_NAME = 'Carson Wagner' 

################################################
# Model Definition
################################################
class CIFAR10_NET( nn.Module ):
    def __init__( self, num_classes, input_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(3,3)) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.mpool = nn.MaxPool2d( kernel_size=2 )
        self.drop1 = nn.Dropout( p=0.25 )
        self.flat  = nn.Flatten()
        self.fc1   = nn.Linear( in_features=64 * 14 * 14, out_features=128 ) 
        self.drop2 = nn.Dropout( p=0.5 )
        self.fc2   = nn.Linear( in_features=128, out_features=num_classes )

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
    
    # Remap Target: 3 -> 0 (Cats), 5 -> 1 (Dogs)
    filtered_targets = np.array(dataset.targets)[filter_mask]
    filtered_targets[filtered_targets == 3] = 0  # Cat -> 0
    filtered_targets[filtered_targets == 5] = 1  # Dog -> 1
    
    # -------------------------------------------------------------
    # NOTE: Since CIFAR10 is loaded with 'transform', the data here
    # is already a tensor/transformed NumPy array. If it's still a 
    # NumPy array, we need to convert it to a Tensor before DataLoader.
    # We must iterate through the filtered indices and apply the transform 
    # individually because torchvision's transforms work on PIL images, 
    # but filtering was done on the numpy array.
    # -------------------------------------------------------------
    
    # Re-apply the transformations to the filtered NumPy data arrays
    # This ensures both the data and the targets are ready for the DataLoader
    final_data = []
    final_targets = torch.tensor(filtered_targets, dtype=torch.long)
    
    for img_array in filtered_data:
        # Convert NumPy array back to PIL Image, apply transform, and collect
        img_pil = Image.fromarray(img_array)
        img_tensor = transform(img_pil)
        final_data.append(img_tensor)
        
    final_data_tensor = torch.stack(final_data)

    # Create a TensorDataset from the filtered, transformed data
    return TensorDataset(final_data_tensor, final_targets)


################################################
# Main Training and Testing Function (Dual Run Logic)
################################################
def train_and_test(is_grayscale):
    
    # --- 1. Define Transformations & Model Setup ---
    if is_grayscale:
        # Grayscale: 1 channel, 1 value for mean/std
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1), 
            transforms.Normalize([0.5], [0.5])
        ])
        input_channels = 1
        model_name = "Grayscale"
    else:
        # RGB: 3 channels, 3 values for mean/std
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        input_channels = 3
        model_name = "RGB"

    # --- 2. Load and Filter Data ---
    print(f"\n--- Preparing {model_name} Data ---")
    trainset = load_and_filter_data(data_transforms, is_train=True)
    testset  = load_and_filter_data(data_transforms, is_train=False)
    
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    start_time = time.time()
    
    model = CIFAR10_NET(NUM_CLASSES, input_channels) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 

    # --- 3. Training Loop ---
    model.train()
    num_batches = len(trainloader)

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

    # --- 4. Testing Loop ---
    model.eval()
    num_correct = 0
    num_attempts = 0

    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            guesses = torch.argmax(outputs, 1)

            num_correct += torch.sum(labels == guesses).item()
            num_attempts += len(guesses)
    
    # --- 5. Final Calculation ---
    end_time = time.time()
    runtime = end_time - start_time
    accuracy = 100 * num_correct / num_attempts

    print(f"--- Finished {model_name}. Accuracy: {accuracy:.1f}% | Runtime: {runtime:.2f} seconds ---\n")
    return accuracy, runtime

################################################
# Main Execution Block
################################################
if __name__ == '__main__':
    
    # 1. Run RGB Algorithm
    rgb_acc, rgb_time = train_and_test(is_grayscale=False)
    
    # 2. Run Grayscale Algorithm
    gray_acc, gray_time = train_and_test(is_grayscale=True)
    
    # 3. Determine Recommendation
    # Recommendation logic: If RGB is > 2% more accurate, recommend RGB. Otherwise, prioritize runtime savings.
    
    if rgb_acc > gray_acc + 2.0:
        recommended_algorithm = "RGB" 
    elif gray_acc > rgb_acc + 2.0:
         recommended_algorithm = "Grayscale"
    elif gray_time < rgb_time: 
        recommended_algorithm = "Grayscale"
    else:
        recommended_algorithm = "RGB" 
    
    
    # --- 4. Print Final Output (REQUIRED FORMAT) ---
    print("\n--- Final Results ---")
    print(f"RGB Accuracy: {rgb_acc:.1f}%")
    print(f"Grayscale Accuracy: {gray_acc:.1f}%")
    print(f"RGB Runtime: {rgb_time:.5f} seconds")
    print(f"Grayscale Runtime: {gray_time:.5f} seconds")
    print(f"{MY_NAME} recommends {recommended_algorithm} algorithm")
