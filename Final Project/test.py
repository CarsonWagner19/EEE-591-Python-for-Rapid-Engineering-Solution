import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

################################################
# Setting up constants for training
################################################
# NOTE: Modify these values (especially EPOCHS) to optimize your results
BATCH_SIZE  = 64    # Recommended batch size for speed and stability
NUM_CLASSES = 2     # Cat (0) and Dog (1)
EPOCHS      = 10    # Start with a moderate number; tune this for better accuracy/runtime
LEARNING_RATE = 0.001
ROOT_DIR = '~/CIFAR10_data'
# YOUR NAME HERE:
MY_NAME = "Your Name" # <--- **MUST BE REPLACED WITH YOUR NAME**

################################################
# Custom Dataset Filtering (Step 1B)
################################################
# CIFAR10 Labels: 3 is cat, 5 is dog
# We map them to 0 (cat) and 1 (dog) for binary classification
class FilteredCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        # Load the base CIFAR10 dataset
        base_dataset = datasets.CIFAR10(root=root, train=train, 
                                        download=download, transform=transform)
        
        # Filter for Cat (3) and Dog (5)
        cat_dog_indices = [i for i, label in enumerate(base_dataset.targets) 
                           if label == 3 or label == 5]
        
        # Filter data and targets
        self.data = base_dataset.data[cat_dog_indices]
        self.targets = np.array(base_dataset.targets)[cat_dog_indices]
        
        # Remap targets: 3 -> 0 (Cat), 5 -> 1 (Dog)
        self.targets[self.targets == 3] = 0
        self.targets[self.targets == 5] = 1
        
        self.targets = self.targets.tolist()
        self.transform = transform
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        
        if self.transform is not None:
            # Need to convert numpy array to PIL image first for transforms to work
            from PIL import Image
            img = Image.fromarray(img)
            img = self.transform(img)
            
        return img, target

################################################
# Create the network (Step 2)
################################################
# Updated for 32x32 CIFAR10 images and dynamic input channels
class CIFAR_NET( nn.Module ):

    def __init__( self, num_classes, input_channels): # input_channels: 3 for RGB, 1 for Grayscale
        super().__init__()
        # Conv1: Input size (32x32). Input channel is dynamic (1 or 3)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(3,3))
        # Conv2: Input size (30x30) -> Output size (28x28)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
        # MaxPool: Input size (28x28) -> Output size (14x14) (Step 2B fix)
        self.mpool = nn.MaxPool2d( kernel_size=2 )
        self.drop1 = nn.Dropout( p=0.25 )
        self.flat  = nn.Flatten()
        
        # FC1: Input features must be 64 channels * 14 * 14 = 12544 (Step 2B fix)
        self.fc1   = nn.Linear( in_features=64 * 14 * 14, out_features=128 ) 
        self.drop2 = nn.Dropout( p=0.5 )
        # FC2: Output must be NUM_CLASSES (2: Cat/Dog)
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
# Main Training and Testing Function (Step 3)
################################################

def train_and_test(is_grayscale):
    
    # --- 1. Define Transformations ---
    if is_grayscale:
        # Grayscale Transformation: 3-channel input -> 1-channel output (Hint: This is your definition)
        # Normalization uses 1 value for mean and std
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1), # <--- GRYSCALE CONVERSION
            transforms.Normalize([0.5], [0.5])
        ])
        input_channels = 1
        model_name = "Grayscale"
    else:
        # RGB Transformation: 3-channel input
        # Normalization uses 3 values for mean and std
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        input_channels = 3
        model_name = "RGB"

    # --- 2. Load and Filter Data ---
    # The FilteredCIFAR10 class handles filtering and label remapping
    trainset = FilteredCIFAR10(ROOT_DIR, train=True, download=True, transform=data_transforms)
    testset  = FilteredCIFAR10(ROOT_DIR, train=False, download=True, transform=data_transforms)
    
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. Initialize Model and Timer ---
    start_time = time.time() # Start timer here to include data loading, training, and testing
    
    # Initialize the model with the correct number of input channels
    model = CIFAR_NET(NUM_CLASSES, input_channels) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Using Adam, tune for best result

    # --- 4. Training Loop ---
    model.train()
    num_batches = len(trainloader)

    print(f"\n--- Starting {model_name} Training (Epochs: {EPOCHS}) ---")
    for epoch in range(EPOCHS):
        for batch_idx, (images, labels) in enumerate(trainloader):
            
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            # Optional: Print progress less often to speed up execution
            if (epoch == EPOCHS-1) and (batch_idx % (num_batches // 5) == 0):
                 print(f"Epoch {epoch+1}/{EPOCHS}\tBatch {batch_idx}/{num_batches}\tLoss: {loss.mean().item():.4f}")

    # --- 5. Testing Loop ---
    model.eval()
    num_correct = 0
    num_attempts = 0

    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            guesses = torch.argmax(outputs, 1)

            num_guess = len(guesses)
            num_right = torch.sum(labels == guesses).item()

            num_correct += num_right
            num_attempts += num_guess
    
    # --- 6. Final Calculation ---
    end_time = time.time()
    runtime = end_time - start_time
    accuracy = 100 * num_correct / num_attempts

    print(f"--- Finished {model_name}. Accuracy: {accuracy:.2f}% | Runtime: {runtime:.2f} seconds ---\n")
    return accuracy, runtime

################################################
# Main Execution
################################################
if __name__ == '__main__':
    
    # Run RGB Algorithm
    rgb_acc, rgb_time = train_and_test(is_grayscale=False)
    
    # Run Grayscale Algorithm
    gray_acc, gray_time = train_and_test(is_grayscale=True)
    
    # --- Determine Recommendation ---
    # Recommendation logic: e.g., if RGB is more than 2% better, recommend RGB, 
    # otherwise, if grayscale is faster, recommend grayscale.
    
    if rgb_acc >= gray_acc:
        if (rgb_acc - gray_acc) >= 2.0:
            recommended_algorithm = "RGB" # Higher accuracy difference is worth the speed hit
        elif rgb_time < gray_time:
            recommended_algorithm = "RGB" # Similar accuracy, RGB is faster
        else:
            recommended_algorithm = "Grayscale" # Similar accuracy, Grayscale is significantly faster
    else: # Grayscale had higher accuracy
         recommended_algorithm = "Grayscale"
    
    
    # --- Print Final Output (Required Format) ---
    print("--- Final Results ---")
    print(f"RGB Accuracy: {rgb_acc:.1f}%")
    print(f"Grayscale Accuracy: {gray_acc:.1f}%")
    print(f"RGB Runtime: {rgb_time:.5f} seconds")
    print(f"Grayscale Runtime: {gray_time:.5f} seconds")
    print(f"{MY_NAME} recommends {recommended_algorithm} algorithm")