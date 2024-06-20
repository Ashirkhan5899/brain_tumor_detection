import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.image_paths = []
        self.labels = []
        
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
#create dataset instances
train_dataset = BrainTumorDataset(root_dir=r'..\Brain-MRI-Dataset\Training', transform=train_transform)
test_dataset = BrainTumorDataset(root_dir=r'..\Brain-MRI-Dataset\Testing', transform=test_transform)

print(f'Train dataset size: {len(train_dataset)}')
print(f'Test dataset size: {len(test_dataset)}')
print(f'Instance of the dataset: {train_dataset[0][0].shape}')

# Ensure the number of threads for OpenMP is set to 1
os.environ["OMP_NUM_THREADS"] = "1"

def data_distribution_over_each_class(train_dataset):
    # Verify the train_dataset has the necessary attributes
    if not hasattr(train_dataset, 'classes') or not hasattr(train_dataset, '__getitem__'):
        raise ValueError("train_dataset must have 'classes' and '__getitem__' attributes.")

    # Initialize class counts
    class_counts = [0] * len(train_dataset.classes)

    # Count the number of instances for each class
    for _, label in train_dataset:
        class_counts[label] += 1

    # Use Seaborn's style settings
    sns.set(style="whitegrid")

    # Plot the data distribution using a pie chart
    plt.figure(figsize=(8, 6))
    plt.gca().set_facecolor('none')  # Set the background color to transparent

    # Plot using Seaborn color palette
    colors = sns.color_palette('pastel')
    plt.pie(class_counts, labels=train_dataset.classes, autopct='%1.1f%%', colors=colors)
    plt.title('Data Distribution over Each Class')
    plt.show()

# Example usage (assuming you have a train_dataset variable):

data_distribution_over_each_class(train_dataset)
