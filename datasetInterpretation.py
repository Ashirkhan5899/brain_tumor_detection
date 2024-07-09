import os
import matplotlib.pyplot as plt
import seaborn as sns
from dataLoader import BrainTumorDataset, train_transform, test_transform


train_dataset = BrainTumorDataset(root_dir=r'..\Brain-MRI-Dataset\Training', transform=train_transform)
test_dataset = BrainTumorDataset(root_dir=r'..\Brain-MRI-Dataset\Testing', transform=test_transform)

# Ensure the number of threads for OpenMP is set to 1
os.environ["OMP_NUM_THREADS"] = "1"

def data_distribution_over_each_class(train_dataset):
    #printing the size of each class.
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')
    print(f'Instance of the dataset: {train_dataset[0][0].shape}')
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

#call the function
data_distribution_over_each_class(train_dataset=train_dataset)