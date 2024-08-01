import torch
from dataLoader import BrainTumorDataset, train_transform, test_transform
from cnnModel import CNNModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from otherModels import vgg_model, resNet_model, mobileNet_model

#specifying the device such as CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#calling the BrainTumorDataset function for train and test 
train_dataset = BrainTumorDataset(root_dir=r'..\Brain-MRI-Dataset\Training', transform=train_transform)
test_dataset = BrainTumorDataset(root_dir=r'..\Brain-MRI-Dataset\Testing', transform=test_transform)

#intializing the dataLoader for train and test dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#Pushing the CNN model to the device which can be either CPU or GPU


def train_model(model, criterion, optimizer, num_epochs=10, save_path='trained_model.pth'):
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("Model training is started...")
    model.train()
    best_accuracy = 0.0  # Track best accuracy to save the best model
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch+1}/{num_epochs}')
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        
        # Save the model if it has the best accuracy
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), save_path)
    print(f"Training complete. Best accuracy: {best_accuracy:.4f}. Model saved to {save_path}")
    return model

#model evaluation

def test_model(model):
    print('Model test started...')
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    print(f"Accuracy on test set: {accuracy:.4f}")
    return true_labels, predictions

def dataVisualization(true_labels, predictions):
    #model prediction
    target_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    #true_labels, predictions = test_model(model)
    #performance evaluation
    print("Classification Report:")
    print(classification_report(true_labels, predictions, target_names=['glioma', 'meningioma', 'notumor', 'pituitary']))
    #confusion matrix 
    cm = confusion_matrix(true_labels, predictions)

    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['glioma', 'meningioma', 'notumor', 'pituitary'], yticklabels=['glioma', 'meningioma', 'notumor', 'pituitary'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

#model choosing
def choose_model():
    model=None
    print("Which model do you want to train?")
    print("1. CNN")
    print("2. ResNet50")
    print("3. VGG16")
    print("4. MobileNet_V3")
    
    choice = input("Enter the number corresponding to your choice: ")
    
    if choice == '1':
        print("You chose to train a CNN model.")
        save_path = 'Trained Model/cnn_trained_model.pth' #the path where the trained will save
        #model training
        model = CNNModel().to(device)
        criterion = nn.CrossEntropyLoss()  #loss function for the categorical classification
        optimizer = optim.Adam(model.parameters(), lr=0.001)   #definging the adam optimizer by setting some basic parameters
        model = train_model(model, criterion, optimizer, num_epochs=10, save_path=save_path)
    elif choice == '2':
        print("You chose to train a ResNet50 model.")
        save_path = 'Trained Model/RestNet50_trained_model.pth'
        # Move model to the device
        resnet50 = resNet_model()
        resnet50 = resnet50.to(device)
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(resnet50.fc.parameters(), lr=0.001)
        model = train_model(resnet50, criterion, optimizer, num_epochs=10, save_path=save_path)
    elif choice == '3':
        print("You chose to train a VGG16 model.")
        save_path = 'Trained Model/VGG16_trained_model.pth'
        # Move model to the device
        vgg16 = vgg_model()
        vgg16 = vgg16.to(device)
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(vgg16.classifier[6].parameters(), lr=0.001)
        model = train_model(vgg16, criterion, optimizer, num_epochs=10, save_path=save_path)
    elif choice == '4':
        print("You chose to train a MobileNet_V3 model.")
        save_path = 'Trained Model/MobileNetV3_trained_model.pth'
        # Move model to the device
        mobilenet_v3 = mobileNet_model()
        mobilenet_v3 = mobilenet_v3.to(device)
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(mobilenet_v3.classifier[3].parameters(), lr=0.001)
        model = train_model(mobilenet_v3, criterion, optimizer, num_epochs=10, save_path=save_path)
    else:
        print("Invalid choice. Please enter a number between 1 and 4.")
        choose_model()  # Ask again if the choice is invalid
    return model

def main():
    print(f'Device detected for training: {device}')
    model = choose_model() #select model and train
    #model prediction
    true_labels, predictions = test_model(model)
    # Data visualization using confusion materix 
    dataVisualization(true_labels=true_labels, predictions=predictions)

if __name__ == "__main__":
    main()