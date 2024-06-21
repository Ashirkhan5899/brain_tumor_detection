import torch
from cnnModel import train_loader, test_loader, device
from dataLoader import train_dataset, test_dataset
#from torchvision import models
from cnnModel import CNNModel
import torch.nn as nn
import torch.optim as optim
import os

model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

save_path = 'Trained Model/trained_model.pth'

# Ensure the directory exists before saving
os.makedirs(os.path.dirname(save_path), exist_ok=True)

def train_model(model, criterion, optimizer, num_epochs=10, save_path='trained_model.pth'):
    model.train()
    best_accuracy = 0.0  # Track best accuracy to save the best model
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
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

#model evaluation

def test_model(model):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
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
#model training
train_model(model, criterion, optimizer, num_epochs=10, save_path=save_path)