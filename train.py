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
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback

#specifying the device such as CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#calling the BrainTumorDataset function for train and test 
train_dataset = BrainTumorDataset(root_dir=r'..\Brain-MRI-Dataset\Training', transform=train_transform)
test_dataset = BrainTumorDataset(root_dir=r'..\Brain-MRI-Dataset\Testing', transform=test_transform)

#intializing the dataLoader for train and test dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=3, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=3, persistent_workers=True)

#Pushing the CNN model to the device which can be either CPU or GPU

class MyModel(pl.LightningModule):
    def __init__(self, model, criterion, optimizer_class, learning_rate=1e-3):
        super(MyModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        acc = (predicted == labels).float().mean()
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_acc', acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        acc = (predicted == labels).float().mean()
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.learning_rate)
        return optimizer
    
#customizing the callback class to update user after each epoch.
class BestMetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics['val_loss']
        val_acc = trainer.callback_metrics['val_acc']

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            trainer.save_checkpoint('best_val_loss_checkpoint.ckpt')

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            trainer.save_checkpoint('best_val_acc_checkpoint.ckpt')

        print(f"\nEpoch {trainer.current_epoch}: Best Val Loss: {self.best_val_loss:.4f}, Best Val Acc: {self.best_val_acc:.4f}")


# Callbacks
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, mode='min')
best_metric_callback = BestMetricsCallback()

learning_rate = 1e-3

#model evaluation

def test_model(model):
    print('Model test started...')
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    y_score = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            y_score.extend(outputs.cpu().numpy())

    accuracy = correct / total
    print(f"Accuracy on test set: {accuracy:.4f}")
    return true_labels, predictions, y_score


def dataVisualization(true_labels, predictions, y_score):
    target_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # Print classification report
    print("Classification Report:")
    print(classification_report(true_labels, predictions, target_names=target_names))

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    # Plot confusion matrix using matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=target_names,
        yticklabels=target_names,
        title='Confusion Matrix',
        ylabel='True Label',
        xlabel='Predicted Label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

    # Binarize the output
    y_true = label_binarize(true_labels, classes=[0, 1, 2, 3])
    n_classes = y_true.shape[1]

    # Convert y_score to numpy array
    y_score = np.array(y_score)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='{0} (area = {1:0.2f})'
                 ''.format(target_names[i], roc_auc[i]))

    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
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
        save_path = 'Trained Model/cnn_checkpoints' #the path where the trained will save
        #model training
        model = CNNModel().to(device)
        criterion = nn.CrossEntropyLoss()  #loss function for the categorical classification
        #optimizer = optim.Adam(model.parameters(), lr=0.001)   #definging the adam optimizer by setting some basic parameters
        optimizer = optim.Adam
        #initializing torch model
        lightning_model = MyModel(model, criterion, optimizer, learning_rate)
        #ensure that the directory is already present
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #Defining checkpoint callback 
        checkpoint_callback = ModelCheckpoint(dirpath=save_path, save_top_k=1, monitor='val_loss', mode='min')
        # Trainer
        trainer = pl.Trainer(max_epochs=50, callbacks=[early_stopping_callback, checkpoint_callback, best_metric_callback])
        # Train the model
        trainer.fit(lightning_model, train_loader, test_loader)
        #model = train_model(model, criterion, optimizer, num_epochs=10, save_path=save_path)
    elif choice == '2':
        print("You chose to train a ResNet50 model.")
        save_path = 'Trained Model/RestNet50_checkpoints'
        # Move model to the device
        resnet50 = resNet_model()
        resnet50 = resnet50.to(device)
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam
        #ensure that the directory is already present
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #Defining checkpoint callback 
        checkpoint_callback = ModelCheckpoint(dirpath=save_path, save_top_k=1, monitor='val_loss', mode='min')

        #initializing torch model
        lightning_model = MyModel(resnet50, criterion, optimizer, learning_rate)
        # Trainer
        trainer = pl.Trainer(max_epochs=50, callbacks=[early_stopping_callback, checkpoint_callback, best_metric_callback])
        # Train the model
        trainer.fit(lightning_model, train_loader, test_loader)
        #model = train_model(resnet50, criterion, optimizer, num_epochs=10, save_path=save_path)
    elif choice == '3':
        print("You chose to train a VGG16 model.")
        save_path = 'Trained Model/VGG16_Checkpoints'
        # Move model to the device
        vgg16 = vgg_model()
        vgg16 = vgg16.to(device)
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam #(vgg16.classifier[6].parameters(), lr=0.001)

        #ensure that the directory is already present
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #Defining checkpoint callback 
        checkpoint_callback = ModelCheckpoint(dirpath=save_path, save_top_k=1, monitor='val_loss', mode='min')

        #initializing torch model
        lightning_model = MyModel(resnet50, criterion, optimizer, learning_rate)
        # Trainer
        trainer = pl.Trainer(max_epochs=50, callbacks=[early_stopping_callback, checkpoint_callback, best_metric_callback])
        # Train the model
        trainer.fit(lightning_model, train_loader, test_loader)
        #model = train_model(vgg16, criterion, optimizer, num_epochs=10, save_path=save_path)
    elif choice == '4':
        print("You chose to train a MobileNet_V3 model.")
        save_path = 'Trained Model/MobileNetV3_Checkpoints'
        # Move model to the device
        mobilenet_v3 = mobileNet_model()
        mobilenet_v3 = mobilenet_v3.to(device)
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam #(mobilenet_v3.classifier[3].parameters(), lr=0.001)
        #model = train_model(mobilenet_v3, criterion, optimizer, num_epochs=10, save_path=save_path)
        #ensure that the directory is already present
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #Defining checkpoint callback 
        checkpoint_callback = ModelCheckpoint(dirpath=save_path, save_top_k=1, monitor='val_loss', mode='min')

        #initializing torch model
        lightning_model = MyModel(resnet50, criterion, optimizer, learning_rate)
        # Trainer
        trainer = pl.Trainer(max_epochs=50, callbacks=[early_stopping_callback, checkpoint_callback, best_metric_callback])
        # Train the model
        trainer.fit(lightning_model, train_loader, test_loader)
    else:
        print("Invalid choice. Please enter a number between 1 and 4.")
        choose_model()  # Ask again if the choice is invalid
    return model

def main():
    print(f'Device detected for training: {device}')
    model = choose_model() #select model and train
    #model prediction
    true_labels, predictions, y_score = test_model(model)
    # Data visualization using confusion materix 
    dataVisualization(true_labels=true_labels, predictions=predictions, y_score=y_score)

if __name__ == "__main__":
    main()