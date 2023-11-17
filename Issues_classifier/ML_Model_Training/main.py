import torch
import torch.nn as nn
from classifier import LogisticRegression
from ML_metrics import accuracy, precision, recall, f1_score
from data_loader import load_data
from tqdm import tqdm
from matplotlib import pyplot as plt

def plot_results(history):
    #Plot Loss vs Epochs
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('Loss_vs_Epochs.png')
    plt.clf()




def training_loop(model, epochs=1500):
    criterion = torch.nn.BCELoss()

    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': [], 'precision': [], 'val_precision': [], 'recall': [], 'val_recall': [], 'f1_score': [], 'val_f1_score': []}
    cross_validation_loaders, X_shape = load_data(n_splits=10, batch_size=32)
    model = LogisticRegression(X_shape)
    optimizer = torch.optim.Adam(model.parameters())
    #Loop 
    for epoch in tqdm(range(epochs)):
        for train_loader, test_loader in cross_validation_loaders:
            

            for inputs, labels in train_loader:
                loss = criterion(model(inputs), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                history["loss"].append(loss.item())
                history["acc"].append(accuracy(model(inputs), labels))
                history["precision"].append(precision(model(inputs), labels))
                history["recall"].append(recall(model(inputs), labels))
                history["f1_score"].append(f1_score(model(inputs), labels))

            for inputs, labels in test_loader:
                val_loss = criterion(model(inputs), labels)
                history["val_loss"].append(val_loss.item())
                history["val_acc"].append(accuracy(model(inputs), labels))
                history["val_precision"].append(precision(model(inputs), labels))
                history["val_recall"].append(recall(model(inputs), labels))
                history["val_f1_score"].append(f1_score(model(inputs), labels))
        
    #save the weights
    torch.save(model.state_dict(), 'Logistic_Regression_model_weights.pth')
    #plot the results
    plot_results(history)
    #avg the results
    for key in history:
        history[key] = sum(history[key])/len(history[key])
    return history


if __name__ == "__main__":
    model = LogisticRegression(100)
    history = training_loop(model)
    print(history)

    
