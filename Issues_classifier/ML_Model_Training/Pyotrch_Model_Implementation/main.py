import torch
import torch.nn as nn
from classifier import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from data_loader import load_data
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse

# def plot_results(history):
#     #Plot Loss vs Epochs
#     plt.plot(history['loss'], label='Training Loss')
#     plt.plot(history['val_loss'], label='Validation Loss')
#     plt.legend()
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.savefig('Loss_vs_Epochs.png')
#     plt.clf()




def training_loop(args: argparse.Namespace, epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.BCELoss()
    
    
    history = {'loss': [], 'val_loss': []}
    best_scores = {'loss': 100, 'val_loss': 100, 'acc': 0, 'val_acc': 0, 'precision': 0, 'val_precision': 0, 'recall': 0, 'val_recall': 0, 'f1_score': 0, 'val_f1_score': 0}
    cross_validation_loaders, X_shape = load_data(args, n_splits=5, batch_size=32)
    model = LogisticRegression(X_shape)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay = 0.01)
    #Loop 
    for epoch in tqdm(range(epochs)):
        for train_loader, test_loader in cross_validation_loaders:
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                history["loss"].append(loss.item())
                if loss.item() < best_scores['loss']:
                    best_scores['loss'] = loss.item()
                    cpu_output = torch.round(output).cpu().detach().numpy()
                    
                    cpu_labels = labels.cpu().detach().numpy()
                    best_scores['acc'] = accuracy_score(cpu_output, cpu_labels)
                    best_scores['precision'] = precision_score(cpu_output, cpu_labels)
                    best_scores['recall'] = recall_score(cpu_output, cpu_labels)
                    best_scores['f1_score'] = f1_score(cpu_output, cpu_labels)
                
            model.eval()
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad():
                    val_output = model(inputs)
                    val_loss = criterion(val_output, labels)
                    history["val_loss"].append(val_loss.item())

                if val_loss.item() < best_scores['val_loss']:
                    best_scores['val_loss'] = val_loss.item()
                    cpu_val_output = torch.round(val_output).cpu().detach().numpy()
                    cpu_labels = labels.cpu().detach().numpy()
                    best_scores['val_acc'] = accuracy_score(cpu_val_output, cpu_labels)
                    best_scores['val_precision'] = precision_score(cpu_val_output, cpu_labels)
                    best_scores['val_recall'] = recall_score(cpu_val_output, cpu_labels)
                    best_scores['val_f1_score'] = f1_score(cpu_val_output, cpu_labels)
                    which_model = args.Framework +" " + args.Training_Data
                    torch.save(model.state_dict(), 'Logistic_Regression_model'+which_model+'_weights.pth')
                    #Printing Best Results with Epoch Number
                    print('Best Results in epoch {epoch} is: ', best_scores)
                    
                    
    #plot the results
    # plot_results(history)
    
    return best_scores


if __name__ == "__main__":
    
    #argParse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--Training_Data", type=str, help="Training Data", choices=['Training_Data_with_Title_BERT', 'Training_Data_With_Title_AND_BODY_BERT'])
    parser.add_argument("--Framework", type=str, help="Framework", choices=['Pytorch', 'Tensorflow'])
    parser.add_argument("--DataPath", type=str, help="DataPath", default = "../Data/GT_bert_data_pytorch.csv")
    args = parser.parse_args()
    
    best_scores = training_loop(args,epochs = args.epochs)
    print(best_scores)
    file_name = args.Framework +" " + args.Training_Data + '.txt'
    with open(file_name, 'w') as f:
        print(best_scores, file=f)

    
