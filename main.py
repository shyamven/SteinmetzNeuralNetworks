import argparse
import sys
import numpy as np
import torch
import os

sys.path.append('./SteinmetzNeuralNetworks')
from utils.upload_data import LoadDataset
from models.train_regression import train_regression
from models.train_classification import train_classification
# to run file: python3 main.py --Dataset "DatasetName" (optional: --Shift "Shift", etc.)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simulations.')
    parser.add_argument('--Dataset', type=str, help='Dataset to upload', required=True)
    parser.add_argument('--Task', type=str, default='Classification', help='Task (Classification or Regression)')
    parser.add_argument('--Model', type=str, default='RVNN', help='Network type (CVNN, RVNN, Steinmetz, or Analytic)')
    parser.add_argument('--BatchSize', type=int, default=1000, help='Size of each training batch')
    parser.add_argument('--Epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--Iterations', type=int, default=5, help='Number of iterations to repeat task')
    parser.add_argument('--Noise', type=float, default=-1, help="Noise Scaling Factor", required=False)
    args = parser.parse_args()
    
    DatasetName = args.Dataset
    task = args.Task
    model = args.Model
    batch_size = args.BatchSize
    epochs = args.Epochs
    iterations = args.Iterations
    noise = args.Noise

    # This function calls the dataset we need for the experiment
    # Distinguish between regression and classification
    if task == 'Regression':
        X_train_real, X_train_imag, y_train, X_test_real, X_test_imag, y_test = LoadDataset(DatasetName)
        error_mag, error_phase = train_regression(X_train_real, X_train_imag, y_train, X_test_real, X_test_imag, y_test, task, model, iterations, epochs, batch_size, noise)
    elif task == 'Classification':
        X_train_real, X_train_imag, y_train, X_test_real, X_test_imag, y_test = LoadDataset(DatasetName)
        result_test = train_classification(X_train_real, X_train_imag, y_train, X_test_real, X_test_imag, y_test, task, model, iterations, epochs, batch_size, noise)
    else:
        print('Error: Task must be Regression or Classification')
        exit(1) 
    
    # Path for the directory
    results_dir = 'Results'
    
    # Check if the directory exists, and if not, create it
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    
    if task == 'Regression':
        # Path for the CSV files
        test_losses_path = f'Results/MAG_{DatasetName}_{task}_Model={model}_BatchSize={batch_size}_Epochs={epochs}_Iterations={iterations}_Noise={noise}.csv'
        np.savetxt(test_losses_path, error_mag, delimiter=',')
        test_losses_path = f'Results/PHASE_{DatasetName}_{task}_Model={model}_BatchSize={batch_size}_Epochs={epochs}_Iterations={iterations}_Noise={noise}.csv'
        np.savetxt(test_losses_path, error_phase, delimiter=',')
        
    elif task == 'Classification':
        # Path for the CSV files
        test_losses_path = f'Results/TEST_{DatasetName}_{task}_Model={model}_BatchSize={batch_size}_Epochs={epochs}_Iterations={iterations}_Noise={noise}.csv'
        np.savetxt(test_losses_path, result_test, delimiter=',')
