# Steinmetz Neural Networks

This code implements the **Steinmetz Neural Network** architecture, and provides a framework for running simulations using Steinmetz and analytic neural networks on specified datasets for two primary tasks: complex-valued classification and complex-valued regression. This framework integrates the comparison of Steinmetz and analytic neural networks versus real-valued and complex-valued neural networks across the aforementioned tasks. We also provide jupyter notebooks for complex-valued regression on the RASPNet and Complex-Valued FSDD datasets. The user can customize several parameters, including the dataset to use, task type, model, and various training parameters. The script saves the train and test losses to CSV files in their respective directories once execution is complete.

## Requirements

- Python 3.x
- Libraries:
  - numpy
  - torch
  - torchvision
  - complexPyTorch
  - argparse
  - os

## Usage

To run the script, use the following command:

```
python3 main.py --Dataset "DatasetName" (optional arguments)
```


## Arguments

1. `--Dataset` (Required): Name of the dataset to be uploaded for the experiment.
2. `--Task` (Default: `Classification`): Specifies the task. It can be either `Classification` or `Regression`.
3. `--Model` (Default: `RVNN`): Type of neural network model to use. Options include `CVNN`, `RVNN`, `Steinmetz`, or `Analytic`.
4. `--BatchSize` (Default: `1000`): Size of each training batch.
5. `--Epochs` (Default: `25`): Number of training epochs.
6. `--Iterations` (Default: `5`): Number of iterations to repeat task.
7. `--Noise`  (Default: `-1`): Noise Scaling Factor. Not required by default.

## Output

After the simulation completes, the script saves the results in the specified format in a results directory based on the dataset, task, and model used.

## Errors

1. If the specified task is not recognized, the script will output an error and exit.
2. If the model specified is not recognized, the script will output an error and exit.

## Note

Before running the script, ensure that the required libraries are installed and that the path to the neural network models is correctly set.


## Citation: 

Plain Text:
```
Shyam Venkatasubramanian, Ali Pezeshki, Vahid Tarokh. Steinmetz Neural Networks for Complex-Valued Data. The 28th International Conference on Artificial Intelligence and Statistics.
```
BibTeX:
```
@inproceedings{venkatasubramanian2025steinmetz,
  title={Steinmetz Neural Networks for Complex-Valued Data},
  author={Venkatasubramanian, Shyam and Pezeshki, Ali and Tarokh, Vahid},
  booktitle={The 28th International Conference on Artificial Intelligence and Statistics}
}
```
