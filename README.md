# ClassicalML

# PCA Codebase Readme

This codebase provides a Python implementation for applying Principal Component Analysis (PCA) on a dataset, followed by the use of Synthetic Minority Over-sampling Technique (SMOTE) for data balancing and Support Vector Machine (SVM) classification for predictive modeling. The goal is to demonstrate how to preprocess data using PCA, address class imbalance using SMOTE, and perform binary classification using SVM. Additionally, the code includes steps to evaluate the performance of the SVM model.

## Getting Started

To use this codebase, follow these steps:

1. Clone the repository to your local machine or server:

```bash
git clone <repository_url>
cd <repository_directory>
```

2. Install the required Python packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

3. Ensure you have a dataset named `data.csv` located in the `data/` directory. The dataset should contain the necessary features and labels for the classification task.

## Code Overview

The main functionalities of the codebase are as follows:

1. **Principal Component Analysis (PCA)**:
   - The `pca.py` module contains the implementation of PCA using the `getPCA` class. You can specify the number of components as a parameter when initializing the PCA object.
   - The `apply_RobustScaler` method applies robust scaling to the input data.
   - The `apply_pca` method performs PCA transformation on the scaled data.

2. **Data Preprocessing and Transformation**:
   - The input data is read from `data/data.csv` using the Pandas library.
   - Categorical columns are converted into binary class using one-hot encoding.
   - The PCA transformation is applied to the data, and the resulting dataset is stored in `pca_data/pca_data.csv` for reuse.

3. **Data Balancing using SMOTE**:
   - The SMOTE module (`smote.py`) contains functions to apply under-sampling and over-sampling using SMOTE. The `appy_undersample` and `apply_smote` functions are used to balance the dataset.

4. **Support Vector Machine (SVM) Classification**:
   - The SVM module (`SVM.py`) contains the `getSVM` class, which is used to initialize, train, and evaluate an SVM classifier.
   - The SVM classifier is trained on the balanced dataset using the specified kernel and hyperparameters.
   - The `get_performance_metrix` method computes and returns the confusion matrix, average precision, and classification report of the SVM model.

## Usage

1. Prepare your dataset as `data.csv` in the `data/` directory.
2. Run the provided code in your Python environment.
3. The PCA-transformed data will be saved to `pca_data/pca_data.csv`.
4. The SVM model will be trained, and its performance metrics will be printed.

## Note

- Ensure that you have the required Python packages installed by running `pip install -r requirements.txt`.
- The code assumes that the dataset is appropriately preprocessed and contains the required columns for the tasks.

## Credits

This codebase was developed by [Your Name] and can be found at [GitHub Repository URL].

For any questions or issues, please contact [Your Email Address].

---
Replace placeholders like `[Your Name]`, `[GitHub Repository URL]`, and `[Your Email Address]` with appropriate values.
