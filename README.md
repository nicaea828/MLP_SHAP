# MLP_SHAP
## 1. First, train the model
python mlp_shap.py

## 2. After the model is trained and saved, run the main analysis
python main.py

## 3. Finally, run the additional model analysis (optional)
python model_analysis.py

# overview
The codebase consists of several Python files that implement a heart disease classification system using both an MLP (Multi-Layer Perceptron) neural network and a decision tree. Here's a overview of what each file does:
mlp_shap.py:

Implements the neural network model (ImprovedHeartDiseaseClassifier)
Handles data preprocessing
Contains the training loop
Implements SHAP analysis functionality


main.py:

Coordinates the overall analysis
Loads the trained model
Performs SHAP analysis
Builds and visualizes a decision tree
Compares predictions


tree.py:

Implements the decision tree algorithm
Uses entropy for split decisions
Includes sample counting and depth tracking


tree_evaluation.py:

Provides evaluation metrics for the decision tree
Implements prediction functionality
Creates confusion matrices and performance plots

# Dataset URL
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset?resource=download&select=heart.csv
model_analysis.py:

Compares SHAP vs tree feature importance
Analyzes model consistency
Examines feature patterns
