# Heart Disease Classification with MLP and Decision Tree Analysis

This project implements a hybrid approach to heart disease classification using a Multi-Layer Perceptron (MLP) neural network and an interpretable decision tree. It combines the power of deep learning with the interpretability of decision trees, enhanced by SHAP (SHapley Additive exPlanations) values for feature importance analysis.

## Features

- MLP neural network with improved architecture
- Decision tree implementation with entropy-based splitting
- SHAP analysis for model interpretation
- Model consistency analysis
- Comprehensive visualization tools
- Performance evaluation metrics

## Dataset URL
> https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset?resource=download&select=heart.csv
> This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.

## Requirements

```bash
numpy
pandas
torch
scikit-learn
shap
seaborn
matplotlib
```

## Project Structure

- `mlp_shap.py`: Neural network implementation and SHAP analysis
- `main.py`: Main coordination script
- `tree.py`: Decision tree implementation
- `tree_evaluation.py`: Tree evaluation metrics
- `model_analysis.py`: Comparative analysis tools

## Installation

Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the MLP model:
```bash
python mlp_shap.py
```
This will train the neural network and save the model weights to `best_model.pth`.

2. Run the main analysis:
```bash
python main.py
```
This performs SHAP analysis and builds an interpretable decision tree.

3. Run additional model analysis (optional):
```bash
python model_analysis.py
```
This provides detailed comparison between MLP and decision tree models.

## Model Architecture

### MLP Neural Network
- Input layer: 13 features
- Hidden layers: [64, 32, 16]
- Output layer: 1 (binary classification)
- Dropout rate: 0.3
- Batch normalization
- ReLU activation

### Decision Tree
- Entropy-based splitting
- Dynamic feature selection based on SHAP values
- Configurable depth and minimum samples for splitting
- Sample counting at each node

## Analysis Features

1. **SHAP Analysis**
   - Feature importance ranking
   - Global model interpretation
   - Individual prediction explanations

2. **Model Comparison**
   - MLP vs Decision Tree predictions
   - Feature importance comparison
   - Prediction consistency analysis

3. **Visualization**
   - Decision tree structure
   - Feature importance plots
   - Confusion matrices
   - Performance metrics

## Performance Metrics

The system evaluates both models using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

## Customization

You can customize various parameters:

1. MLP Architecture:
```python
hidden_sizes = [64, 32, 16]  # in mlp_shap.py
dropout_rate = 0.3
```

2. Decision Tree:
```python
max_depth = 4  # in main.py
min_samples_split = 5
```

3. Training Parameters:
```python
batch_size = 32  # in mlp_shap.py
epochs = 300
learning_rate = 0.001
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source.
