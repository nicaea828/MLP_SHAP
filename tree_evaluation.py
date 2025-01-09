import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class TreeEvaluator:
    def __init__(self, tree):
        self.tree = tree
        
    def predict_single(self, sample):
        """预测单个样本"""
        current_node = self.tree
        
        while current_node.value is None:
            if current_node.feature_index is None or current_node.threshold is None:
                return 0  # 默认返回0类
            
            feature_value = sample[current_node.feature_index]
            if feature_value <= current_node.threshold:
                current_node = current_node.left
            else:
                current_node = current_node.right
                
        return current_node.value
    
    def predict(self, X):
        """预测多个样本"""
        return np.array([self.predict_single(sample) for sample in X])
    
    def evaluate(self, X, y_true):
        """评估模型性能"""
        y_pred = self.predict(X)
        
        # 计算各种指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, cm):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Decision Tree Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

def evaluate_tree_performance(tree, X_train, y_train, X_test, y_test):
    """评估决策树在训练集和测试集上的性能"""
    evaluator = TreeEvaluator(tree)
    
    # 评估训练集性能
    print("\nTraining Set Performance:")
    train_metrics = evaluator.evaluate(X_train, y_train)
    print(f"Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Precision: {train_metrics['precision']:.4f}")
    print(f"Recall: {train_metrics['recall']:.4f}")
    print(f"F1-score: {train_metrics['f1']:.4f}")
    
    print("\nTraining Set Confusion Matrix:")
    evaluator.plot_confusion_matrix(train_metrics['confusion_matrix'])
    
    # 评估测试集性能
    print("\nTest Set Performance:")
    test_metrics = evaluator.evaluate(X_test, y_test)
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-score: {test_metrics['f1']:.4f}")
    
    print("\nTest Set Confusion Matrix:")
    evaluator.plot_confusion_matrix(test_metrics['confusion_matrix'])
    
    return train_metrics, test_metrics
