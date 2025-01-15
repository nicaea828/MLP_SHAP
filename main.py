import torch
import numpy as np
from mlp_shap import DataPreprocessor, ImprovedHeartDiseaseClassifier, ShapAnalyzer
from tree import TreeNode, build_tree
from tree_evaluation import TreeEvaluator, evaluate_tree_performance

class TreeVisualizer:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.tree = None
        
    def visualize_tree(self, node, depth=0, prefix="Root"):
        # 可视化决策树结构
        if node is None:
            return
            
        indent = "  " * depth
        
        if node.value is not None:
            print(f"{indent}{prefix}: Predict class {node.value} (samples={node.samples})")
            return
            
        feature_name = self.feature_names[node.feature_index]
        print(f"{indent}{prefix}: if {feature_name} <= {node.threshold:.2f} "
              f"(samples={node.samples})")
        
        self.visualize_tree(node.left, depth + 1, "Left")
        self.visualize_tree(node.right, depth + 1, "Right")
    
    def explain_prediction(self, node, sample):
        # 解释单个样本的预测路径
        path = []
        current_node = node
        
        while current_node is not None and current_node.value is None:
            # 确保当前节点是内部节点
            if current_node.feature_index is None or current_node.threshold is None:
                break
                
            feature_name = self.feature_names[current_node.feature_index]
            feature_value = sample[current_node.feature_index]
            
            path.append(f"{feature_name} = {feature_value:.2f}")
            
            if feature_value <= current_node.threshold:
                path[-1] += f" ≤ {current_node.threshold:.2f}"
                current_node = current_node.left
            else:
                path[-1] += f" > {current_node.threshold:.2f}"
                current_node = current_node.right
        
        # 检查是否到达有效的叶子节点
        if current_node is not None and current_node.value is not None:
            prediction = current_node.value
            path_str = " → ".join(path) + f" → Class {prediction}"
        else:
            prediction = None
            path_str = " → ".join(path) + " → Invalid path"
            
        return path_str, prediction

def main():
    # 1. 加载和预处理数据
    preprocessor = DataPreprocessor()
    X_tensor, y_tensor, df = preprocessor.preprocess('heart.csv')
    feature_names = df.columns[:13].tolist()

    # 2. 加载训练好的MLP模型
    input_size = X_tensor.shape[1]
    hidden_sizes = [64, 32, 16]
    output_size = 1
    
    mlp_model = ImprovedHeartDiseaseClassifier(input_size, hidden_sizes, output_size)
    mlp_model.load_state_dict(torch.load('best_model.pth'))
    mlp_model.eval()

    # 3. SHAP分析
    print("\nPerforming SHAP Analysis...")
    analyzer = ShapAnalyzer(mlp_model, X_tensor, feature_names)
    shap_values = analyzer.explain_predictions(background_size=100)
    
    # 4. 计算特征重要性排序
    feature_importance = np.mean(np.abs(shap_values), axis=0).flatten()
    
    # 创建特征名称和重要性的列表
    features_and_importance = list(zip(feature_names, feature_importance))
    # 按重要性排序
    features_and_importance.sort(key=lambda x: x[1], reverse=True)
    
    # 获取排序后的特征顺序
    feature_order = [feature_names.index(feat) for feat, _ in features_and_importance]
    
    print("\nFeature Importance Rankings (SHAP):")
    for feature_name, importance in features_and_importance:
        print(f"{feature_name}: {importance:.4f}")
    
    # 5. 构建决策树
    print("\nBuilding decision tree based on SHAP importance order...")
    X_numpy = X_tensor.numpy()
    y_numpy = y_tensor.numpy().astype(int)
    
    tree = build_tree(X_numpy, y_numpy, feature_order, max_depth=4, min_samples_split=5)
    
    # 6. 可视化决策树
    visualizer = TreeVisualizer(feature_names)
    print("\nDecision Tree Structure:")
    visualizer.visualize_tree(tree)
    
    # 7. 评估树的性能
    # 划分数据集
    train_size = int(0.8 * len(X_numpy))
    X_train, X_test = X_numpy[:train_size], X_numpy[train_size:]
    y_train, y_test = y_numpy[:train_size], y_numpy[train_size:]

    # 评估树的性能
    train_metrics, test_metrics = evaluate_tree_performance(tree, X_train, y_train, X_test, y_test)

    # 8. 示例：解释第一个样本的预测
    first_sample = X_numpy[0]
    explanation, prediction = visualizer.explain_prediction(tree, first_sample)
    print(f"\nExample Prediction Explanation:")
    print(explanation)
    if prediction is not None:
        print(f"Prediction: Class {prediction}")
        print(f"Actual class: {int(y_tensor[0].item())}")
    else:
        print("Could not make a prediction")

if __name__ == "__main__":
    main()
