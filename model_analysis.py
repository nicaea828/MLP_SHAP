import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import seaborn as sns
from mlp_shap import DataPreprocessor, ImprovedHeartDiseaseClassifier, ShapAnalyzer
from tree import TreeNode, build_tree
from tree_evaluation import TreeEvaluator
import shap

class ModelAnalyzer:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def analyze_tree_feature_usage(self, tree_node, feature_usage=None):
        """统计决策树中每个特征被使用的次数"""
        if feature_usage is None:
            feature_usage = defaultdict(int)
        
        # 如果是叶节点，返回
        if tree_node.value is not None:
            return feature_usage
        
        # 统计当前节点的特征使用
        feature_usage[tree_node.feature_index] += 1
        
        # 递归统计左右子树
        if tree_node.left:
            self.analyze_tree_feature_usage(tree_node.left, feature_usage)
        if tree_node.right:
            self.analyze_tree_feature_usage(tree_node.right, feature_usage)
            
        return feature_usage

    def compare_shap_vs_tree(self, shap_values, tree):
        """比较SHAP重要性和决策树特征使用频率"""
        # 计算SHAP重要性，确保转换为普通Python数值
        shap_importance = np.mean(np.abs(shap_values), axis=0).flatten()
        if isinstance(shap_importance, np.ndarray):
            shap_importance = shap_importance.tolist()
        
        # 获取树中特征使用频率
        feature_usage = self.analyze_tree_feature_usage(tree)
        
        # 创建对比数据
        comparison_data = []
        for i, feature in enumerate(self.feature_names):
            comparison_data.append({
                'feature': feature,
                'shap_importance': float(shap_importance[i]),
                'tree_usage': float(feature_usage.get(i, 0))
            })
        
        # 转换为DataFrame并排序
        df = pd.DataFrame(comparison_data)
        df_sorted = df.sort_values('shap_importance', ascending=False)
        
        # 绘制对比图
        plt.figure(figsize=(15, 6))
        
        # SHAP重要性
        plt.subplot(1, 2, 1)
        plt.bar(range(len(df_sorted)), df_sorted['shap_importance'].values)
        plt.xticks(range(len(df_sorted)), df_sorted['feature'], rotation=45, ha='right')
        plt.title('SHAP Feature Importance')
        
        # 树使用频率
        plt.subplot(1, 2, 2)
        plt.bar(range(len(df_sorted)), df_sorted['tree_usage'].values)
        plt.xticks(range(len(df_sorted)), df_sorted['feature'], rotation=45, ha='right')
        plt.title('Tree Feature Usage')
        
        plt.tight_layout()
        plt.savefig('shap_vs_tree_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return df_sorted

    def analyze_model_consistency(self, mlp_model, tree_evaluator, X):
        """分析神经网络和决策树的预测一致性"""
        # 获取两个模型的预测
        mlp_pred = mlp_model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
        mlp_pred = (mlp_pred > 0.5).astype(int)
        tree_pred = tree_evaluator.predict(X)
        
        # 找出预测不一致的样本
        disagreements = X[mlp_pred.flatten() != tree_pred]
        agreement_rate = 1 - len(disagreements) / len(X)
        
        # 分析不一致样本的特征分布
        if len(disagreements) > 0:
            feature_stats = pd.DataFrame(disagreements, columns=self.feature_names)
            feature_stats = feature_stats.describe()
            
            # 绘制不一致样本的特征分布
            plt.figure(figsize=(15, 6))
            plt.boxplot([disagreements[:, i] for i in range(disagreements.shape[1])],
                    labels=self.feature_names)
            plt.xticks(rotation=45, ha='right')
            plt.title('Feature Distribution in Disagreement Cases')
            plt.tight_layout()
            plt.savefig('disagreement_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 绘制特征相关性热图
            plt.figure(figsize=(12, 10))
            correlation_matrix = pd.DataFrame(disagreements, columns=self.feature_names).corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlation in Disagreement Cases')
            plt.tight_layout()
            plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return {
            'agreement_rate': agreement_rate,
            'disagreement_samples': len(disagreements),
            'feature_stats': feature_stats if len(disagreements) > 0 else None
        }

    def analyze_feature_patterns(self, mlp_model, tree_evaluator, X):
        """分析特征值与模型预测差异的关系，并在一张图中显示所有特征"""
        mlp_pred = mlp_model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
        mlp_pred = (mlp_pred > 0.5).astype(int)
        tree_pred = tree_evaluator.predict(X)
        
        # 绘制所有特征的一致性变化
        plt.figure(figsize=(15, 8))
        
        results = []
        for i, feature in enumerate(self.feature_names):
            # 将特征值分成10个区间
            bins = np.percentile(X[:, i], np.linspace(0, 100, 11))
            agreement_rates = []
            percentiles = []
            
            for j in range(len(bins)-1):
                mask = (X[:, i] >= bins[j]) & (X[:, i] < bins[j+1])
                if np.any(mask):
                    agreement = np.mean(mlp_pred[mask].flatten() == tree_pred[mask])
                    agreement_rates.append(agreement)
                    percentiles.append(np.linspace(0, 100, len(agreement_rates))[-1])
            
            # 绘制特征值与一致性的关系
            plt.plot(percentiles, agreement_rates, label=feature, marker='o', markersize=4)
            
            results.append({
                'feature': feature,
                'min_agreement': min(agreement_rates),
                'max_agreement': max(agreement_rates),
                'avg_agreement': np.mean(agreement_rates)
            })
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Feature Percentile')
        plt.ylabel('Agreement Rate')
        plt.title('Agreement Rate vs Feature Percentile for All Features')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('feature_agreement_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 为每个特征单独绘制图表并保存
        for i, feature in enumerate(self.feature_names):
            plt.figure(figsize=(8, 4))
            bins = np.percentile(X[:, i], np.linspace(0, 100, 11))
            agreement_rates = []
            percentiles = []
            
            for j in range(len(bins)-1):
                mask = (X[:, i] >= bins[j]) & (X[:, i] < bins[j+1])
                if np.any(mask):
                    agreement = np.mean(mlp_pred[mask].flatten() == tree_pred[mask])
                    agreement_rates.append(agreement)
                    percentiles.append(np.linspace(0, 100, len(agreement_rates))[-1])
            
            plt.plot(percentiles, agreement_rates, marker='o', markersize=4)
            plt.grid(True, alpha=0.3)
            plt.xlabel('Feature Percentile')
            plt.ylabel('Agreement Rate')
            plt.title(f'Agreement Rate vs {feature} Percentile')
            plt.tight_layout()
            plt.savefig(f'feature_agreement_{feature}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return pd.DataFrame(results)


def main():
    # 数据预处理
    preprocessor = DataPreprocessor()
    X_tensor, y_tensor, df = preprocessor.preprocess('heart.csv')
    feature_names = df.columns[:13].tolist()

    # 加载MLP模型
    input_size = X_tensor.shape[1]
    hidden_sizes = [64, 32, 16]
    output_size = 1
    
    mlp_model = ImprovedHeartDiseaseClassifier(input_size, hidden_sizes, output_size)
    mlp_model.load_state_dict(torch.load('best_model.pth'))
    mlp_model.eval()

    # SHAP分析
    print("\nPerforming SHAP Analysis...")
    analyzer = ShapAnalyzer(mlp_model, X_tensor, feature_names)
    shap_values = analyzer.explain_predictions(background_size=100)
    
    # 构建决策树
    X_numpy = X_tensor.numpy()
    y_numpy = y_tensor.numpy().astype(int)
    
    # 计算特征重要性排序
    feature_importance = np.mean(np.abs(shap_values), axis=0).flatten()
    feature_order = np.argsort(-feature_importance)
    
    # 构建决策树
    tree = build_tree(X_numpy, y_numpy, feature_order, max_depth=4, min_samples_split=5)
    
    # 初始化分析器
    model_analyzer = ModelAnalyzer(feature_names)
    
    # 1. 比较SHAP和树的特征使用
    print("\nAnalyzing SHAP vs Tree relationship...")
    shap_tree_comparison = model_analyzer.compare_shap_vs_tree(shap_values, tree)
    print("\nFeature importance comparison:")
    print(shap_tree_comparison)
    
    # 2. 分析模型一致性
    print("\nAnalyzing model consistency...")
    tree_evaluator = TreeEvaluator(tree)
    consistency_results = model_analyzer.analyze_model_consistency(mlp_model, tree_evaluator, X_numpy)
    
    print(f"\nModel agreement rate: {consistency_results['agreement_rate']:.4f}")
    print(f"Number of disagreement samples: {consistency_results['disagreement_samples']}")
    
    if consistency_results['feature_stats'] is not None:
        print("\nFeature statistics in disagreement cases:")
        print(consistency_results['feature_stats'])
    
    # 3. 分析特征模式
    print("\nAnalyzing feature patterns...")
    feature_patterns = model_analyzer.analyze_feature_patterns(mlp_model, tree_evaluator, X_numpy)
    print("\nFeature pattern analysis:")
    print(feature_patterns)

if __name__ == "__main__":
    main()
