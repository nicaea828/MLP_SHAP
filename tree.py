import numpy as np

class TreeNode:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        self.samples = 0  # 添加样本数量统计
        self.depth = 0    # 添加深度信息

def calculate_entropy(y):
    """使用信息熵替代基尼不纯度"""
    if len(y) == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def find_best_split(X, y, feature_index, min_samples_split=5):
    """改进的分裂点选择"""
    if len(y) < min_samples_split:
        return None, float('inf')
        
    thresholds = np.percentile(X[:, feature_index], np.arange(10, 100, 10))  # 使用百分位数作为候选阈值
    best_threshold = None
    best_entropy = float('inf')
    
    current_entropy = calculate_entropy(y)
    
    for threshold in thresholds:
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        
        # 确保每个子节点至少有min_samples_split个样本
        if np.sum(left_mask) < min_samples_split or np.sum(right_mask) < min_samples_split:
            continue
            
        left_entropy = calculate_entropy(y[left_mask])
        right_entropy = calculate_entropy(y[right_mask])
        
        # 计算信息增益
        weighted_entropy = (np.sum(left_mask) * left_entropy + np.sum(right_mask) * right_entropy) / len(y)
        information_gain = current_entropy - weighted_entropy
        
        if weighted_entropy < best_entropy:
            best_entropy = weighted_entropy
            best_threshold = threshold
            
    return best_threshold, best_entropy

def build_tree(X, y, feature_order, max_depth=None, min_samples_split=5, depth=0):
    node = TreeNode()
    node.samples = len(y)
    node.depth = depth
    
    if len(y) < min_samples_split or (max_depth is not None and depth >= max_depth):
        node.value = np.bincount(y).argmax()
        return node
        
    best_entropy = float('inf')
    best_feature = None
    best_threshold = None
    
    # 在给定的特征顺序中寻找最佳分裂
    for feature_index in feature_order:
        threshold, entropy = find_best_split(X, y, feature_index, min_samples_split)
        if threshold is not None and entropy < best_entropy:
            best_entropy = entropy
            best_feature = feature_index
            best_threshold = threshold
    
    if best_feature is None:
        node.value = np.bincount(y).argmax()
        return node
        
    node.feature_index = best_feature
    node.threshold = best_threshold
    
    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask
    
    node.left = build_tree(X[left_mask], y[left_mask], feature_order, 
                                  max_depth, min_samples_split, depth + 1)
    node.right = build_tree(X[right_mask], y[right_mask], feature_order, 
                                   max_depth, min_samples_split, depth + 1)
    
    return node
