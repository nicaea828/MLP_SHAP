import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import shap


# 检查是否可用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed()

# 数据预处理
class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def preprocess(self, file_path):
        # 读取数据
        df = pd.read_csv(file_path)
        
        # 检查缺失值
        print("\nMissing values:\n", df.isnull().sum())
        
        # 特征和目标变量分离
        X = df.iloc[:, :13].values
        y = df['target'].values
        
        # 标准化特征
        X = self.scaler.fit_transform(X)
        
        # 转换为张量
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        return X_tensor, y_tensor, df

# 改进的神经网络模型
class ImprovedHeartDiseaseClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super(ImprovedHeartDiseaseClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 构建层次结构
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# 训练器类
class ModelTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model.to(device)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            labels = labels.view(-1, 1)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                labels = labels.view(-1, 1)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                predictions.extend((outputs > 0.5).float().cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        return (total_loss / len(data_loader), 
                np.array(predictions).reshape(-1),
                np.array(true_labels).reshape(-1))



# SHAP分析器类
class ShapAnalyzer:
    def __init__(self, model, train_data, feature_names):
        self.model = model
        self.train_data = train_data.detach().numpy() if torch.is_tensor(train_data) else train_data
        self.feature_names = np.array(feature_names)
        
    def explain_predictions(self, background_size=100):
        self.model.cpu().eval()
        background = torch.FloatTensor(self.train_data[:background_size])
        explainer = shap.DeepExplainer(self.model, background)
        all_data = torch.FloatTensor(self.train_data)
        shap_values = explainer.shap_values(all_data)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        return np.array(shap_values)
    
    def plot_feature_importance(self, shap_values):
        try:
            # 确保shap_values是numpy数组
            shap_values = np.array(shap_values)
            
            # 计算特征重要性
            importances = np.mean(np.abs(shap_values), axis=0).flatten()
            
            # 创建特征名称和重要性的列表
            features_and_importances = list(zip(self.feature_names, importances))
            
            # 按重要性排序
            features_and_importances.sort(key=lambda x: float(x[1]), reverse=True)
            
            # 分离特征名称和重要性
            features = [x[0] for x in features_and_importances]
            importances = [float(x[1]) for x in features_and_importances]
            
            # 创建图形
            plt.figure(figsize=(12, 8))
            y_pos = np.arange(len(features))
            
            print("Debug info:")
            print(f"y_pos shape: {y_pos.shape}")
            print(f"importances length: {len(importances)}")
            print(f"importances values: {importances}")
            
            # 使用plt.bar而不是plt.barh，并确保数据是列表或一维数组
            plt.bar(y_pos, importances)
            plt.xticks(y_pos, features, rotation=45, ha='right')
            plt.xlabel('Features')
            plt.ylabel('Mean |SHAP value|')
            plt.title('Feature Importance Based on SHAP Values')
            
            # 调整布局以确保标签可见
            plt.tight_layout()
            plt.show()
            
            # 打印数值结果
            print("\nFeature Importance Rankings:")
            for feat, imp in zip(features, importances):
                print(f"{feat}: {imp:.6f}")
                
        except Exception as e:
            print(f"Error in feature importance plot: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            # 进一步的调试信息
            print("\nDebug information:")
            print(f"Shape of shap_values: {shap_values.shape}")
            print(f"Type of importances: {type(importances)}")
            print(f"First few importance values: {importances[:5] if len(importances) > 5 else importances}")






# 主训练流程
def main():
    # 数据预处理
    preprocessor = DataPreprocessor()
    X_tensor, y_tensor, df = preprocessor.preprocess('heart.csv')
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # 划分数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 模型参数
    input_size = X_tensor.shape[1]
    hidden_sizes = [64, 32, 16]
    output_size = 1
    
    # 初始化模型和训练组件
    model = ImprovedHeartDiseaseClassifier(input_size, hidden_sizes, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, verbose=True
    )
    
    # 训练器
    trainer = ModelTrainer(model, criterion, optimizer, scheduler)
    
    # 训练参数
    epochs = 300
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # 训练循环
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_pred, val_true = trainer.evaluate(val_loader)
        val_accuracy = (val_pred == val_true).mean()
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Val Accuracy: {val_accuracy:.4f}')
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_pred, test_true = trainer.evaluate(test_loader)
    test_accuracy = (test_pred == test_true).mean()
    
    print("\nTest Results:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_true, test_pred))
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(test_true, test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Validation Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    def analyze_with_shap(model, X_tensor, feature_names):
        print("\nPerforming SHAP Analysis...")
        
        try:
            # 初始化分析器
            analyzer = ShapAnalyzer(model, X_tensor, feature_names)
            
            # 计算SHAP值
            shap_values = analyzer.explain_predictions(background_size=100)
            print("SHAP values calculated successfully")
            
            if shap_values is not None:
                print(f"SHAP values shape: {shap_values.shape}")
            
            # 绘制特征重要性图
            print("\nGenerating Feature Importance Plot...")
            analyzer.plot_feature_importance(shap_values)
            
            return shap_values, analyzer
            
        except Exception as e:
            print(f"Error during SHAP analysis: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None, None


    print("\nPerforming SHAP Analysis...")
    feature_names = df.columns[:13].tolist()
    shap_values, analyzer = analyze_with_shap(model, X_tensor, feature_names)



if __name__ == "__main__":
    main()
