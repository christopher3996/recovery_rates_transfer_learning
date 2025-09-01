import torch
import torch.nn as nn

# 定义 ScaledSigmoid 激活函数
class ScaledSigmoid(nn.Module):
    def __init__(self, scale=1.1):
        super(ScaledSigmoid, self).__init__()
        self.scale = scale

    def forward(self, x):
        return torch.sigmoid(x) * self.scale

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(out_features, out_features)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 如果输入和输出维度不同，通过线性层调整
        if in_features != out_features:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = nn.Identity()
        
    def forward(self, x):
        residual = self.residual(x)
        
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        
        out += residual
        out = self.relu(out)
        return out

# 定义主模型
class EnhancedMLP(nn.Module):
    def __init__(self, input_size):
        super(EnhancedMLP, self).__init__()
        self.input_layer1 = nn.Identity()
        self.input_layer2 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.dropout_input = nn.Dropout(0.1)
        
        self.res_block1 = ResidualBlock(256, 128, dropout_rate=0.1)
        self.res_block2 = ResidualBlock(128, 64, dropout_rate=0.1)
        self.res_block3 = ResidualBlock(64, 32, dropout_rate=0.1)
        
        self.output_layer = nn.Sequential(
            nn.Linear(32, 1),
            ScaledSigmoid(scale=1.1)  # 确保输出在 0 到 1.1 之间
        )
    
    def forward(self, x):
        x = self.input_layer1(x)
        x = self.input_layer2(x)
        x = self.relu(x)
        x = self.dropout_input(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        x = self.output_layer(x)
        return x

import torch.nn.functional as F
class SimpleMLP(nn.Module):
    def __init__(self, input_size):
        super(SimpleMLP, self).__init__()
        self.input_layer1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(p=0.5)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        # 对 input_layer1 做 ReLU + Dropout
        x = F.relu(self.input_layer1(x))
        x = self.dropout1(x)

        # 第二层
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        # 输出层
        x = self.out(x)
        return x