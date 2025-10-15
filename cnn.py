import numpy as np
import time  # 导入计时模块

class Conv2D:
    """二维卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        self.kernel = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.bias = np.zeros(out_channels)
        self.padding = padding

    def __call__(self, x):
        # x形状: (batch, in_channels, H, W)
        batch, in_c, H, W = x.shape
        out_c, _, kH, kW = self.kernel.shape
        # 填充
        x_pad = np.pad(x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        # 输出尺寸
        out_H = H + 2*self.padding - kH + 1
        out_W = W + 2*self.padding - kW + 1
        out = np.zeros((batch, out_c, out_H, out_W))
        
        # 卷积计算
        for b in range(batch):
            for c_out in range(out_c):
                for h in range(out_H):
                    for w in range(out_W):
                        # 提取感受野
                        field = x_pad[b, :, h:h+kH, w:w+kW]
                        out[b, c_out, h, w] = np.sum(field * self.kernel[c_out]) + self.bias[c_out]
        return out

# np.mean(x, axis=(0, 2, 3)) 理解
#x = [
#    # 样本0：(3,4,4)
#    [
#        [[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]],  # 通道0
#        [[2,2,2,2], [2,2,2,2], [2,2,2,2], [2,2,2,2]],  # 通道1
#        [[5,5,5,5], [5,5,5,5], [5,5,5,5], [5,5,5,5]]   # 通道2
#    ],
#    # 样本1：(3,4,4)
#    [
#        [[3,3,3,3], [3,3,3,3], [3,3,3,3], [3,3,3,3]],  # 通道0
#        [[4,4,4,4], [4,4,4,4], [4,4,4,4], [4,4,4,4]],  # 通道1
#        [[7,7,7,7], [7,7,7,7], [7,7,7,7], [7,7,7,7]]   # 通道2
#    ]
#]
#
#x = np.array(x)  # 形状为 (2, 3, 4, 4)
#axis=(0, 2, 3) 表示 “对这三个维度的所有元素求平均”，保留通道维度(axis=1)
#通道 0 的均值计算：
#元素来源：样本 0 的通道 0（16 个 1） + 样本 1 的通道 0（16 个 3）
#总元素个数：2（样本）×4（行）×4（列）= 32 个
#总和：16×1 + 16×3 = 16 + 48 = 64
#均值：64 ÷ 32 = 2
#
#取均值的最终结果是, np.array([2, 3, 6])

class BatchNorm2D:
    """批归一化层"""
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)  # 缩放参数
        self.beta = np.zeros(num_features)  # 偏移参数
        self.running_mean = np.zeros(num_features)  # 移动平均均值
        self.running_var = np.ones(num_features)   # 移动平均方差
        self.training = True  # 训练/推理模式

    def __call__(self, x):
        # x形状: (batch, channels, H, W)
        batch, channels, H, W = x.shape
        if self.training:
            # 计算批次均值和方差
            mean = np.mean(x, axis=(0, 2, 3))  # 按通道计算均值
            var = np.var(x, axis=(0, 2, 3))    # 按通道计算方差
            # 更新移动平均
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # 推理时使用移动平均
            mean = self.running_mean
            var = self.running_var
        
        # 归一化 + 缩放偏移
        x_norm = (x - mean.reshape(1, channels, 1, 1)) / np.sqrt(var.reshape(1, channels, 1, 1) + self.eps)
        return self.gamma.reshape(1, channels, 1, 1) * x_norm + self.beta.reshape(1, channels, 1, 1)


class ReLU:
    """ReLU激活函数"""
    def __call__(self, x):
        return np.maximum(0, x)

class MaxPool2D:
    """最大池化层"""
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, x):
        # x形状: (batch, channels, H, W)
        batch, c, H, W = x.shape
        out_H = (H - self.kernel_size) // self.stride + 1
        out_W = (W - self.kernel_size) // self.stride + 1
        out = np.zeros((batch, c, out_H, out_W))
        
        for b in range(batch):
            for ch in range(c):
                for h in range(out_H):
                    for w in range(out_W):
                        # 池化窗口
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        out[b, ch, h, w] = np.max(x[b, ch, h_start:h_end, w_start:w_end])
        return out


class AdaptiveAvgPool2D:
    """自适应平均池化（固定输出1x1）"""
    def __call__(self, x):
        # 对H和W维度求平均，输出形状: (batch, channels, 1, 1)
        return np.mean(x, axis=(2, 3), keepdims=True)

class Linear:
    """全连接层"""
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(out_features, in_features) * 0.01
        self.bias = np.zeros(out_features)

    def __call__(self, x):
        # x形状: (batch, in_features)
        return x @ self.weight.T + self.bias  # 矩阵乘法 + 偏置

class Dropout:
    """Dropout层"""
    def __init__(self, p=0.5):
        self.p = p
        self.training = True  # 训练/推理模式

    def __call__(self, x):
        if self.training:
            mask = np.random.binomial(1, 1 - self.p, size=x.shape) / (1 - self.p)
            return x * mask
        else:
            return x

class Flatten:
    """展平层"""
    def __call__(self, x):
        # 展平为(batch, features)
        return x.reshape(x.shape[0], -1)

class CNN:
    """简洁CNN模型（纯NumPy实现）"""
    def __init__(self, in_channels=3, num_classes=10, hidden_channels=[64, 32]):
        # 特征提取模块
        self.features = [
            # 卷积块1
            Conv2D(in_channels, hidden_channels[0]),
            BatchNorm2D(hidden_channels[0]),
            ReLU(),
            MaxPool2D(),
            
            # 卷积块2
            Conv2D(hidden_channels[0], hidden_channels[1]),
            BatchNorm2D(hidden_channels[1]),
            ReLU(),
            MaxPool2D(),
            
            AdaptiveAvgPool2D()
        ]
        
        # 分类头
        self.classifier = [
            Flatten(),
            Linear(hidden_channels[-1], 128),
            ReLU(),
            Dropout(p=0.5),
            Linear(128, num_classes)
        ]

    def train(self):
        """切换到训练模式"""
        for layer in self.features + self.classifier:
            if hasattr(layer, 'training'):
                layer.training = True

    def eval(self):
        """切换到推理模式"""
        for layer in self.features + self.classifier:
            if hasattr(layer, 'training'):
                layer.training = False

    def __call__(self, x):
        # 特征提取 
        for layer in self.features:
            x = layer(x)
        # 分类
        for layer in self.classifier:
            x = layer(x)
        return x

# 测试模型
if __name__ == "__main__":
    # 初始化模型（RGB图像分类10类)
    # 输入是224*224个像素点，每个像素点是rgb，于是又乘以了3, 10类是输出i分类结果，可以认为是猫，狗...,等10种动物）
    model = CNN(in_channels=3, num_classes=10)
    model.train()  # 切换到训练模式
    
    # 模拟输入：(batch=2, 3通道, 224x224)
    dummy_input = np.random.randn(2, 3, 224, 224)
   

    start_time = time.perf_counter()  # 记录开始时间（高精度计时）
    output = model(dummy_input)       # 执行前向传播
    end_time = time.perf_counter()    # 记录结束时间 
   
    # 计算并打印耗时
    duration = end_time - start_time

    print(f"单次forward耗时: {duration:.6f}秒")  # 保留6位小数，精确到微秒级

    # 验证输出形状
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")  # 应输出(2, 10)
    #print(f"输出: {output}")  # 应输出(2, 10)
    print("模型结构验证通过！")
