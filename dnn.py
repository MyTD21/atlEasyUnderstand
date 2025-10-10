import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU激活函数（隐藏层用）"""
    return np.maximum(0, x)

def softmax(x: np.ndarray) -> np.ndarray:
    """softmax激活函数（输出层分类用）"""
    x_max = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class DNN:
    def __init__(self, layer_sizes: list[int], output_activation: str = "linear"):
        """
        初始化DNN模型
        :param layer_sizes: 各层维度列表，如[输入维度, 隐藏层1, 隐藏层2, 输出维度]
        :param output_activation: 输出层激活函数，可选"linear"(回归)或"softmax"(分类)
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1  # 层数 = 总层数 - 1
        self.output_activation = output_activation
       
        # 初始化权重和偏置（简单随机初始化，模拟已训练模型）
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            # 权重: (当前层输入维度, 当前层输出维度)
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            # 偏置: (1, 当前层输出维度)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
            # 例如输入layer信息为[1, 64, 32, 1]，需要生成矩阵为1 * 64，64 * 32，32 * 1；
            # bias作为偏移，size为64，32，1

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播（处理批量输入）
        :param x: 输入数据，形状(batch_size, input_dim)
        :return: 输出层结果，形状(batch_size, output_dim)
        """
        a = x  # 初始激活值为输入
        for i in range(self.num_layers):
            # 计算当前层加权和: z = w·a + b
            z = np.dot(a, self.weights[i]) + self.biases[i]
            # 隐藏层用ReLU，输出层用指定激活函数
            if i < self.num_layers - 1:
                a = relu(z)
            else:
                if self.output_activation == "softmax":
                    a = softmax(z)
                else:  # linear
                    a = z
        return a

    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测接口（封装前向传播）"""
        return self.forward(x)


# 测试示例
if __name__ == "__main__":

    # 1. 定义回归任务DNN（输入1维→隐藏层64→隐藏层32→输出1维，线性输出）
    reg_dnn = DNN(layer_sizes=[1, 64, 32, 1], output_activation="linear")
    
    # 输入5个测试样本（x值）
    test_x = np.array([[-3.0], [-1.0], [0.0], [2.0], [5.0]])  # 形状(5, 1)
    reg_pred = reg_dnn.predict(test_x)
    print("回归任务, 例如，模拟函数y=3x²+5:")
    print(f"输入x: {test_x.flatten()}")
    print(f"预测y: {reg_pred.flatten().round(4)}\n")
    
    # 2. 定义分类任务DNN（输入4维→隐藏层16→输出3维，softmax输出）
    cls_dnn = DNN(layer_sizes=[4, 16, 32, 16, 3], output_activation="softmax")
    
    # 输入2个测试样本（4维特征）
    cls_x = np.random.randn(2, 4)  # 随机生成2个4维样本
    cls_pred = cls_dnn.predict(cls_x)
    print("分类任务预测（概率分布）：")
    print(f"样本1概率: {cls_pred[0].round(4)}（和为{cls_pred[0].sum().round(4)}）")
    print(f"样本2概率: {cls_pred[1].round(4)}（和为{cls_pred[1].sum().round(4)}）")
