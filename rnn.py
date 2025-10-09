import numpy as np

class RNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        初始化RNN参数
        :param input_size: 输入特征维度（如词嵌入维度）
        :param hidden_size: 隐藏层维度
        :param output_size: 输出维度（如词汇表大小）
        """
        # 权重初始化（均值0，方差0.01，避免初始值过大）
        self.w_xh = np.random.randn(hidden_size, input_size) * 0.01  # 输入→隐藏
        self.w_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # 隐藏→隐藏
        self.w_hy = np.random.randn(output_size, hidden_size) * 0.01  # 隐藏→输出
        
        # 偏置初始化
        self.b_h = np.zeros((hidden_size, 1))  # 隐藏层偏置
        self.b_y = np.zeros((output_size, 1))  # 输出层偏置
        
        self.hidden_size = hidden_size

    def init_hidden(self, batch_size: int) -> np.ndarray:
        """初始化隐藏状态（全0）"""
        return np.zeros((self.hidden_size, batch_size))

    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        前向传播
        :param x: 输入序列，形状为(seq_len, batch_size, input_size)
        :param h_prev: 初始隐藏状态，形状为(hidden_size, batch_size)
        :return: 
            outputs: 所有时间步的输出，形状为(seq_len, batch_size, output_size)
            h_final: 最后一个时间步的隐藏状态，形状为(hidden_size, batch_size)
        """
        #import pdb; pdb.set_trace()
        seq_len, batch_size = x.shape[0], x.shape[1]
        outputs = []  # 存储每个时间步的输出
        
        for t in range(seq_len):
            x_t = x[t].T  # 取第t步输入，转置为(input_size, batch_size)
           
            # tanh 是一个激活函数，是一个S形曲线;
            # 计算当前隐藏状态：h_t = tanh(W_xh·x_t + W_hh·h_prev + b_h)
            h_t = np.tanh(
                np.dot(self.w_xh, x_t) +  # (hidden_size, batch_size)
                np.dot(self.w_hh, h_prev) +  # (hidden_size, batch_size)
                self.b_h  # 广播到(hidden_size, batch_size)
            )
            
            # 计算当前输出：y_t = W_hy·h_t + b_y
            y_t = np.dot(self.w_hy, h_t) + self.b_y  # (output_size, batch_size)
            outputs.append(y_t.T)  # 转置回(batch_size, output_size)并存储
            
            h_prev = h_t  # 更新隐藏状态，用于下一时间步
        
        return np.array(outputs), h_prev  # outputs形状：(seq_len, batch_size, output_size)


# 测试示例
if __name__ == "__main__":
    # 超参数
    input_size = 10    # 输入特征维度（如词嵌入维度）
    hidden_size = 20   # 隐藏层维度
    output_size = 5    # 输出维度（如分类类别数）
    seq_len = 3        # 序列长度
    batch_size = 2     # 批量大小

    # 初始化RNN
    rnn = RNN(input_size, hidden_size, output_size)
    
    # 生成随机输入序列 (seq_len, batch_size, input_size)
    # 输入理解：
    # 举例，输入句子1：["机", "器", "学", "习", "趣"]，句子2:["我", "喜", "欢", "AI", "<PAD>"]
    #    seq_len, 每个句子的长度，短句子加pad，如例子中，句长为5
    #    batch_size, 同时处理两个句子，batch size是 2
    #    input_size, 每个token用一个向量表示， 例如，“我”→[0.2, 0.5, -0.1, ..., 0.3]
    x = np.random.randn(seq_len, batch_size, input_size)
    
    # 初始化隐藏状态
    h0 = rnn.init_hidden(batch_size)
    
    # 前向传播
    outputs, h_final = rnn.forward(x, h0)
    
    # 输出形状验证
    print(f"输入序列形状: {x.shape}")              # (3, 2, 10)
    print(f"输出序列形状: {outputs.shape}")         # (3, 2, 5)
    print(f"最终隐藏状态形状: {h_final.shape}")     # (20, 2)
