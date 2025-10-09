import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class LSTM:
    # 省略LSTM类的初始化和前向传播代码（与之前相同）
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        # rnn中仅有一套(w_xh,w_xh),这里有四套，ifoc
        self.W_xi = np.random.randn(hidden_size, input_size) * 0.01
        self.W_xf = np.random.randn(hidden_size, input_size) * 0.01
        self.W_xo = np.random.randn(hidden_size, input_size) * 0.01
        self.W_xc = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hi = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hf = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_ho = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hc = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_i = np.zeros((hidden_size, 1))
        self.b_f = np.ones((hidden_size, 1))
        self.b_o = np.zeros((hidden_size, 1))
        self.b_c = np.zeros((hidden_size, 1))
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros((output_size, 1))
        self.hidden_size = hidden_size

    def init_states(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((self.hidden_size, batch_size)), np.zeros((self.hidden_size, batch_size))

    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        i = sigmoid(np.dot(self.W_xi, x) + np.dot(self.W_hi, h_prev) + self.b_i) # 决定了当前时间步的c_tilde中，有多少需要被 “写入” 到细胞状态（长期记忆）中，后边要乘c_tilde
        f = sigmoid(np.dot(self.W_xf, x) + np.dot(self.W_hf, h_prev) + self.b_f) # 决定了上一时间步的细胞状态c_prev中，有多少信息需要被 “遗忘”,后边要乘 c_prev
        o = sigmoid(np.dot(self.W_xo, x) + np.dot(self.W_ho, h_prev) + self.b_o) # 决定了当前细胞状态c中，有多少信息需要被 “提取” 到隐藏状态（h，短期记忆），并用于当前时间步的输出 y_t。
        c_tilde = np.tanh(np.dot(self.W_xc, x) + np.dot(self.W_hc, h_prev) + self.b_c) # 信息载体, 前边的ifo都是门，只有这个是可能被写入细胞状态的 “候选内容”
        c = f * c_prev + i * c_tilde # 遗忘多少c_prev，要写入多少c_tilde
        h = o * np.tanh(c) # 有多少要提取到h，用于本次输出
        y = np.dot(self.W_hy, h) + self.b_y
        y = softmax(y.T)
        return y, h, c
    
    # 举例
    # 假设处理句子 “我昨天学习了 LSTM，今天继续____”：
    # 遗忘门 f：保留 “我”“学习”“LSTM”（核心信息），遗忘 “昨天”（时间状语，当前时间变为 “今天”）；
    # 细胞候选 c_tilde：生成 “今天”“继续” 的新信息；
    # 输入门 i：允许 “今天”“继续” 写入细胞状态（更新长期记忆）；
    # 细胞状态 c：最终长期记忆 = 被保留的旧信息（我、学习、LSTM） + 被允许的新信息（今天、继续）；
    # 输出门 o：提取 “继续”“学习” 相关信息到隐藏状态 h，用于预测下一个词（如 “学习”）。
    
    def forward_sequence(self, x_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        seq_len, batch_size = x_seq.shape[0], x_seq.shape[1]
        h, c = self.init_states(batch_size)
        outputs = []
        for t in range(seq_len):
            x_t = x_seq[t].T
            y_t, h, c = self.forward(x_t, h, c)
            outputs.append(y_t)
        return np.array(outputs), h, c


def greedy_decode(outputs: np.ndarray) -> np.ndarray:
    """
    贪心算法解码：逐时间步选择概率最高的token索引
    :param outputs: LSTM输出，形状(seq_len, batch_size, output_size)
    :return: 解码后的序列（token索引），形状(batch_size, seq_len)
    """
    # 对每个时间步、每个样本，取output_size维度的最大值索引
    # outputs.shape: (seq_len, batch_size, output_size) → 沿axis=2取argmax
    token_indices = np.argmax(outputs, axis=2)  # 结果形状: (seq_len, batch_size)
    # 转置为(batch_size, seq_len)，方便按样本查看序列
    return token_indices.transpose(1, 0)


# 测试示例
if __name__ == "__main__":
    # 超参数
    input_size = 8    # 输入特征维度（假设词汇表大小为8）
    hidden_size = 16
    output_size = 8   # 输出维度=词汇表大小
    seq_len = 4       # 输入序列长度
    batch_size = 2    # 2个样本

    # 词汇表（示例：8个token）
    vocab = ["<pad>", "我", "爱", "学习", "LSTM", "算法", "！", "。"]
    vocab_idx = {token: i for i, token in enumerate(vocab)}  # token→索引映射

    # 初始化LSTM
    lstm = LSTM(input_size, hidden_size, output_size)

    # 生成输入序列（模拟one-hot编码，每个时间步一个token）
    # 输入序列1: "我 爱 学习" → 补全到seq_len=4（最后加<pad>）
    x1 = np.array([
        [1,0,0,0,0,0,0,0],  # "我"的one-hot
        [0,1,0,0,0,0,0,0],  # "爱"的one-hot
        [0,0,1,0,0,0,0,0],  # "学习"的one-hot
        [0,0,0,1,0,0,0,0]   # "<pad>"的one-hot（补位）
    ])
    # 输入序列2: "LSTM 算法 ！" → 补全到seq_len=4
    x2 = np.array([
        [0,0,0,0,1,0,0,0],  # "LSTM"的one-hot
        [0,0,0,0,0,1,0,0],  # "算法"的one-hot
        [0,0,0,0,0,0,1,0],  # "！"的one-hot
        [0,0,0,1,0,0,0,0]   # "<pad>"的one-hot
    ])
    x_seq = np.stack([x1, x2], axis=1)  # 组合成(batch_size=2)，形状(4, 2, 8)

    # LSTM前向传播得到输出
    outputs, _, _ = lstm.forward_sequence(x_seq)
    print("outputs形状:", outputs.shape)  # (4, 2, 8) → (seq_len, batch_size, output_size)

    # 贪心解码得到token索引序列
    pred_indices = greedy_decode(outputs)
    print("\n解码后的索引序列（batch_size, seq_len）:\n", pred_indices)

    # 将索引映射为实际token
    pred_sequences = []
    for b in range(batch_size):
        sequence = [vocab[idx] for idx in pred_indices[b]]
        pred_sequences.append(" ".join(sequence))

    print("\n解码后的token序列:")
    for i, seq in enumerate(pred_sequences):
        print(f"样本{i+1}: {seq}")
