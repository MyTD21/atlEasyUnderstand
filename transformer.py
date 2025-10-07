import numpy as np
import math
from typing import Optional

# 生成随机矩阵（均值0，方差0.01的正态分布）
def random_matrix(rows: int, cols: int) -> np.ndarray:
    return np.random.normal(0, 0.01, size=(rows, cols)).astype(np.float32)

# 层归一化：对每个样本的特征维度做归一化
# LayerNorm vs BatchNorm
# x = np.array([
#     [
#        [1.0, 2.0, 3.0, 4.0, 5.0],   # 第1个token（seq=0）
#        [2.0, 4.0, 6.0, 8.0, 10.0],  # 第2个token（seq=1）
#        [3.1, 6.1, 9.1, 12.1, 15.1]  # 第3个token（seq=2）
#     ]
# ],  dtype=np.float32)

# LayerNorm mean = (1.0 + 2.0 + 3.0 + 4.0 + 5.0) / 5
# BatchNorm mean = (1.0 + 2.0 + 3.1) / 3

class LayerNorm:
    def __init__(self, d_model: int):
        self.gamma = np.ones((1, 1, d_model), dtype=np.float32)  # 缩放参数（适配[batch, seq, d_model]）
        self.beta = np.zeros((1, 1, d_model), dtype=np.float32)  # 偏移参数
        self.eps = 1e-5  # 防止除零

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: 输入形状 [batch_size, seq_len, d_model]
        返回：归一化后的结果，形状相同
        """
        # 按特征维度（d_model）计算均值和方差
        mean = x.mean(axis=2, keepdims=True)  # [batch_size, seq_len, 1]
        var = ((x - mean) **2).mean(axis=2, keepdims=True)  # [batch_size, seq_len, 1]

        # 归一化
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # 缩放和平移（广播机制）
        return self.gamma * x_norm + self.beta

# 前馈网络：对每个位置的特征独立做非线性变换
class FeedForward:
    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model  # 输入输出维度
        self.d_ff = d_ff        # 中间层维度
        self.w1 = random_matrix(d_model, d_ff)  # 第一层权重 [d_model, d_ff]
        self.b1 = np.zeros((1, 1, d_ff), dtype=np.float32)  # 第一层偏置（适配广播）
        self.w2 = random_matrix(d_ff, d_model)  # 第二层权重 [d_ff, d_model]
        self.b2 = np.zeros((1, 1, d_model), dtype=np.float32)  # 第二层偏置

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: 输入形状 [batch_size, seq_len, d_model]
        返回：输出形状相同
        """
        # 第一层：Linear + ReLU → [batch, seq, d_ff]
        out1 = x @ self.w1 + self.b1  # 利用广播自动适配batch和seq维度
        out1 = np.maximum(out1, 0)  # ReLU激活

        # 第二层：Linear → [batch, seq, d_model]
        out2 = out1 @ self.w2 + self.b2
        return out2


# 多头注意力：并行计算多个注意力头
class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model      # 模型总维度
        self.n_heads = n_heads      # 注意力头数量
        self.d_k = d_model // n_heads  # 每个头的维度（必须整除）

        # Q、K、V和输出投影的权重矩阵
        self.w_q = random_matrix(d_model, d_model)
        self.w_k = random_matrix(d_model, d_model)
        self.w_v = random_matrix(d_model, d_model)
        self.w_o = random_matrix(d_model, d_model)

    def scaled_dot_product(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        缩放点积注意力计算
        q/k/v: 形状 [batch_size, n_heads, seq_len, d_k]
        mask: 可选掩码，形状 [seq_len, seq_len]（1表示需要掩码的位置）
        返回：注意力输出，形状 [batch_size, seq_len, n_heads*d_k]
        """
        batch_size, n_heads, seq_len, d_k = q.shape

        # 计算Q*K^T / sqrt(d_k)：[batch, n_heads, seq_len, seq_len]
        scores = np.matmul(q, k.transpose(0, 1, 3, 2))  # K转置最后两个维度
        scores /= math.sqrt(self.d_k)

        # 应用掩码（解码器自注意力用）
        if mask is not None:
            # 扩展掩码维度以匹配scores：[1, 1, seq_len, seq_len]
            mask = mask[np.newaxis, np.newaxis, :, :]
            scores = scores + (mask * -1e9)  # 掩码位置设为负无穷

        # 计算Softmax得到注意力权重
        attn_weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)

        # 加权求和：attn_weights * V → [batch, n_heads, seq_len, d_k]
        attn_output = np.matmul(attn_weights, v)

        # 拼接多头：[batch, seq_len, n_heads*d_k]
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, n_heads * d_k)
        return attn_output

    def forward(self, x: np.ndarray, mem: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        x: 输入序列 [batch_size, seq_len, d_model]
        mem: 编码器输出（自注意力时为None，交叉注意力时使用）
        mask: 解码器掩码
        返回：注意力输出，形状相同
        """
        batch_size, seq_len, _ = x.shape

        # 生成Q、K、V（自注意力用x，交叉注意力用mem作为K和V）
        q = x @ self.w_q  # [batch, seq_len, d_model]
        k = x @ self.w_k if mem is None else mem @ self.w_k
        v = x @ self.w_v if mem is None else mem @ self.w_v

        # 拆分多头：[batch, seq_len, d_model] → [batch, n_heads, seq_len, d_k]
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, k.shape[1], self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, v.shape[1], self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # 计算注意力
        attn_out = self.scaled_dot_product(q, k, v, mask)  # [batch, seq_len, d_model]

        # 输出投影
        return attn_out @ self.w_o


# 编码器层：多头自注意力 + 前馈网络
class EncoderLayer:
    def __init__(self, d_model: int, d_ff: int, n_heads: int):
        self.attn = MultiHeadAttention(d_model, n_heads)  # 自注意力
        self.ff = FeedForward(d_model, d_ff)              # 前馈网络
        self.ln1 = LayerNorm(d_model)                      # 归一化1
        self.ln2 = LayerNorm(d_model)                      # 归一化2

    def forward(self, x: np.ndarray) -> np.ndarray:
        # 自注意力 + 残差连接 + 归一化
        attn_out = self.attn.forward(x)
        x = self.ln1.forward(x + attn_out)

        # 前馈网络 + 残差连接 + 归一化
        ff_out = self.ff.forward(x)
        return self.ln2.forward(x + ff_out)


# 解码器层：掩码自注意力 + 交叉注意力 + 前馈网络
class DecoderLayer:
    def __init__(self, d_model: int, d_ff: int, n_heads: int):
        self.self_attn = MultiHeadAttention(d_model, n_heads)  # 掩码自注意力
        self.cross_attn = MultiHeadAttention(d_model, n_heads) # 交叉注意力
        self.ff = FeedForward(d_model, d_ff)                   # 前馈网络
        self.ln1 = LayerNorm(d_model)                           # 归一化1
        self.ln2 = LayerNorm(d_model)                           # 归一化2
        self.ln3 = LayerNorm(d_model)                           # 归一化3

    def forward(self, x: np.ndarray, enc_out: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # 掩码自注意力 + 残差 + 归一化
        self_attn_out = self.self_attn.forward(x, mask=mask)
        x = self.ln1.forward(x + self_attn_out)

        # 交叉注意力（关注编码器输出） + 残差 + 归一化
        cross_attn_out = self.cross_attn.forward(x, mem=enc_out)
        x = self.ln2.forward(x + cross_attn_out)

        # 前馈网络 + 残差 + 归一化
        ff_out = self.ff.forward(x)
        return self.ln3.forward(x + ff_out)

# 位置编码：给模型提供序列位置信息
def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """生成形状为 [seq_len, d_model] 的位置编码"""
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # 偶数维度用sin，奇数用cos（原论文公式）
            pe[pos, i] = math.sin(pos / (10000** (2*i / d_model)))
            if i + 1 < d_model:
                pe[pos, i+1] = math.cos(pos / (10000 **(2*i / d_model)))
    return pe


# Transformer主类
class Transformer:
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int, n_layers: int):
        self.d_model = d_model                # 模型维度,即embedding的维度
        self.vocab_size = vocab_size          # 词汇表大小
        self.n_layers = n_layers
        
        self.embed = random_matrix(vocab_size, d_model)  # 词嵌入矩阵 [vocab_size, d_model], 提前训练好，查表即可；
        self.final_ln = LayerNorm(d_model)    # 最终输出归一化
        self.proj = random_matrix(d_model, vocab_size)   # 输出投影矩阵 [d_model, vocab_size]

        # 初始化编码器和解码器层
        self.encoders = [EncoderLayer(d_model, d_ff, n_heads) for _ in range(n_layers)]
        self.decoders = [DecoderLayer(d_model, d_ff, n_heads) for _ in range(n_layers)]

    def create_decoder_mask(self, seq_len: int) -> np.ndarray:
        """生成解码器掩码（上三角矩阵，1表示需要掩码的位置）"""
        return np.triu(np.ones((seq_len, seq_len)), k=1).astype(np.float32)

    def forward(self, src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        """
        前向传播
        src: 源序列，形状 [batch_size, src_seq_len]（元素为token索引）
        tgt: 目标序列，形状 [batch_size, tgt_seq_len]（元素为token索引）
        返回：输出形状 [batch_size, tgt_seq_len*vocab_size]
        """
        batch_size = src.shape[0]
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        # 1. 编码器处理源序列
        # 1.1 词嵌入：[batch, src_seq] → [batch, src_seq, d_model]
        src_embed = self.embed[src]  # 直接索引得到三维形状

        # 1.2 添加位置编码
        pe_src = positional_encoding(src_seq_len, self.d_model)  # [src_seq, d_model]
        src_embed += pe_src[np.newaxis, :, :]  # 扩展批次维度并广播

        # 1.3 编码器堆叠
        enc_out = src_embed
        for enc in self.encoders:
            enc_out = enc.forward(enc_out)

        # 2. 解码器处理目标序列
        # 2.1 词嵌入：[batch, tgt_seq] → [batch, tgt_seq, d_model]
        tgt_embed = self.embed[tgt]  # 三维形状

        # 2.2 添加位置编码
        pe_tgt = positional_encoding(tgt_seq_len, self.d_model)  # [tgt_seq, d_model]
        tgt_embed += pe_tgt[np.newaxis, :, :]  # 扩展批次维度并广播

        # 2.3 解码器堆叠（带掩码）
        dec_out = tgt_embed
        dec_mask = self.create_decoder_mask(tgt_seq_len)
        for dec in self.decoders:
            dec_out = dec.forward(dec_out, enc_out, dec_mask)

        # 3. 最终输出：归一化 + 投影到词汇表 → [batch, tgt_seq, vocab_size]
        out = self.final_ln.forward(dec_out) @ self.proj
        # 展平为 [batch_size, tgt_seq_len*vocab_size] 匹配原输出格式
        return out.reshape(batch_size, tgt_seq_len * self.vocab_size)


def transformer_worker():
    vocab_size = 10000 # 词汇表大小
    
    batch_size = 1      # 批量大小    
    src_seq_len = 10    # 源序列长度
    tgt_seq_len = 15    # 目标序列长度

    # 初始化Transformer（使用较小参数）
    transformer = Transformer(
        vocab_size=vocab_size,
        d_model=512,    # 模型维度, 即embedding的维度
        n_heads=2,     # 注意力头数
        d_ff=1024,      # 前馈网络中间维度
        n_layers=4     # 编码器/解码器层数
    )

    # 生成示例输入
    src = np.random.randint(0, vocab_size, size=(batch_size, src_seq_len), dtype=np.int32)
    tgt = np.random.randint(0, vocab_size, size=(batch_size, tgt_seq_len), dtype=np.int32)

    # 前向传播
    output = transformer.forward(src, tgt)

    output_reshaped = output.reshape(batch_size, tgt_seq_len, vocab_size)  # 还原形状
    probs = np.exp(output_reshaped) / np.exp(output_reshaped).sum(axis=-1, keepdims=True)  # softmax计算概率
    predicted_tokens = np.argmax(probs, axis=-1)  # 每个位置取概率最高的token
    print("预测的token序列形状：", predicted_tokens.shape)  # (batch_size, tgt_seq_len)

    # 输出结果说明
    print(f"输入源序列形状：{src.shape}")  # (batch_size, src_seq_len)
    print(f"输入目标序列形状：{tgt.shape}")  # (batch_size, tgt_seq_len)
    print(f"输出形状：{output.shape}")      # (batch_size, tgt_seq_len * vocab_size)

if __name__ == '__main__':
    transformer_worker()

