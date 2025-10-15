import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    """因果自注意力层（GPT 核心组件，保证只能关注前文）"""
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model  # 隐藏层维度
        self.n_head = n_head    # 注意力头数
        self.d_k = d_model // n_head  # 每个头的维度
        
        # Q、K、V 线性投影（共享权重）
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        # 输出线性投影
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 缓存因果掩码（避免重复计算）
        self.causal_mask = None

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 1. 生成因果掩码（上三角为 0，下三角为 1，对角线及以下可见）
        if self.causal_mask is None or self.causal_mask.size(0) != seq_len:
            self.causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        
        # 2. Q、K、V 投影 + 拆分多头
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3*d_model]
        q, k, v = qkv.split(self.d_model, dim=2)  # 拆分 Q、K、V，各为 [batch, seq_len, d_model]
        # 多头拆分：[batch, seq_len, d_model] → [batch, n_head, seq_len, d_k]
        q = q.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        
        # 3. 计算注意力分数（带因果掩码）
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [batch, n_head, seq_len, seq_len]
        attn_scores = attn_scores.masked_fill(self.causal_mask == 0, -1e10)  # 屏蔽未来位置
        attn_weights = F.softmax(attn_scores, dim=-1)  # 归一化
        
        # 4. 注意力加权求和 + 合并多头
        attn_output = attn_weights @ v  # [batch, n_head, seq_len, d_k]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # 合并
        
        # 5. 输出投影
        return self.out_proj(attn_output)

# 理解mask
#causal_mask，下三角单位矩阵
#[[1, 0, 0],
# [1, 1, 0],
# [1, 1, 1]]
#
#attn_scores
#[[5, 3, 2],  # 第0个位置对0、1、2的关注度
# [4, 6, 1],  # 第1个位置对0、1、2的关注度
# [2, 5, 7]]  # 第2个位置对0、1、2的关注度
#
#attn_scores.masked_fill(self.causal_mask == 0, -1e10),将满足causal_mask中等于0的，换成-1e10
#[[5, -1e10, -1e10],  # 第0个位置只能关注自己（l=0），屏蔽l=1、2（未来）
# [4, 6, -1e10],      # 第1个位置可关注l=0、1，屏蔽l=2（未来）
# [2, 5, 7]]          # 第2个位置可关注所有前文（l=0、1、2）

#softmax之后
#attn_weights = [
#    [1.0, 0.0, 0.0],  # 位置0只关注自己（权重100%）
#    [0.2, 0.8, 0.0],  # 位置1关注0（20%）和自己（80%），未来2权重0
#    [0.01, 0.24, 0.75]# 位置2关注0（1%）、1（24%）、自己（75%）
#]
#
#v = [
#    [v0_0, v0_1, ..., v0_d],  # 位置0的特征
#    [v1_0, v1_1, ..., v1_d],  # 位置1的特征
#    [v2_0, v2_1, ..., v2_d]   # 位置2的特征（未来位置的特征）
#]
#
#attn_output = attn_weights @ v
#位置 0 的输出：1.0×v0 + 0.0×v1 + 0.0×v2 = v0（只包含自身特征，完全排除未来的 v1、v2）；
#位置 1 的输出：0.2×v0 + 0.8×v1 + 0.0×v2 = 0.2v0 + 0.8v1（只包含前文 0 和自身 1 的特征，排除未来的 v2）；
#位置 2 的输出：0.01×v0 + 0.24×v1 + 0.75×v2（无未来位置，包含所有前文）。

class TransformerDecoderBlock(nn.Module):
    """Transformer 解码器块（GPT 的基础单元）"""
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)  # 注意力前的层归一化
        self.attn = CausalSelfAttention(d_model, n_head)  # 因果自注意力
        self.dropout1 = nn.Dropout(dropout)
        
        self.ln2 = nn.LayerNorm(d_model)  # 前馈网络前的层归一化
        self.ffn = nn.Sequential(  # 前馈网络（GPT 用 4 倍升维）
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),  # GELU 激活
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # 残差连接 + 注意力
        x = x + self.dropout1(self.attn(self.ln1(x)))
        # 残差连接 + 前馈网络
        x = x + self.ffn(self.ln2(x))
        return x

class GPT3(nn.Module):
    """简化版 GPT-3 模型（Decoder-only 架构）"""
    def __init__(self, vocab_size, d_model=768, n_layer=6, n_head=12, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 1. 词嵌入（将 token 映射到 d_model 维度）
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 2. 位置嵌入（学习型位置编码，GPT 原版用此方式）
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # 3. Transformer 解码器块堆叠
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_head, dropout)
            for _ in range(n_layer)
        ])
        
        # 4. 最终层归一化 + 输出投影（映射到词汇表）
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """权重初始化（遵循 GPT 风格）"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        """前向传播（输入 token 序列，输出 logits）"""
        batch_size, seq_len = input_ids.shape
       
        # 1. 词嵌入 + 位置嵌入
        token_emb = self.token_embedding(input_ids)  # [batch, seq_len, d_model]
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)  # [1, seq_len]
        pos_emb = self.pos_embedding(pos_ids)  # [1, seq_len, d_model]
        x = token_emb + pos_emb  # 嵌入相加
        
        # 2. 经过所有解码器块
        for layer in self.layers:
            x = layer(x)
        
        # 3. 输出 logits（未归一化的概率）
        x = self.ln_final(x)
        logits = self.output_proj(x)  # [batch, seq_len, vocab_size]
        return logits

    def generate(self, input_ids, max_new_tokens=10, temperature=1.0, top_k=50):
        """自回归生成函数（核心功能：从输入 prompt 生成后续文本）"""
        self.eval()  # 推理模式
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 预测下一个 token 的 logits
                logits = self.forward(input_ids)  # [batch, seq_len, vocab_size]
                next_logits = logits[:, -1, :] / temperature  # 取最后一个位置，加温度调节
                
                # Top-K 采样（过滤低概率 token）
                if top_k is not None:
                    next_logits = self._top_k_filter(next_logits, top_k)
                
                # 转为概率并采样
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # 采样下一个 token
                
                # 拼接序列
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        self.train()  # 恢复训练模式
        return input_ids

    def _top_k_filter(self, logits, top_k):
        """保留 top_k 个最高概率的 token，其余设为负无穷"""
        values, _ = torch.topk(logits, top_k)
        min_val = values[:, -1].unsqueeze(1)  # 第 k 大的概率值
        return logits.masked_fill(logits < min_val, -1e10)


# -------------------------- 测试代码 --------------------------
if __name__ == "__main__":
    # 超参数（简化版，实际 GPT-3 为 1750 亿参数，这里用小参数演示）
    vocab_size = 50257  # GPT 原版词汇表大小
    d_model = 768       # 隐藏层维度（GPT-3 小版本用 768）
    n_layer = 6         # 解码器块数量（GPT-3 为 96，这里简化）
    n_head = 12         # 注意力头数（768/12=64，每个头 64 维）

    # 初始化模型
    model = GPT3(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer, n_head=n_head)
    print("模型结构初始化完成，参数数量：", sum(p.numel() for p in model.parameters())/1e6, "M")

    # 测试前向传播
    input_ids = torch.randint(0, vocab_size, (2, 10))  # 随机输入：2个样本，每个10个token
    logits = model(input_ids)
    print("前向传播输出形状：", logits.shape)  # 应输出 (2, 10, 50257)

    # 测试生成功能（模拟生成）
    prompt_ids = torch.tensor([[101, 102, 103]])  # 假设 101/102/103 是起始 token
    generated_ids = model.generate(prompt_ids, max_new_tokens=5)
    print("生成序列形状：", generated_ids.shape)  # 应输出 (1, 3+5=8)
    print("生成的 token 序列：", generated_ids[0].tolist())
