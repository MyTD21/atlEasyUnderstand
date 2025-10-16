import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """专家网络：每个专家是一个简单的MLP，负责处理输入并输出特征"""
    def __init__(self, input_dim, hidden_dim, output_dim, activation=nn.GELU()):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: [batch_size, input_dim]
        return self.mlp(x)  # [batch_size, output_dim]


class Gating(nn.Module):
    """门控网络：预测每个输入对应的专家权重，并选择Top-K专家"""
    def __init__(self, input_dim, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(input_dim, num_experts)  # 输出每个专家的原始分数

    def forward(self, x):
        # x: [batch_size, input_dim]
        logits = self.gate(x)  # [batch_size, num_experts]：每个专家的原始分数
        weights = F.softmax(logits, dim=-1)  # [batch_size, num_experts]：归一化权重
        
        # 选择Top-K专家的权重和索引
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)  # 均为 [batch_size, top_k]
        # 重新归一化Top-K权重（确保和为1）
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return top_k_weights, top_k_indices


class MoE(nn.Module):
    """混合专家模型：通过门控选择专家，加权融合输出"""
    def __init__(self, input_dim, output_dim, num_experts, top_k, expert_hidden_dim):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 初始化专家网络（所有专家共享输入/输出维度）
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dim, output_dim) 
            for _ in range(num_experts)
        ])
        
        # 初始化门控网络
        self.gating = Gating(input_dim, num_experts, top_k)

    def forward(self, x):
        # x: [batch_size, input_dim]
        batch_size, input_dim = x.shape
    
        # 1. 门控网络预测Top-K专家
        top_k_weights, top_k_indices = self.gating(x)  # [batch_size, top_k], [batch_size, top_k]
        
        # 2. 收集选中的专家输出（仅计算Top-K专家，节省计算）
        # 初始化输出容器：[batch_size, top_k, output_dim]
        expert_outputs = torch.zeros(batch_size, self.top_k, output_dim, device=x.device)
        
        for i in range(self.top_k):
            # 获取第i个选中的专家索引（[batch_size]）
            expert_idx = top_k_indices[:, i]
            # 对每个样本，选择对应的专家并计算输出
            for b in range(batch_size):
                expert_outputs[b, i] = self.experts[expert_idx[b]](x[b])
        
        # 3. 加权融合专家输出：[batch_size, top_k] × [batch_size, top_k, output_dim] → [batch_size, output_dim]
        moe_output = torch.sum(top_k_weights.unsqueeze(-1) * expert_outputs, dim=1)
        
        return moe_output


# 测试代码
if __name__ == "__main__":
    # 配置参数
    input_dim = 64        # 输入特征维度
    output_dim = 64        # 输出特征维度
    num_experts = 8       # 专家数量
    top_k = 2             # 每个输入激活的专家数量
    expert_hidden_dim = 128  # 专家网络隐藏层维度
    batch_size = 1        # 批量大小

    # 初始化MoE模型
    moe = MoE(input_dim, output_dim, num_experts, top_k, expert_hidden_dim)
    
    # 随机生成输入
    x = torch.randn(batch_size, input_dim)
    
    # 前向传播
    output = moe(x)
    
    # 验证输出维度
    print(f"输入形状: {x.shape}")          # [4, 64]
    print(f"输出形状: {output.shape}")      # [4, 64]（与输入维度匹配，符合残差连接需求）
    print("MoE模型前向传播成功！")
