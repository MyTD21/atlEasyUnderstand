import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------- 配置参数 --------------------------
class Config:
    img_size = 32        # 图像尺寸（32x32）
    patch_size = 4       # 分块大小（4x4像素/块）
    hidden_dim = 128     # 隐藏层维度
    num_heads = 4        # 注意力头数
    num_layers = 3       # Transformer层数
    text_dim = 64        # 文本嵌入维度（简化）
    T = 100              # 扩散步数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# 预计算扩散系数
beta = torch.linspace(0.0001, 0.02, config.T, device=config.device)
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha, dim=0)


# -------------------------- 核心组件 --------------------------
class PatchEmbed(nn.Module):
    """图像分块与嵌入"""
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) **2
        # 每个patch通过卷积投影为向量
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        import pdb; pdb.set_trace()
        # x: [batch, 3, H, W]
        x = self.proj(x)  # [batch, embed_dim, H/patch, W/patch]
        x = x.flatten(2)  # [batch, embed_dim, num_patches]
        return x.transpose(1, 2)  # [batch, num_patches, embed_dim]


class TimeEmbed(nn.Module):
    """时间步嵌入（正弦余弦编码）"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t):
        # t: [batch]
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=config.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=config.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)  # [batch, half_dim]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)  # [batch, dim]
        return self.mlp(emb)  # [batch, dim]


class CrossAttention(nn.Module):
    """文本-图像交叉注意力（条件生成用）"""
    def __init__(self, hidden_dim, text_dim, num_heads):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(text_dim, hidden_dim)
        self.v_proj = nn.Linear(text_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, x, text_emb):
        # x: [batch, num_patches, hidden_dim]（图像特征）
        # text_emb: [batch, text_dim]（文本嵌入，扩展为序列长度1）
        text_emb = text_emb.unsqueeze(1)  # [batch, 1, text_dim]
        q = self.q_proj(x)
        k = self.k_proj(text_emb)
        v = self.v_proj(text_emb)
        attn_out, _ = self.attn(q, k, v)  # 图像关注文本
        return self.out_proj(attn_out) + x  # 残差连接


class TransformerBlock(nn.Module):
    """DiT的Transformer块（自注意力+交叉注意力+前馈网络）"""
    def __init__(self, hidden_dim, text_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = CrossAttention(hidden_dim, text_dim, num_heads)  # 条件注意力
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, text_emb):
        # 自注意力（图像内部关联）
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # 交叉注意力（文本引导）
        x = x + self.cross_attn(self.norm2(x), text_emb)
        # 前馈网络
        x = x + self.mlp(self.norm3(x))
        return x


class DiT(nn.Module):
    """Diffusion Transformer主模型"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.hidden_dim
        )
        self.time_embed = TimeEmbed(config.hidden_dim)
        self.text_proj = nn.Linear(config.text_dim, config.hidden_dim)  # 文本嵌入投影
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.hidden_dim, config.text_dim, config.num_heads)
            for _ in range(config.num_layers)
        ])
        # 输出层：预测噪声（与输入图像块维度匹配）
        self.head = nn.Linear(
            config.hidden_dim,
            config.patch_size** 2 * 3  # 每个patch对应3通道（RGB）的噪声
        )

    def forward(self, x, t, text_emb):
        # x: [batch, 3, H, W]（带噪声图像）
        # t: [batch]（时间步）
        # text_emb: [batch, text_dim]（文本嵌入）

        import pdb; pdb.set_trace()
        # 1. 图像分块嵌入
        x = self.patch_embed(x)  # [batch, num_patches, hidden_dim]
        num_patches = x.shape[1]

        # 2. 时间嵌入与文本嵌入融合
        t_emb = self.time_embed(t)  # [batch, hidden_dim]
        text_emb_proj = self.text_proj(text_emb)  # [batch, hidden_dim]
        x = x + t_emb.unsqueeze(1) + text_emb_proj.unsqueeze(1)  # 广播到所有patch

        # 3. Transformer块序列
        for block in self.transformer_blocks:
            x = block(x, text_emb)

        # 4. 预测噪声（还原为图像形状）
        noise_pred = self.head(x)  # [batch, num_patches, patch_size^2 * 3]
        # 重组为图像：[batch, num_patches, 3, patch_size, patch_size]
        noise_pred = noise_pred.view(
            -1, num_patches, 3, self.config.patch_size, self.config.patch_size
        )
        # 拼接为完整图像：[batch, 3, H, W]
        H = W = self.config.img_size // self.config.patch_size
        noise_pred = noise_pred.permute(0, 2, 1, 3, 4)  # [batch, 3, num_patches, p, p]
        noise_pred = noise_pred.reshape(
            -1, 3, H * self.config.patch_size, W * self.config.patch_size
        )
        return noise_pred


# -------------------------- 扩散过程 --------------------------
def forward_diffusion(x0, t, eps=None):
    """正向扩散：给清晰图像x0加噪声，得到xt"""
    if eps is None:
        eps = torch.randn_like(x0)
    alpha_bar_t = alpha_bar[t].unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, 1]
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps
    return xt, eps


def reverse_diffusion(model, xt, t, text_emb):
    """反向扩散：从xt去噪一步得到xt-1"""
    alpha_t = alpha[t].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    alpha_bar_t = alpha_bar[t].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    eps_pred = model(xt, t, text_emb)
    # 计算xt-1
    xt_prev = (1 / torch.sqrt(alpha_t)) * (xt - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_pred)
    if t[0] > 0:
        beta_t = beta[t].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        xt_prev += torch.sqrt(beta_t) * torch.randn_like(xt)
    return xt_prev


# -------------------------- 训练与生成示例 --------------------------
if __name__ == "__main__":
    # 1. 初始化模型和优化器
    model = DiT(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 2. 模拟数据集（随机图像+随机文本嵌入）
    class DummyDataset(Dataset):
        def __len__(self):
            return 1000
        def __getitem__(self, idx):
            img = torch.randn(3, config.img_size, config.img_size)  # 模拟标准化图像
            text_emb = torch.randn(config.text_dim)  # 模拟文本嵌入
            return img, text_emb

    dataloader = DataLoader(DummyDataset(), batch_size=8, shuffle=True)

    # 3. 训练（简化版，仅演示流程）
    model.train()
    for epoch in range(3):
        for imgs, text_embs in dataloader:
            imgs = imgs.to(config.device)
            text_embs = text_embs.to(config.device)
            t = torch.randint(0, config.T, (imgs.shape[0],), device=config.device)  # 随机时间步
            xt, eps = forward_diffusion(imgs, t)  # 正向扩散
            eps_pred = model(xt, t, text_embs)    # 模型预测噪声
            loss = F.mse_loss(eps_pred, eps)      # 损失：预测噪声 vs 真实噪声

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # 4. 生成图像（文本引导）
    model.eval()
    with torch.no_grad():
        batch_size = 1
        text_emb = torch.randn(batch_size, config.text_dim, device=config.device)  # 文本嵌入（可替换为真实文本编码）
        xt = torch.randn(batch_size, 3, config.img_size, config.img_size, device=config.device)  # 初始噪声

        # 反向扩散迭代去噪
        for t in reversed(range(config.T)):
            t_tensor = torch.tensor([t]*batch_size, device=config.device)
            xt = reverse_diffusion(model, xt, t_tensor, text_emb)

        # 输出生成的图像（范围调整到[0, 255]）
        generated_img = (xt.clamp(-1, 1) + 1) / 2 * 255
        print(f"生成图像形状: {generated_img.shape}")  # [1, 3, 32, 32]
