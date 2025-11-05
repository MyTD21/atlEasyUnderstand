import torch
import torch.nn as nn

# -------------------------- 1. 扩散参数（适中配置） --------------------------
class DiffusionConfig:
    T = 50  # 扩散步数
    beta_start = 0.0001
    beta_end = 0.02
    img_size = 16  # 图像尺寸
    channels = 1   # 单通道
    dim = 32       # 隐藏维度

config = DiffusionConfig()

# 预计算扩散系数（在CPU上）

# torch.linspace，构造等差数列；torch.logspace，构造等比数列；
# β，噪声强度参数，表示每一步正向扩散中 “添加噪声的比例”；
# 使用等差数列表示，随着步数的增加，噪声越来越多；
beta = torch.linspace(config.beta_start, config.beta_end, config.T, device="cpu")

# α, 信号保留比例;表示每一步正向扩散中 “原始信号未被噪声污染的比例”;
alpha = 1. - beta

# alpha_bar, 累积信号保留比例,表示经过 t 步正向扩散后，原始图像 x0 中 “未被噪声污染的总信号比例”
# alpha_bar[t] = alpha[0] * alpha[1] * ... * alpha[t]
# torch.cumprod, 计算张量沿指定维度的累积乘积
alpha_bar = torch.cumprod(alpha, dim=0)

# 第 t-1 步的累积信号保留比例,就是在alpha_bar数组前边插入一个1
alpha_bar_prev = torch.cat([torch.tensor([1.], device="cpu"), alpha_bar[:-1]])

# 强制使用CPU
device = torch.device("cpu")  # 直接指定为CPU，不检测GPU

# -------------------------- 2. 改进版U-Net --------------------------
class TimeEmbedding(nn.Module):
    """时间步嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return self.layers(emb)


class SmallUNet(nn.Module):
    """修复通道维度匹配问题的U-Net"""
    def __init__(self, config):
        super().__init__()
        self.channels = config.channels
        self.dim = config.dim
        self.time_emb = TimeEmbedding(config.dim * 2)  # 时间嵌入维度与x2通道匹配

        # 下采样路径
        self.down1 = nn.Sequential(  # (B, C, 16, 16) → (B, dim, 8, 8)
            nn.Conv2d(config.channels, config.dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.dim, config.dim, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.down2 = nn.Sequential(  # (B, dim, 8, 8) → (B, dim*2, 4, 4)
            nn.Conv2d(config.dim, config.dim*2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.dim*2, config.dim*2, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )

        # 瓶颈层
        self.bottleneck = nn.Sequential(  # (B, dim*2, 4, 4) → (B, dim*2, 4, 4)
            nn.Conv2d(config.dim*2, config.dim*4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.dim*4, config.dim*2, 3, padding=1),
            nn.ReLU()
        )

        # 上采样路径（适配跳跃连接）
        self.up1 = nn.Sequential(  # 输入: (B, dim*4, 4, 4) → 输出: (B, dim, 8, 8)
            nn.ConvTranspose2d(config.dim*4, config.dim, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(config.dim, config.dim, 3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(  # 输入: (B, dim*2, 8, 8) → 输出: (B, channels, 16, 16)
            nn.ConvTranspose2d(config.dim*2, config.channels, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(config.channels, config.channels, 3, padding=1),
            nn.ReLU()
        )

        # 输出层
        self.out_conv = nn.Conv2d(config.channels, config.channels, 1)

    def forward(self, x, t):
        t_emb = self.time_emb(t)  # (B, dim*2)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # (B, dim*2, 1, 1)

        # 下采样
        x1 = self.down1(x)  # (B, dim, 8, 8)
        x2 = self.down2(x1)  # (B, dim*2, 4, 4)

        # 瓶颈层 + 时间信息融合
        x = self.bottleneck(x2) + t_emb

        # 上采样 + 跳跃连接
        x = torch.cat([x, x2], dim=1)  # (B, dim*4, 4, 4)
        x = self.up1(x)  # (B, dim, 8, 8)
        x = torch.cat([x, x1], dim=1)  # (B, dim*2, 8, 8)
        x = self.up2(x)  # (B, channels, 16, 16)

        return self.out_conv(x)


# -------------------------- 3. 正向扩散 --------------------------
def forward_diffusion(x0, t):
    # torch.randn_like, 用于生成与给定输入张量形状相同、但元素值服从标准正态分布（均值为 0，标准差为 1）的新张量
    # ε, 表示高斯噪声，是正向扩展时候想原始数据中加入的污染源；
    eps = torch.randn_like(x0, device=device)  # 噪声在CPU上生成
    alpha_bar_t = alpha_bar[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # x0,表示原始数据；eps，表示高斯噪声
    # xt是带噪图像，是由原始数据和高斯噪声根据一定比例混合而成的
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps
    return xt, eps


# -------------------------- 4. 反向扩散 --------------------------
def reverse_diffusion(model, xt, t):
    with torch.no_grad():
        eps_pred = model(xt, t) # 用模型预测当前图像xt中的噪声,模型输出的是对噪声的估计
        alpha_t = alpha[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # t步时信号保留比例
        alpha_bar_t = alpha_bar[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # t步积累信号保留比例
        alpha_bar_prev_t = alpha_bar_prev[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # t - 1步积累信号保留比例
        
        # 从xt计算xt_prev
        # (1 - alpha_t) 第 t 步单独添加的噪声比例
        # sqrt(1 - alpha_bar_t) 是前 t 步累积的总噪声的标准差
        # (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)，当前步噪声在累计噪声中占比, 简称nsf(Noise scaling factor)
        # eps_pred, 是模型预测的、从原始图像 x0 到第 t 步图像 xt 累积的总噪声 
        # nsf * eps_pred, 第 t 步正向扩散时单独添加的噪声分量
        # xt, 是第 t 步的带噪声图像
        # (xt - ndf * eps_pred)，从当前带噪声图像 xt 中，减去第 t 步添加的噪声
        # torch.sqrt(alpha_t)是正向扩散中对原始信号的衰减，除以他，就是把信号恢复到t-1步应有的信号幅度
        xt_prev = (1 / torch.sqrt(alpha_t)) * (xt - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_pred)
        
        if t[0] > 0: # 若不是最后一步（t>0），添加少量随机噪声
            z = torch.randn_like(xt, device=device)  # 噪声在CPU上生成
            # 噪声强度系数，用于控制反向扩散中添加的噪声幅度。其大小随 t 变化：
            #   当 t 较大（接近纯噪声）时，sigma_t 较大（需要保留更多随机性）；
            #   当 t 较小（接近清晰图像）时，sigma_t 较小（避免噪声破坏细节）;
            
            sigma_t = torch.sqrt((1 - alpha_bar_prev_t) / (1 - alpha_bar_t) * (1 - alpha_t))
            xt_prev += sigma_t * z # 让反向过程符合正向扩散的概率特性，同时保证生成结果的多样性;

        return xt_prev

# -------------------------- 5. 完整流程 --------------------------
def full_diffusion_flow():
    # 模型在CPU上创建
    model = SmallUNet(config).to(device)
    batch_size = 2
    # 原始图像张量在CPU上创建
    x0 = torch.randn(
        batch_size, config.channels, config.img_size, config.img_size, device=device
    )
    print(f"原始图像x0形状: {x0.shape}")

    print("\n=== 正向扩散（加噪）过程 ===")
    for t in [0, 10, 25, 49]:
        # 时间步张量在CPU上创建
        t_tensor = torch.tensor([t]*batch_size, device=device)
        xt, eps = forward_diffusion(x0, t_tensor)
        print(f"t={t}时，带噪声图像xt形状: {xt.shape}，噪声eps形状: {eps.shape}")

    t_final = config.T - 1
    xt = forward_diffusion(x0, torch.tensor([t_final]*batch_size, device=device))[0]
    print(f"\n反向扩散起点（t={t_final}的噪声）形状: {xt.shape}")

    print("\n=== 反向扩散（去噪）过程 ===")
    for t in range(config.T-1, -1, -1):
        t_tensor = torch.tensor([t]*batch_size, device=device)
        xt = reverse_diffusion(model, xt, t_tensor)
        if t % 10 == 0:
            print(f"反向扩散t={t}后，张量形状: {xt.shape}")

    print(f"\n最终恢复结果形状: {xt.shape}")
    print("完整流程（正向+反向）运行完成，无张量维度错误（CPU模式）")


if __name__ == "__main__":
    full_diffusion_flow()
