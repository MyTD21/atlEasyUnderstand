import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch import Tensor


# -------------------------- 配置参数 --------------------------
class Config:
    sample_rate = 16000  # 音频采样率
    n_mels = 80  # 梅尔频谱维度
    win_length = 400  # 窗长（25ms @16kHz）
    hop_length = 160  # 步长（10ms @16kHz）
    hidden_dim = 128  # 编码器隐藏维度
    num_layers = 2  # Conformer层数（简化版）
    num_heads = 4  # 注意力头数
    codebook_size = 1024  # 码本大小
    num_codebooks = 2  # 多码本数量
    mask_prob = 0.15  # 掩码比例
    mask_length = 4  # 连续掩码帧数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()


# -------------------------- 1. 梅尔频谱提取 --------------------------
class MelFeatureExtractor(nn.Module):
    """将原始音频转换为梅尔频谱图"""
    def __init__(self, config):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.win_length,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            # 移除可能不兼容的normalized参数
        )

    def forward(self, wav: Tensor) -> Tensor:
        """输入: 原始音频 [batch, time]；输出: 梅尔频谱 [batch, n_mels, time_steps]"""
        mel = self.mel_transform(wav)  # [batch, n_mels, time_steps]
        return mel.log1p()  # 取对数增强鲁棒性


# -------------------------- 2. 随机投影量化器 --------------------------
class RandomProjectionQuantizer(nn.Module):
    """随机投影量化器：生成离散目标标记（不训练）"""
    def __init__(self, config):
        super().__init__()
        self.n_mels = config.n_mels
        self.codebook_size = config.codebook_size
        self.num_codebooks = config.num_codebooks
        
        # 随机投影矩阵（固定，不训练）
        self.proj = nn.Parameter(
            torch.randn(config.num_codebooks, config.n_mels, 64, device=config.device),  # 显式指定设备
            requires_grad=False
        )
        
        # 随机码本（固定，不训练）
        self.codebooks = nn.Parameter(
            torch.randn(config.num_codebooks, config.codebook_size, 64, device=config.device),  # 显式指定设备
            requires_grad=False
        )

    def forward(self, mel: Tensor) -> Tensor:
        """输入: 梅尔频谱 [batch, n_mels, time_steps]；输出: 量化标记 [num_codebooks, batch, time_steps]"""
        batch, _, time_steps = mel.shape
        mel = mel.transpose(1, 2)  # [batch, time_steps, n_mels]
       
        all_codes = []
        for c in range(self.num_codebooks):
            # 随机投影
            # 作用是降维，为后续生成离散表征做准备；
            # 过程是通过一个矩阵乘，将([2, 101, 80])转为([2, 101, 64])
            proj = torch.matmul(mel, self.proj[c]) 
            proj = F.normalize(proj, dim=-1)
            codebook = F.normalize(self.codebooks[c], dim=-1)

            # 输入proj是经过规整后的音频低纬表征，
            # 输入codebook.T是规整后的码本做矩阵
            # 两者相乘，输出的是第b个样本，第t时间帧的特征和码本里每个码向量的相似度
            sim = torch.matmul(proj, codebook.T)

            # 第 b 个样本、第 t 个时间帧的音频特征，与码本中第 k 个码向量最相似
            # 第 b 个样本、第 t 个时间帧的音频特征，对应一个label，用来标识这帧音频像什么
            codes = sim.argmax(dim=-1)  # 最佳匹配索引
            all_codes.append(codes)
        
        return torch.stack(all_codes, dim=0)


# -------------------------- 3. 简化版Conformer编码器 --------------------------
# cnn, 获取局部特征
# attention，获得全局感知能力
# fnn，提升模型的表达能力
class ConformerBlock(nn.Module):
    """简化的Conformer块：卷积+注意力+前馈网络"""
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),  # 深度卷积
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),  # 点卷积
            nn.GELU(),
            # 移除这里的LayerNorm，移到transpose之后
        )
        self.conv_norm = nn.LayerNorm(hidden_dim)  # 新增：单独定义LayerNorm
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """输入: [batch, time_steps, hidden_dim]；输出: 同输入形状"""
        # 卷积分支
        residual = x
        x_conv = x.transpose(1, 2)  # [batch, hidden_dim, time_steps]
        x_conv = self.conv(x_conv).transpose(1, 2)  # [batch, time_steps, hidden_dim]
        x = residual + 0.5 * x_conv
        
        # 注意力分支
        residual = x
        x_attn, _ = self.attn(x, x, x)
        x = residual + x_attn
        
        # 前馈分支
        residual = x
        x = residual + self.ffn(x)
        return self.norm(x)

# 获取音频表征
class ConformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.n_mels, config.hidden_dim)
        self.layers = nn.ModuleList([
            ConformerBlock(config.hidden_dim, config.num_heads)
            for _ in range(config.num_layers)
        ])

    def forward(self, mel: Tensor) -> Tensor:
        """输入: 梅尔频谱 [batch, n_mels, time_steps]；输出: 编码特征 [batch, time_steps, hidden_dim]"""
        x = mel.transpose(1, 2)  # [batch, time_steps, n_mels]
        x = self.proj(x)  # 映射到隐藏维度
        for layer in self.layers:
            x = layer(x)
        return x


# -------------------------- 4. 掩码策略 --------------------------
# 对标记为掩码区域的部分用随机噪声替换，未被掩码部分保留原始的mel特征
def apply_mask(mel: Tensor, config) -> (Tensor, Tensor):  # 修正类型注解
    """输入: 梅尔频谱；输出: 掩码后的梅尔频谱、掩码标记"""
    batch, n_mels, time_steps = mel.shape
    mask = torch.zeros(batch, time_steps, device=config.device)  # 显式指定设备
   
    for b in range(batch):
        num_masks = max(1, int(config.mask_prob * time_steps))  # 确保至少有1个掩码
        for _ in range(num_masks):
            # 避免start越界（确保mask_length不超过time_steps）
            max_start = max(0, time_steps - config.mask_length)
            start = torch.randint(0, max_start + 1, (1,), device=config.device).item()
            mask[b, start:start+config.mask_length] = 1
    
    noise = torch.randn_like(mel, device=config.device) * 0.1  # 噪声显式指定设备
    mel_masked = mel * (1 - mask.unsqueeze(1)) + noise * mask.unsqueeze(1)
    return mel_masked, mask


# -------------------------- 5. BEST-RQ完整模型 --------------------------
class BEST_RQ(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mel_extractor = MelFeatureExtractor(config)
        self.quantizer = RandomProjectionQuantizer(config)
        self.encoder = ConformerEncoder(config)
        self.predictor = nn.Linear(
            config.hidden_dim,
            config.num_codebooks * config.codebook_size
        )

    def forward(self, wav: Tensor) -> (Tensor, Tensor):  # 修正类型注解
        """输入: 原始音频 [batch, time]；输出: 损失、掩码标记"""
        mel = self.mel_extractor(wav)  # 梅尔频谱
        target_codes = self.quantizer(mel)  # 目标量化标记
        mel_masked, mask = apply_mask(mel, config)  # 掩码
        enc_feat = self.encoder(mel_masked)  # 编码特征
        
        # 预测量化标记
        # 将编码器输出的音频表征投影到 量化标记的预测空间
        logits = self.predictor(enc_feat)
        logits = logits.view(
            logits.shape[0], logits.shape[1], config.num_codebooks, config.codebook_size
        )
        
        # 计算损失
        loss = 0.0
        for c in range(config.num_codebooks):
            target = target_codes[c]
            pred = logits[:, :, c, :]
            loss += F.cross_entropy(
                pred.transpose(1, 2),  # [batch, codebook_size, time_steps]
                target,
                reduction="none"
            ).masked_select(mask.bool()).mean()
        
        return loss, mask

    def inference(self, wav: Tensor, return_enc_feat: bool = False, return_codes: bool = True) -> dict:
        """
        推理函数：用于训练完成后，对输入音频进行特征提取或量化标记生成
        推理阶段不进行掩码和损失计算，仅输出有用的特征或标记

        参数:
            wav: 输入原始音频，形状为 [batch, time]（单通道音频）
            return_enc_feat: 是否返回编码器输出的高级音频表征
            return_codes: 是否返回原始梅尔频谱对应的量化标记

        返回:
            字典，包含以下可选键值对：
                - 'codes': 量化标记，形状为 [num_codebooks, batch, time_steps]（若return_codes=True）
                - 'enc_feat': 编码器输出特征，形状为 [batch, time_steps, hidden_dim]（若return_enc_feat=True）
                - 'mel': 提取的梅尔频谱，形状为 [batch, n_mels, time_steps]
        """
        self.eval()

        with torch.no_grad():
            mel = self.mel_extractor(wav)  # [batch, n_mels, time_steps]
            result = {'mel': mel}  # 始终返回梅尔频谱，作为基础特征

            # 生成原始梅尔频谱对应的量化标记（可选）
            if return_codes:
                codes = self.quantizer(mel)  # [num_codebooks, batch, time_steps]
                result['codes'] = codes

            # 生成编码器输出的高级音频表征（可选）
            # 注意：推理时使用完整梅尔频谱作为输入（不掩码），更符合实际应用场景
            if return_enc_feat:
                # enc_feat 是通用的核心表征：适合大多数需要 “音频语义理解” 的下游任务（如识别、分类），是模型学到的最有价值的抽象特征;
                enc_feat = self.encoder(mel)  # [batch, time_steps, hidden_dim]
                result['enc_feat'] = enc_feat

        self.train()

        return result

# -------------------------- 测试代码 --------------------------
if __name__ == "__main__":
    # 初始化模型并移到设备
    model = BEST_RQ(config).to(config.device)
    
    # 生成随机音频（1秒，16kHz）
    wav = torch.randn(2, 16000, device=config.device)  # [batch=2, time=16000]
    
    # 前向传播
    model.train()
    loss, mask = model(wav)
    
    print(f"train 损失值: {loss.item():.4f}")
    print(f"train 掩码标记形状: {mask.shape}")  # 预期 (2, 100)，因为16000/160=100帧

    # 推理：获取量化标记和编码器特征
    result = model.inference(wav, return_enc_feat=True, return_codes=True)

    print("inference 梅尔频谱形状：", result['mel'].shape)          # [2, 80, 100]
    print("inference 量化标记形状：", result['codes'].shape)         # [2, 2, 100]（num_codebooks=2）
