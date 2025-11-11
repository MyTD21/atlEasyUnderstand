import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np

# 通过1D卷积操作，对原始wav进行局部时序特征提取
# 输出局部声学特征
class FeatureExtractor(nn.Module):
    """语音特征提取器：将原始波形转换为声学特征"""
    def __init__(self, input_dim=1, feat_dim=80, kernel_size=3, stride=4):
        super().__init__()
        self.conv = nn.Conv1d(
            input_dim, feat_dim, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, time]
        x = self.conv(x)    # [batch, feat_dim, time_steps]
        return x.transpose(1, 2)  # [batch, time_steps, feat_dim]

# 纯transformer模块
# 输入，多头注意力（含残差，layer normal），fnn（含残差，layer normal），输出
class TransformerEncoder(nn.Module):
    """Transformer编码器：建模语音时序依赖"""
    def __init__(self, feat_dim=80, hidden_dim=128, num_layers=2, num_heads=4):
        super().__init__()
        self.proj = nn.Linear(feat_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=4*hidden_dim,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.proj(x)  # [batch, time_steps, hidden_dim]
        return self.transformer(x)


class KMeansQuantizer:
    """K-means量化器：生成声学单元伪标签"""
    def __init__(self, num_clusters=100):
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        self.is_fit = False

    def fit(self, features):
        self.kmeans.fit(features)
        self.is_fit = True

    # 输入局部时序特征 
    # 通过k-means算法得到离散量化表征
    def quantize(self, features):
        assert self.is_fit, "需先调用fit训练量化器"
        batch, time, dim = features.shape
        flat_feats = features.reshape(-1, dim).cpu().detach().numpy()
        codes = self.kmeans.predict(flat_feats)
        return torch.from_numpy(codes).reshape(batch, time).to(features.device)


class HuBERT(nn.Module):
    """HuBERT主模型：特征提取+编码+掩码预测+推理"""
    def __init__(self, feat_dim=80, hidden_dim=128, num_clusters=100, mask_prob=0.15):
        super().__init__()
        self.feature_extractor = FeatureExtractor(feat_dim=feat_dim)
        self.encoder = TransformerEncoder(feat_dim=feat_dim, hidden_dim=hidden_dim)
        self.predictor = nn.Linear(hidden_dim, num_clusters)
        self.mask_prob = mask_prob

    def generate_mask(self, x):
        batch, time = x.shape[0], x.shape[1]
        mask = torch.zeros(batch, time, device=x.device)
        for b in range(batch):
            num_masks = int(self.mask_prob * time)
            for _ in range(num_masks // 8):
                start = torch.randint(0, max(1, time - 8), (1,)).item()
                mask[b, start:start+8] = 1
        return mask.bool()

    def forward(self, wav, quantizer):
        """训练时前向传播（计算损失）"""
        feats = self.feature_extractor(wav)
        target_codes = quantizer.quantize(feats).long()
        mask = self.generate_mask(feats)
        feats_masked = feats.masked_fill(mask.unsqueeze(-1), 0.0)
        enc_feat = self.encoder(feats_masked)
        logits = self.predictor(enc_feat)
        loss = F.cross_entropy(logits[mask], target_codes[mask])
        return loss

    def inference(self, wav, quantizer=None, return_codes=False):
        """
        推理函数：提取语音高级表征（可选返回量化标记）
        Args:
            wav: 输入语音波形，形状 [batch, time]
            quantizer: 预训练的KMeansQuantizer（return_codes=True时必需）
            return_codes: 是否返回量化标记（离散化特征）
        Returns:
            enc_feat: 编码器输出的高级表征，形状 [batch, time_steps, hidden_dim]
            （可选）codes: 量化标记，形状 [batch, time_steps]
        """
        with torch.no_grad():  # 推理时不计算梯度
            # 1. 提取声学特征
            feats = self.feature_extractor(wav)  # [batch, time_steps, feat_dim]
            
            # 2. 编码器输出高级表征（核心结果）
            enc_feat = self.encoder(feats)  # [batch, time_steps, hidden_dim]
            
            # 3. 可选：生成量化标记
            if return_codes:
                assert quantizer is not None, "return_codes=True时需传入预训练的quantizer"
                codes = quantizer.quantize(feats).long()  # [batch, time_steps]
                return enc_feat, codes
            return enc_feat


# 测试代码（包含推理函数调用）
if __name__ == "__main__":
    wav = torch.randn(2, 2000)  # 随机波形（batch=2，长度2000）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav = wav.to(device)

    # 初始化模型和量化器
    hubert = HuBERT(num_clusters=100).to(device)
    quantizer = KMeansQuantizer(num_clusters=100)

    # 预训练量化器
    with torch.no_grad():
        feats = hubert.feature_extractor(wav)
        quantizer.fit(feats.reshape(-1, feats.shape[-1]).cpu().numpy())

    # 训练模型（模拟1轮）
    optimizer = torch.optim.Adam(hubert.parameters(), lr=1e-4)
    hubert.train()
    loss = hubert(wav, quantizer)
    loss.backward()
    optimizer.step()
    print(f"训练损失: {loss.item():.4f}")

    # 推理：提取高级表征
    hubert.eval()  # 切换到评估模式
    enc_feat = hubert.inference(wav)
    print(f"推理：编码器输出形状: {enc_feat.shape}")

    # 推理：同时提取表征和量化标记
    enc_feat, codes = hubert.inference(wav, quantizer=quantizer, return_codes=True)
    print(f"推理：量化标记形状: {codes.shape}")
