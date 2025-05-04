import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
import math

from utils import *

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
def get_model_size(model):
    torch.save(model.state_dict(), 'temp.pth')
    size = np.round(os.path.getsize('temp.pth') / (1024 * 1024), 2)  # in MB
    os.remove('temp.pth')
    return size
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2).squeeze()
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
class FiLM(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h
class FFN(nn.Module):
    """
    FFN block
    """
    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = FiLM(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + self.proj_out(y, emb)
        return y
class TemporalSelfAttention(nn.Module):
    """
    self-attention block
    """
    def __init__(self, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = FiLM(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, T, D
        key = self.key(self.norm(x)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        # B, T, T, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.norm(x)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        """ FiLM Modualtion"""
        y = x + self.proj_out(y, emb)
        return y
class MultiScaleTimeEmbeddingGenerator(nn.Module):
    def __init__(self, time_embed_dim):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        # 定义不同尺度的变换层
        self.short_term_transform = nn.Linear(time_embed_dim, time_embed_dim)
        self.mid_term_transform = nn.Linear(time_embed_dim, time_embed_dim)
        self.long_term_transform = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, emb):
        # 应用不同的变换以生成不同尺度的嵌入
        short_term_emb = F.relu(self.short_term_transform(emb))
        mid_term_emb = F.relu(self.mid_term_transform(emb))
        long_term_emb = F.relu(self.long_term_transform(emb))

        # 将不同尺度的嵌入合并为一个更丰富的时间嵌入
        # 这里我们简单地通过加权求和来合并这些嵌入，权重可以根据需要进行调整
        # 也可以考虑其他合并方式，如拼接(concatenation)等
        combined_emb = short_term_emb * 0.3 + mid_term_emb * 0.3 + long_term_emb * 0.4

        return combined_emb
class MyModule(nn.Module):
    """
    This is TranLinear Block related to paper
    """
    def __init__(self,
                 latent_dim=32,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.5,
                 multi_scale_time_embedding=False,  # 新增多尺度时间嵌入标志
                 gate_module=True  # 新增自适应特征融合标志
                 ):
        super().__init__()
        self.multi_scale_time_embedding = multi_scale_time_embedding
        self.gate_module = gate_module
        self.sa_block = TemporalSelfAttention(
            latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

        # 可选的多尺度时间嵌入模块
        if self.multi_scale_time_embedding:
            self.multi_scale_embedding_generator = MultiScaleTimeEmbeddingGenerator(time_embed_dim)

        # 可选的自适应特征融合模块
        if self.gate_module:
            self.feature_fusion_gate = nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.Sigmoid()
            )
    def forward(self, x, emb):
        if self.multi_scale_time_embedding:
            emb = self.multi_scale_embedding_generator(emb)

        attention_output = self.sa_block(x, emb)
        ffn_output = self.ffn(attention_output, emb)

        if self.gate_module:
            bias = self.feature_fusion_gate(torch.cat([attention_output, ffn_output], dim=-1))
            x = bias * attention_output + (1 - bias) * ffn_output
        else:
            x = ffn_output

        return x
class MyModel(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
                 activation="gelu",
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)

        self.cond_embed = nn.Linear(self.input_feats * self.num_frames, self.time_embed_dim)
        # "原來的時間步嵌入模塊"
        # self.time_embed = nn.Sequential(
        #     nn.Linear(self.latent_dim, self.time_embed_dim),
        #     nn.SiLU(),
        #     nn.Linear(self.time_embed_dim, self.time_embed_dim),
        # )

        """更換原來的時間步嵌入模塊"""
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                MyModule(
                    latent_dim=latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout,
                )
            )

        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))

    def forward(self, x, timesteps, mod=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]
        # print('shape', x.shape)  [64,20,48]
        """
            原來的時間步嵌入模塊
            # emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))
        """
        # emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))
        emb = self.embed_timestep(timesteps) # 我的
        # print('emb.shape', emb.shape) [64, 512]
        if mod is not None:
            mod_proj = self.cond_embed(mod.reshape(B, -1))
            emb = emb + mod_proj


        # B, T, latent_dim
        h = self.joint_embed(x)
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]

        i = 0
        prelist = []
        for module in self.temporal_decoder_blocks:
            if i < (self.num_layers // 2):
                prelist.append(h)
                h = module(h, emb)
            elif i >= (self.num_layers // 2):
                h = module(h, emb)
                h += prelist[-1]
                prelist.pop()
            i += 1


        output = self.out(h).view(B, T, -1).contiguous()
        return output

if __name__ == '__main__':
    batch_size = 8
    model = MyModel(
        input_feats=48,  # 3 means x, y, z
        num_frames=125,
        num_layers=8,
        num_heads=8,
        latent_dim=512,
        dropout=0.2,
    )
    print(model)
    # x = torch.randn((8, 125, 48))
    # t = torch.randint(1000, (batch_size, ))
    # y = model(x, t)
    # print(y.shape)
    # model_size = get_model_size(model)
    # print(f"Model size: {model_size} MB")