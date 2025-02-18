import torch
from einops import rearrange, repeat
from torch import nn, einsum, softmax
import math
import torch.nn.functional as F

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim),
                                 GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)               
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                                              Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x
    

###########################################################
############# output = mean of the all tokens #############
###########################################################
class Temporal_Transformer_Mean(nn.Module):
    def __init__(self, num_patches, input_dim, depth, heads, mlp_dim, dim_head):
        super().__init__()
        dropout=0.0
        self.num_patches = num_patches
        self.input_dim = input_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, input_dim))
        self.temporal_transformer = Transformer(input_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        x = x.contiguous().view(-1, self.num_patches, self.input_dim)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.temporal_transformer(x)
        x = x.mean(dim=1)
        return x

###########################################################
#############      output = class tokens      #############
###########################################################
class Temporal_Transformer_Cls(nn.Module):
    def __init__(self, num_patches, input_dim, depth, heads, mlp_dim, dim_head):
        super().__init__()
        dropout=0.1
        self.num_patches = num_patches
        self.input_dim = input_dim
        self.layernorm = nn.LayerNorm(input_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim)) #[1, 1, 512]
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, input_dim))
        self.temporal_transformer = Transformer(input_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        b, n, _ = x.shape #[B, T, 512]
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b) #[1, 1, 512] -> [B, 1, 512]
        x = torch.cat((cls_tokens, x), dim=1) #[B, T + 1, 512]
        # x = x + self.pos_embedding[:, :(n+1)]
        x = self.layernorm(x + self.pos_embedding[:, :(n+1)])
        x = self.temporal_transformer(x)
        x = x[:, 0] #[B, T+1, 512]
        return x
    
###########################################################
#############        output = all tokens      #############
###########################################################
class Temporal_Transformer_All(nn.Module):
    def __init__(self, num_patches, input_dim, depth, heads, mlp_dim, dim_head):
        super().__init__()
        dropout=0.0
        self.num_patches = num_patches
        self.input_dim = input_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, input_dim))
        self.temporal_transformer = Transformer(input_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        x = x.contiguous().view(-1, self.num_patches, self.input_dim)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.temporal_transformer(x)
        return x

class OUTPUTMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)  # L2 归一化
        weight_norm = F.normalize(self.weight, p=2, dim=1)  # 归一化权重
        s = torch.matmul(x_norm, weight_norm.t())  # 计算得分
        return s #s的取值范围是[-1, 1]


class Chomp1d(nn.Module):
    """
    这一层用于确保卷积输出的长度与输入长度相同。
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    """
    TCN的一个基本块，包含一个因果卷积层和一个非线性激活函数。
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    

class TemporalConvNet(nn.Module):
    """
    TCN模型，包含多个TemporalBlock。
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.0):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        # Ensure the output of the last layer is the same as the input feature dimension
        self.last_conv = nn.Conv1d(num_channels[-1], num_inputs, 1) if num_channels[-1] != num_inputs else None

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)  # Convert (B, T, 49) to (B, 49, T)
        x = self.network(x)
        if self.last_conv is not None:
            x = self.last_conv(x)
        return x.transpose(1, 2)  # Convert back to (B, T, 49)
    



class GAP_Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, mlp_dim, depth, heads, dim_head, t=16,dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), #[256]
                                 GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, mlp_dim),
                                 )


        self.pos_embedding = nn.Parameter(torch.randn(1, t, mlp_dim))#位置嵌入
        self.temporal_transformer = Transformer(mlp_dim, depth, heads, dim_head, mlp_dim,dropout) #[512] -> [512]
    #[B, T, 49] -> [B, T, 512]
    def forward(self, x):
        x = self.mlp(x)
        b, t, _ = x.shape
        x = x + self.pos_embedding[:, :t]

        x = self.temporal_transformer(x)
        return x

class GAP_TCN(nn.Module):
    def __init__(self, num_inputs=49, hidden_dim=256, mlp_dim=512, num_channels=[25, 50, 100], kernel_size=3, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(num_inputs, hidden_dim), #[256]
                                 GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, mlp_dim),
                                 )
        
        self.tcn = TemporalConvNet(num_inputs=mlp_dim, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x):
        x = self.mlp(x)

        return self.tcn(x)


'''
test
'''
#[B, T, 512]
# gap_model = GAP_Transformer(input_dim=49, hidden_dim=256, mlp_dim=512, depth=1, heads=8, dim_head=64, t=16, dropout=0.)
# x = torch.randn(8, 16, 49) #[B, T, 49]
# x = gap_model(x) #[B, T, C]
# #CLIP: [B, 1, 512]
# clip_features = torch.randn(8, 1, 512) #[B, 1, C]
# #每个时间维度上信息重要程度不同，因此针对时间进行attention
# x = x.transpose(1, 2) #[B, C, T]
# attn_score = torch.matmul(clip_features, x)# [B, 1, C] @ [B, C, T] -> [B, 1, T]
# attn_score = softmax(attn_score, dim=-1) #[B, 1, T]
# #[B, T, C]
# #[B, C, T] @ [B, T, 1] -> [B, C, 1]
# x = x @ attn_score.transpose(1, 2)
# x = x.transpose(1, 2) #[B, 1, C]
# x = torch.concat([x, clip_features], dim=-1) #[B, 1, 2*C]
# B, _, _C = x.shape
# C = _C // 2
# net = nn.Sequential(nn.Linear(_C, C),
#                          GELU(),
#                          nn.Dropout(0),
#                          nn.Linear(C, C),
#                          nn.Dropout(0))
# x = net(x) #[B, 1, C]
# print(f'x.shape{x.shape}')

# num_inputs = 49
# num_channels = [25, 50, 100]
# kernel_size = 3 #卷积核大小
# dropout = 0.2

# model =  TemporalConvNet(num_inputs,num_channels, kernel_size, dropout)

# intput_data = torch.randn(32, 100, 49)
# output = model(intput_data)

# print(output)
# print(output.shape)
