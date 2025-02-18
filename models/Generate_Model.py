from torch import nn
from torch.linalg import cross
from torch.nn.functional import dropout

from models.Temporal_Model import *
from models.Prompt_Learner import *


class ClipModel(nn.Module):
    def __init__(self, input_text, clip_model, args):
        super().__init__()
        self.args = args
        self.input_text = input_text  # 固定的文本提示 a smiling mouse ...
        self.prompt_learner = PromptLearner(input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual
        self.temporal_net = Temporal_Transformer_Cls(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)

        self.gap_net = GAP_Transformer(input_dim=56,
                                       hidden_dim=256,
                                       mlp_dim=512,
                                       depth=args.temporal_layers,
                                       heads=8,
                                       dim_head=64,
                                       t=args.t,
                                       dropout=0., )

        self.cross_net = nn.Sequential(nn.Linear(512 * 2, 512),
                                       GELU(),
                                       nn.Dropout(0.),
                                       nn.Linear(512, 512),
                                       nn.Dropout(0.))
        self.clip_model_ = clip_model

    def forward(self, image, gap):
        ################# Visual Part And GAP Part #################
        n, t, c, h, w = image.shape 
        image = image.contiguous().view(-1, c, h, w)  # [n*t, 3, 224, 224]
        image_features = self.image_encoder(image.type(self.dtype))  # [n*t, 512]
        # TODO: 目前是concat然后通过 MLP还原形状，尝试直接相加
        image_features = image_features.contiguous().view(n, t, -1)  # [n, t, 512]
        video_features = self.temporal_net(image_features)  # [n, 512] nan
        gap_features = self.gap_net(gap.type(self.dtype))  # [n, t, 512]
        gap_features = gap_features.transpose(1, 2)  # [n, 512, t]
        video_features = video_features.unsqueeze(1)  # [n, 1, 512]
        attn_score = torch.matmul(video_features, gap_features)  # [n, 1, t]
        attn_score = softmax(attn_score, dim=-1)

        gap_features = gap_features @ attn_score.transpose(1, 2)  # [n, 512, 1]
        gap_features = gap_features.transpose(1, 2)  # [n, 1, 512]
        cross_features = torch.concat([gap_features, video_features], dim=-1)  # [B, 1, 2*C]

        cross_video_features = self.cross_net(cross_features)  # [n, 1, 512]
        cross_video_features = cross_video_features / cross_video_features.norm(dim=-1, keepdim=True)
        cross_video_features = cross_video_features.squeeze(1)
        # video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        ###############################################

        ################## Text Part ##################
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        ###############################################

        output = cross_video_features @ text_features.t() / 0.1

        return  output

class ClipModelWOGAP(nn.Module):
    def __init__(self, input_text, clip_model, args):
        super().__init__()
        self.args = args
        self.input_text = input_text  # 固定的文本提示 a smiling mouse ...
        self.prompt_learner = PromptLearner(input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual
        self.temporal_net = Temporal_Transformer_Cls(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)

        # self.gap_net = GAP_Transformer(input_dim=44,
        #                                hidden_dim=256,
        #                                mlp_dim=512,
        #                                depth=args.temporal_layers,
        #                                heads=8,
        #                                dim_head=64,
        #                                t=args.t,
        #                                dropout=0., )

        # self.cross_net = nn.Sequential(nn.Linear(512 * 2, 512),
        #                                GELU(),
        #                                nn.Dropout(0.),
        #                                nn.Linear(512, 512),
        #                                nn.Dropout(0.))
        self.clip_model_ = clip_model

    def forward(self, image, gap):
        ################# Visual Part And GAP Part #################
        n, t, c, h, w = image.shape 
        image = image.contiguous().view(-1, c, h, w)  # [n*t, 3, 224, 224]
        image_features = self.image_encoder(image.type(self.dtype))  # [n*t, 512]
        # TODO: 目前是concat然后通过 MLP还原形状，尝试直接相加
        image_features = image_features.contiguous().view(n, t, -1)  # [n, t, 512]
        video_features = self.temporal_net(image_features)  # [n, 512]
        # gap_features = self.gap_net(gap.type(self.dtype))  # [n, t, 512]
        # gap_features = gap_features.transpose(1, 2)  # [n, 512, t]
        video_features = video_features.unsqueeze(1)  # [n, 1, 512]
        # attn_score = torch.matmul(video_features, gap_features)  # [n, 1, t]
        # attn_score = softmax(attn_score, dim=-1)

        # gap_features = gap_features @ attn_score.transpose(1, 2)  # [n, 512, 1]
        # gap_features = gap_features.transpose(1, 2)  # [n, 1, 512]
        # cross_features = torch.concat([gap_features, video_features], dim=-1)  # [B, 1, 2*C]

        # cross_video_features = self.cross_net(cross_features)  # [n, 1, 512]
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        video_features = video_features.squeeze(1)
        # video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        ###############################################

        ################## Text Part ##################
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        ###############################################

        output = video_features @ text_features.t() / 0.1

        return  output
class CrossAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super(CrossAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)

        self.fc = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, image, gap):
        #[512], [T, 512]
        image = image.permute(1,0,2) #[1, B, 512]
        gap = gap.permute(1,0,2) #[T, B, 512]

        attention_out, _ = self.cross_attention(image, gap, gap) #[1, B, 512]
        attention_out = attention_out.permute(1, 0, 2) #[B, 1 ,512]
        fused_out = self.norm(F.relu(self.fc(attention_out)))

        return fused_out

'''
使用CrossAttention
'''
class ClipModel2(nn.Module):
    def __init__(self, input_text, clip_model, args):
        super().__init__()
        self.args = args
        self.input_text = input_text  # 固定的文本提示 a smiling mouse ...
        self.prompt_learner = PromptLearner(input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual
        self.temporal_net = Temporal_Transformer_Cls(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)

        self.gap_net = GAP_Transformer(input_dim=49,
                                       hidden_dim=256,
                                       mlp_dim=512,
                                       depth=args.temporal_layers,
                                       heads=8,
                                       dim_head=64,
                                       t=args.t,
                                       dropout=0., )

        # self.cross_net = nn.Sequential(nn.Linear(512 * 2, 512),
        #                                GELU(),
        #                                nn.Dropout(0.),
        #                                nn.Linear(512, 512),
        #                                nn.Dropout(0.))

        self.cross_net = CrossAttention()
        self.clip_model_ = clip_model

    def forward(self, image, gap):
        ################# Visual Part And GAP Part #################
        n, t, c, h, w = image.shape 
        image = image.contiguous().view(-1, c, h, w)  # [n*t, 3, 224, 224]
        image_features = self.image_encoder(image.type(self.dtype))  # [n*t, 512]
        # TODO: 目前是concat然后通过 MLP还原形状，尝试直接相加
        image_features = image_features.contiguous().view(n, t, -1)  # [n, t, 512]
        video_features = self.temporal_net(image_features)  # [n, 512]
        gap_features = self.gap_net(gap.type(self.dtype))  # [n, t, 512]
        # gap_features = gap_features.transpose(1, 2)  # [n, 512, t]
        video_features = video_features.unsqueeze(1)  # [n, 1, 512]
        fused_output = self.cross_net(video_features,  gap_features) #[B, 1, 512]
        fused_output = fused_output.squeeze(1)
        fused_output = fused_output / fused_output.norm(dim=-1, keepdim=True)
        # attn_score = torch.matmul(video_features, gap_features)  # [n, 1, t]
        # attn_score = softmax(attn_score, dim=-1)

        # gap_features = gap_features @ attn_score.transpose(1, 2)  # [n, 512, 1]
        # gap_features = gap_features.transpose(1, 2)  # [n, 1, 512]
        # cross_features = torch.concat([gap_features, video_features], dim=-1)  # [B, 1, 2*C]

        # cross_video_features = self.cross_net(cross_features)  # [n, 1, 512]
        # cross_video_features = cross_video_features / cross_video_features.norm(dim=-1, keepdim=True)
        # cross_video_features = cross_video_features.squeeze(1)
        # video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        ###############################################

        ################## Text Part ##################
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        ###############################################

        output = fused_output @ text_features.t() / 0.1

        return  output



class ClipMode3(nn.Module):
    def __init__(self, input_text, clip_model, args):
        super().__init__()
        self.args = args
        self.input_text = input_text  # 固定的文本提示 a smiling mouse ...
        self.prompt_learner = PromptLearner(input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual
        self.temporal_net = Temporal_Transformer_Cls(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)


        #输入形状为[B, T, 49]
        
        # self.gap_net = GAP_Transformer(input_dim=49,
        #                                hidden_dim=256,
        #                                mlp_dim=512,
        #                                depth=args.temporal_layers,
        #                                heads=8,
        #                                dim_head=64,
        #                                t=args.t,
        #                                dropout=0., )

        self.gap_net = GAP_TCN(num_inputs=44, 
                               hidden_dim=256,
                               mlp_dim=512,
                               num_channels=[25, 50, 100], 
                               kernel_size=3,
                               dropout=0.1)
        
        self.cross_net = nn.Sequential(nn.Linear(512 * 2, 512),
                                       GELU(),
                                       nn.Dropout(0.),
                                       nn.Linear(512, 512),
                                       nn.Dropout(0.))
        self.clip_model_ = clip_model

    def forward(self, image, gap):
        ################# Visual Part And GAP Part #################
        n, t, c, h, w = image.shape 
        image = image.contiguous().view(-1, c, h, w)  # [n*t, 3, 224, 224]
        image_features = self.image_encoder(image.type(self.dtype))  # [n*t, 512]
        # TODO: 目前是concat然后通过 MLP还原形状，尝试直接相加
        image_features = image_features.contiguous().view(n, t, -1)  # [n, t, 512]
        video_features = self.temporal_net(image_features)  # [n, 512]
        gap_features = self.gap_net(gap.type(self.dtype))  # [n, t, 512]
        gap_features = gap_features.transpose(1, 2)  # [n, 512, t]
        video_features = video_features.unsqueeze(1)  # [n, 1, 512]
        attn_score = torch.matmul(video_features, gap_features)  # [n, 1, t]
        attn_score = softmax(attn_score, dim=-1)

        gap_features = gap_features @ attn_score.transpose(1, 2)  # [n, 512, 1]
        gap_features = gap_features.transpose(1, 2)  # [n, 1, 512]
        cross_features = torch.concat([gap_features, video_features], dim=-1)  # [B, 1, 2*C]

        cross_video_features = self.cross_net(cross_features)  # [n, 1, 512]
        cross_video_features = cross_video_features / cross_video_features.norm(dim=-1, keepdim=True)
        cross_video_features = cross_video_features.squeeze(1)
        # video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        ###############################################

        ################## Text Part ##################
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        ###############################################

        output = cross_video_features @ text_features.t() / 0.1

        return  output
    

class ClipModel4(nn.Module):
    def __init__(self, input_text, clip_model, args):
        super().__init__()
        self.args = args
        self.input_text = input_text  # 固定的文本提示 a smiling mouse ...
        self.prompt_learner = PromptLearner(input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual
        self.temporal_net = Temporal_Transformer_Cls(num_patches=16,
                                                     input_dim=512,
                                                     depth=args.temporal_layers,
                                                     heads=8,
                                                     mlp_dim=1024,
                                                     dim_head=64)

        self.gap_net = GAP_Transformer(input_dim=49,
                                       hidden_dim=256,
                                       mlp_dim=512,
                                       depth=args.temporal_layers,
                                       heads=8,
                                       dim_head=64,
                                       t=args.t,
                                       dropout=0., )

        self.cross_net = nn.Sequential(nn.Linear(512 * 2, 512),
                                       GELU(),
                                       nn.Dropout(0.),
                                       nn.Linear(512, 512),
                                       nn.Dropout(0.))
        self.clip_model_ = clip_model

    def forward(self, image, gap):
        ################# Visual Part And GAP Part #################
        n, t, c, h, w = image.shape 
        image = image.contiguous().view(-1, c, h, w)  # [n*t, 3, 224, 224]
        image_features = self.image_encoder(image.type(self.dtype))  # [n*t, 512]
        # TODO: 目前是concat然后通过 MLP还原形状，尝试直接相加
        image_features = image_features.contiguous().view(n, t, -1)  # [n, t, 512]
        video_features = self.temporal_net(image_features)  # [n, 512]
        gap_features = self.gap_net(gap.type(self.dtype))  # [n, t, 512]
        gap_features = gap_features.transpose(1, 2)  # [n, 512, t]
        video_features = video_features.unsqueeze(1)  # [n, 1, 512]
        attn_score = torch.matmul(video_features, gap_features)  # [n, 1, t]
        attn_score = softmax(attn_score, dim=-1)

        gap_features = gap_features @ attn_score.transpose(1, 2)  # [n, 512, 1]
        gap_features = gap_features.transpose(1, 2)  # [n, 1, 512]
        cross_features = torch.concat([gap_features, video_features], dim=-1)  # [B, 1, 2*C]

        cross_video_features = self.cross_net(cross_features)  # [n, 1, 512]
        cross_video_features = cross_video_features / cross_video_features.norm(dim=-1, keepdim=True)
        cross_video_features = cross_video_features.squeeze(1)
        # video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        ###############################################

        ################## Text Part ##################
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        ###############################################

        output = cross_video_features @ text_features.t() / 0.1

        return  output

from efficientnet_pytorch import EfficientNet 


class EfficentLstm(nn.Module):
    def __init__(self, model_name="efficientnet-b7", hidden_size=1024, dropout=0.1, num_classes=4):
        super().__init__()
        self.base_model = EfficientNet.from_pretrained(model_name)
        self.base_model._fc = nn.Identity()
        self.input_dim = -1 
        if model_name == "efficientnet-b7":
            self.input_dim = 2560

        self.lstm = nn.LSTM(input_size=self.input_dim, batch_first=True, hidden_size=hidden_size)

        self.drop_out = nn.Dropout(dropout)
        self.out_fc = nn.Sequential(nn.Linear(hidden_size, num_classes))
        #softmax在使用CE的时候自动使用

    def forward(self, x):
        B, T, _, H, W = x.shape

        x = x.view(B*T, -1, H, W)
        x = self.base_model(x) #[B *t, 2560]
        x = x.view(B, T, -1)
        lstm_output, (h_n, c_n)= self.lstm(x)
        x = h_n[-1] #取最后一个隐藏状态
        x = self.drop_out(x)
        x = self.out_fc(x) #[B, 4]


        return x 



#后融合
from models.Conformer import TS_Stream
class EfficientConformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conformer = TS_Stream(emb_size=40)

        self.classification_head_conformer = nn.Sequential(
            nn.Linear(520, 128),  
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)
        )

        self.efficient = EfficentLstm()
        self.conformer_weight = torch.tensor(0.5)
        self.efficient_weight = torch.tensor(0.5)

    def forward(self, images, gaps):
        B, T, C, H, W = images.shape
        _, L, M = gaps.shape

        gaps = gaps.transpose(-1, -2) #[B, M,L] [B, 3, 280]

        # images = images.view(-1, C, H, W)
        efficient_out = self.efficient(images)
       
        gaps = gaps.unsqueeze(1) #[B, 1, M, 280]
    
        gaps = self.conformer(gaps)

        gaps = gaps.view(B, -1)
        conformer_out = self.classification_head_conformer(gaps)

        return self.conformer_weight * conformer_out + self.efficient_weight * efficient_out

class EfficentBiLSTM(nn.Module):
    def __init__(self, model_name="efficientnet-b7", hidden_size=1024, dropout=0.1, num_classes=4):

        super().__init__()
        self.base_model = EfficientNet.from_pretrained(model_name)
        self.base_model._fc = nn.Identity()
        self.input_dim = 0
        if model_name == 'efficientnet-b7':
            self.input_dim = 2560

        self.lstm = nn.LSTM(input_size=self.input_dim, batch_first=True, hidden_size=hidden_size, bidirectional=True)

        self.dropout = nn.Dropout(dropout)


        self.out_fc = nn.Sequential(nn.Linear(hidden_size *2, num_classes))
    def forward(self, x):
        n, t, c, h, w = x.shape
        x = x.contiguous().view(-1, c, h, w)
        x = self.base_model(x)
        x = x.contiguous().view(n, -1, 2560)
        output, _ = self.lstm(x)
        x = torch.concat((output[:, -1, :1024], output[:, 0, 1024:]), dim=1)
        x = self.dropout(x)
        x = self.out_fc(x)
        return x

from torchvision import models
#InceptionNet
class InceptionNetV3(nn.Module):
    def __init__(self, model_name='inception_v3', hidden_size=1024, dropout=0.0, num_classes=4, seq=False):
        super().__init__()
        self.base_model = models.inception_v3(pretrained=True, aux_logis=False)
        
        #remove fc net from InceptionNet
        self.base_model.fc = nn.Identity()
        if model_name == 'inception_v3':
           self.input_dim = 2048 # inceptionnet-v3 最后一层输出 
        self.input_dim = hidden_size
        self.seq = seq
        self.dropout = nn.Dropout(dropout)

        self.out_fc = nn.Sequential(
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        n, t, c, h, w = x.shape()
        x = x.contiguous().view(-1, c, h, w)
        x = self.base_model(x)

        x = x.contiguous().view(n, t, -1)
        
        _, (h_n, c_n)= self.lstm(x)
        x = h_n[-1] #取最后一个隐藏状态
        x = self.drop_out(x)
        x = self.out_fc(x) #[B, 4]
        
        return x


        
class InceptionNetV1(nn.Module):
    def __init__(self, model_name='inception_v1', hidden_size=1024, dropout=0.0, num_classes=4, seq=False):
        super().__init__()
        #InceptionNet:googlenet
        self.base_model = models.googlenet(pretrained=True)
        
        #remove fc net from InceptionNet
        self.base_model.fc = nn.Identity()
        if model_name == 'inception_v1':
            self.input_dim = 2048 # inceptionnet-v3 最后一层输出 
        self.input_dim = hidden_size
        self.seq = seq
        self.drop_out = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=self.input_dim, batch_first=True, hidden_size=hidden_size)

        self.out_fc = nn.Sequential(
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        n, t, c, h, w = x.size()
        x = x.contiguous().view(-1, c, h, w)
        x = self.base_model(x)

        x = x.contiguous().view(n, t, -1)
        
        _, (h_n, c_n)= self.lstm(x)
        x = h_n[-1] #取最后一个隐藏状态
        x = self.drop_out(x)
        x = self.out_fc(x) #[B, 4]
        
        return x


# class EfficentBiLstm(nn.Module):

#     def forward(self, x):
#         n, t, c, h, w = x.shape
#         x = x.contiguous().view(-1, c, h, w)
#         x = self.base_model(x)  # [B, 2560]
#         x = x.contiguous().view(n, -1, 2560)
#         bilstm_output, (h_n, c_n) = self.bilstm(x)
#         # 对于双向LSTM，我们需要取两个方向的最后一个隐藏状态的平均值或拼接
#         # 这里我们取最后一个时间步的隐藏状态，并拼接两个方向的特征
#         x = torch.cat((bilstm_output[:, -1, :1024], bilstm_output[:, 0, 1024:]), dim=1)
#         x = self.drop_out(x)
#         x = self.out_fc(x)  # [B, 4]
#         return x
