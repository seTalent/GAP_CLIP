from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch

# from models.Module.ResNet import __all__, resnet50, resnet18

#DAiSEE数据集上将每个样本分割成 5个片段，每个片段60帧


class EfficientnetLSTM(nn.Module):
    def __init__(self, model_name="efficientnet-b7", hidden_size=1024, dropout=0.1, num_classes=4):
        super().__init__()
        self.base_model = EfficientNet.from_pretrained(model_name)
        self.base_model._fc = nn.Identity()
        self.input_dim = 0
        if model_name == "efficientnet-b7":
            self.input_dim = 2560

        self.lstm = nn.LSTM(input_size=self.input_dim, batch_first=True, hidden_size=hidden_size)

        self.drop_out = nn.Dropout(dropout)
        self.out_fc = nn.Sequential(nn.Linear(hidden_size, num_classes))
        #softmax在使用CE的时候自动使用

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(-1, C, H, W)

        x = self.base_model(x) #[B*T, 2560]
        x = x.view(B, T, -1)
        lstm_output, (h_n, c_n)= self.lstm(x)
        x = h_n[-1] #取最后一个隐藏状态
        x = self.drop_out(x)
        x = self.out_fc(x) #[B, 4]


        return x 



