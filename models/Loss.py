import torch
import torch.nn as nn
import debugpy

torch.manual_seed(seed=1)


# try:
#     debugpy.listen(('localhost', 12323))
#     print('Waiting for debugger attach')
#     debugpy.wait_for_client()
# except Exception as e:
#     print(e)


class RankLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, use_gpu=True, alpha=0.5, beta=0.5, theta=10):
        
        super(RankLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        # self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

        self.alpha = alpha
        self.theta = theta
        # self.centers_distances = nn.parameter(torch.zeros(size=(num_classes, num_classes)))

        self.register_buffer("centers", torch.randn(num_classes, feat_dim))
        self.register_buffer("centers_distances", torch.zeros(size=(num_classes - 1, num_classes)))
        if use_gpu:
            self.centers = self.centers.cuda()
            self.centers_distances = self.centers_distances.cuda()

    def forward(self, x, labels):
        # print(f'centers.shape :{self.centers_distances.shape}')
        # 计算每个样本与其类别中心的距离
        centers = self.centers.index_select(0, labels)  # [B, 128]

        dist = torch.norm(x - centers, dim=1)
        loss = torch.mean(0.5 * dist)

        
        rk_loss = self.get_distance_loss()
        self.update_centers(x, labels)
        return self.alpha * (loss + rk_loss)

    def update_centers(self, x, labels):
        unique_labels, counts = torch.unique(labels, return_counts=True)

        # 遍历每个类别，更新中心
        for i, label in enumerate(unique_labels):
            # 获取属于该类的样本
            mask = labels == label
            x_class = x[mask]  # 获取属于该类的所有样本特征
            count_class = counts[i].item()  # 属于该类的样本数量

            # 计算新的类别中心
            # 更新公式: centers_new = (centers_old * N_class + sum(x_class)) / (N_class + 1)
            new_center = (self.centers[label] * count_class + x_class.sum(dim=0)) / (count_class + 1)

            # 更新该类的中心点
            self.centers.data[label] = new_center
            self.update_centers_distances()

    def update_centers_distances(self):
        for i in range(self.num_classes - 1):
            for j in range(self.num_classes - 1):
                if (i + j) > (self.num_classes - 1):
                    continue
                else:
                    self.centers_distances[i, j] = torch.norm(self.centers[i] - self.centers[j + i])

    def get_distance_loss(self):
        # 获取 loss_rk1, loss_rk2
        loss_rk1 = 0.0
        loss_rk2 = 0.0
        for i in range(self.num_classes - 2):
            for j in range(self.num_classes - 1):
                loss_rk1 += torch.max(torch.tensor(0.0),
                                      self.theta - (self.centers_distances[i, 2] - self.centers_distances[j, 1]))

        for i in range(1):
            for j in range(self.num_classes - 1):
                loss_rk2 += torch.max(torch.tensor(0.0),
                                      2 * self.theta - (self.centers_distances[i, 3] - self.centers_distances[j, 1]))

        return loss_rk1 + loss_rk2

class TestLoss(nn.Module):
    def __init__(self):
        super(TestLoss, self).__init__()

    def forward(self, y_pred, y_true):

        loss = torch.mean(torch.abs(y_pred - y_true))
        return loss.mean()
    
class SmoothLoss(nn.Module):
    def __init__(self, num_classes=4):
        super(SmoothLoss, self).__init__()
        self.alpha = 1.1
        self.num_classes = num_classes
        self.smooth_labels = self._compute_smooth_labels()
    
    def _compute_smooth_labels(self):
        smooth_labels = []
        for target in range(self.num_classes):
            indices = torch.arange(self.num_classes)
            distances = torch.abs(indices - target).float()
            same_group = ((target < 2) & (indices < 2)) | ((target >= 2) & (indices >= 2))
            distances = torch.where(same_group, distances, distances * self.alpha)
            exp_distances = torch.exp(-distances)
            normalized_labels = exp_distances / exp_distances.sum()
            smooth_labels.append(normalized_labels)

        return torch.stack(smooth_labels)  # Shape: [num_classes, num_classes]

    def forward(self, y_pred, y_true):
        B = y_pred.size(0)

        y_true = y_true.long()
        smooth_labels = self.smooth_labels.to(y_true.device)
        smooth_labels = smooth_labels[y_true] #[B, 4]

        y_pred = torch.softmax(y_pred, dim=-1)
        loss = -torch.sum(smooth_labels * torch.log(y_pred + 1e-10), dim=1)
        
        return loss.mean()
    

        
# criterion = TestLoss().cuda()

# y_pred = torch.randn(32, 4)
# y_true = torch.randn(32, 4)

# loss = criterion(y_pred, y_true)

# print(loss)
# print(loss.item())
# num_classes = 4
# feat_dim = 128
# center_loss = RankLoss(num_classes, feat_dim)

# x = torch.randn(16, feat_dim)
# label = torch.randint(0, num_classes, (16,))

# loss = center_loss(x, label)

# print(loss)

# criterion = SmoothLoss(num_classes=4).cuda()

# y_pred = torch.randn(32, 4)#[B, 4]
# y_true = torch.randn(32,) #[B, 1]
# loss = criterion(y_pred, y_true)
# # print(loss)
# print(loss.item())