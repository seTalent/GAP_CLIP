import argparse
import os

import time
import shutil
from logging import debug

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.Generate_Model import ClipModel, EfficientConformer, ClipModel2, ClipMode3, ClipModelWOGAP, EfficentLstm, EfficentBiLSTM, InceptionNetV1, InceptionNetV3
from models.SwinTransformer import SwinFace
from models.RestNetTCN import ResNet50TCN
from models.Loss import SmoothLoss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
import datetime
# from dataloader.swin_dataloader import train_data_loader
from dataloader.video_dataloader import train_data_loader, test_data_loader
from sklearn.metrics import confusion_matrix
from models.clip import clip
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from models.Text import *
import random
from tqdm import tqdm



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)

parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=48)
parser.add_argument('--lr-gap-net', type=float, default=1e-5)
parser.add_argument('--lr-temporal-net', type=float, default=1e-2)
parser.add_argument('--lr-image-encoder', type=float, default=1e-5)
parser.add_argument('--lr-prompt-learner', type=float, default=1e-3)
parser.add_argument('--lr-cross-net', type=float, default=1e-3)

parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--milestones', nargs='+', type=int)

parser.add_argument('--contexts-number', type=int, default=4)
parser.add_argument('--class-token-position', type=str, default="end")
parser.add_argument('--class-specific-contexts', type=str, default='False')

parser.add_argument('--text-type', type=str)
parser.add_argument('--exper-name', type=str)
parser.add_argument('--seed', type=int)
parser.add_argument('--temporal-layers', type=int, default=1)
parser.add_argument("--pretrain", type=str, default="pretrain")
parser.add_argument("--load_and_tune_prompt_learner", type=bool, default=True)
parser.add_argument("--t", type=int, default=16)
args = parser.parse_args()

random.seed(args.seed)  
np.random.seed(args.seed) 
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

now = datetime.datetime.now()
time_str = now.strftime("%y%m%d%H%M")
time_str = time_str + args.exper_name

print('************************')
for k, v in vars(args).items():
    print(k,'=',v)
print('************************')


if args.dataset == "DAiSEE" or args.dataset == 'EmotiW':
    number_class = 4
    class_names = class_names_4
    class_names_with_context = class_names_with_context_4
    class_descriptor = class_descriptor_4
def main(set):

    data_set = set+1

    dataset = args.dataset
    print(f"*********************Exp On {dataset}*********************")
    log_txt_path = './log/' + f'{dataset}-' + time_str + '-log.txt'
    log_curve_path = './log/' + f'{dataset}-' + time_str + '-log.png'
    log_confusion_matrix_path = './log/' + f'{dataset}-' + time_str + '-set' + '-cn.png' 
    checkpoint_path = './checkpoint/' + f'{dataset}-' + time_str + '-model.pth'
    best_checkpoint_path = './checkpoint/' + f'{dataset}-' + time_str + '-model_best.pth'
    train_annotation_file_path = f"./annotation/{dataset}_Train.txt"
    test_annotation_file_path = f"./annotation/{dataset}_Test.txt"
    valid_annotation_file_path = f"./annotation/{dataset}_Validation.txt"
    
    best_acc = 0
    recorder = RecorderMeter(args.epochs)
    print('The training name: ' + time_str)

    clip_path = args.pretrain
    # create model and load pre_trained parameters
    CLIP_model, _ = clip.load(clip_path, device='cpu')

    if args.text_type=="class_names":
        input_text = class_names
    elif args.text_type=="class_names_with_context":
        input_text = class_names_with_context
    elif args.text_type=="class_descriptor":
        input_text = class_descriptor

    print("Input Text: ")
    for i in range(len(input_text)):
        print(input_text[i])

    # model = ClipModel(input_text=input_text, clip_model=CLIP_model, args=args)
    # model = ClipModelWOGAP(input_text=input_text, clip_model=CLIP_model, args=args)
    # model = ClipModel2(input_text=input_text, clip_model=CLIP_model, args=args)
    # model = ClipMode3(input_text=input_text, clip_model=CLIP_model, args=args)
    # model = EfficientConformer()
    model = EfficentLstm()
    # model = EfficentBiLSTM()
    # model = ResNet50TCN(num_classes=4)
    # model = InceptionNetV1()
    optimizer = None
    # if args.flow == True:
    #     model = SwinFace(img_size=224, patch_size=4,in_chans=3,num_classes=4, embed_dim=96, window_size=7)
    #     #SGD or Adan
    #     optimizer = torch.optim.SGD(lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)

    '''
    GAP-CLIP
    '''
    # for name, param in model.named_parameters():
    #     # if "image_encoder" in name:
    #     #     param.requires_grad = True  
    #     if "temporal_net" in name:
    #         param.requires_grad = True
    #     if "prompt_learner" in name:  
    #         param.requires_grad = True
    #     if "gap_net" in name:
    #         param.requires_grad = True
    #     if 'cross_net' in name:
    #         param.requires_grad = True
    '''
    LSTM: Inceptionnet/EfficientNet
    '''
    # for name, param in model.named_parameters():
    #     if "lstm" in name:
    #         param.requires_grad = True  
    #     if "out_fc" in name:
    #         param.requires_grad = True
        # if "conformer" in name:  
        #     param.requires_grad = True
        # if "classification_head_conformer" in name:
        #     param.requires_grad = True
    '''
    ResNet50+TCN
    '''
    for name, param in model.named_parameters():
        if "tcn" in name:
            param.requires_grad = True  
        if "output" in name:
            param.requires_grad = True
        # if "conformer" in name:  
        #     param.requires_grad = True
        # if "classification_head_conformer" in name:
        #     param.requires_grad = True
    #multi gpus
    model = torch.nn.DataParallel(model).cuda()
    
    # # print params   
    # print('************************')
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    # print('************************')
    
    with open(log_txt_path, 'a') as f:
        for k, v in vars(args).items():
            f.write(str(k) + '=' + str(v) + '\n')
    
    # define loss function (criterion)

    criterion = nn.CrossEntropyLoss().cuda()

    # criterion = SmoothLoss(num_classes=4).cuda()


    # define optimizer
    '''
    GAP_CLIP
    '''
    # optimizer = torch.optim.SGD([{"params": model.module.temporal_net.parameters(), "lr": args.lr_temporal_net},
    #                             #  {"params": model.module.image_encoder.parameters(), "lr": args.lr_image_encoder},
    #                              {"params": model.module.prompt_learner.parameters(), "lr": args.lr_prompt_learner},
    #                              {"params": model.module.gap_net.parameters(), "lr": args.lr_gap_net},
    #                              {"params": model.module.cross_net.parameters(), "lr": args.lr_cross_net}
    #                              ],
    #                              lr=args.lr_temporal_net,
    #                              momentum=args.momentum,
    #                              weight_decay=args.weight_decay)


    # optimizer = torch.optim.SGD([{"params": model.module.efficient.lstm.parameters(), "lr": args.lr_temporal_net},
    #                             {"params": model.module.efficient.out_fc.parameters(), "lr": args.lr_temporal_net},
    #                             # {"params": model.module.conformer.parameters(), "lr": args.lr_temporal_net},
    #                             # {"params": model.module.classification_head_conformer.parameters(), "lr": args.lr_temporal_net}
    #                            ],
    #                             lr=args.lr_temporal_net,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    '''
    LSTM
    '''
    optimizer = torch.optim.SGD([{"params": model.module.lstm.parameters(), "lr": args.lr_temporal_net},
                            {"params": model.module.out_fc.parameters(), "lr": args.lr_temporal_net},
                            # {"params": model.module.conformer.parameters(), "lr": args.lr_temporal_net},
                            # {"params": model.module.classification_head_conformer.parameters(), "lr": args.lr_temporal_net}
                            ],
                            lr=args.lr_temporal_net,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    '''
    ResNet+TCN
    '''
    # optimizer = torch.optim.SGD([{"params": model.module.tcn.parameters(), "lr": args.lr_temporal_net},
    #                     {"params": model.module.output.parameters(), "lr": args.lr_temporal_net},
    #                     # {"params": model.module.conformer.parameters(), "lr": args.lr_temporal_net},
    #                     # {"params": model.module.classification_head_conformer.parameters(), "lr": args.lr_temporal_net}
    #                     ],
    #                     lr=args.lr_temporal_net,
    #                     momentum=args.momentum,
    #                     weight_decay=args.weight_decay)
    # define scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.milestones,
                                                     gamma=0.1)

    cudnn.benchmark = True

    # Data loading code
    train_data = train_data_loader(list_file=train_annotation_file_path,
                                   num_segments=16,
                                   duration=1,
                                   image_size=224,
                                   args=args)
    test_data = test_data_loader(list_file=test_annotation_file_path,
                                 num_segments=16,
                                 duration=1,
                                 image_size=224)
    

    valid_data = test_data_loader(list_file=valid_annotation_file_path,
                                  num_segments=16,
                                  duration=1,
                                  image_size=224)


    #merge train and val data:
    train_data = torch.utils.data.ConcatDataset([train_data, valid_data])

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    # valid_loader = torch.utils.data.DataLoader(valid_data,
    #                                          batch_size=args.batch_size,
    #                                          shuffle=False,
    #                                          num_workers=args.workers,
    #                                          pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)
    for epoch in range(0, args.epochs):
        # break
        inf = '********************Epoch:' + str(epoch) + '********************'
        start_time = time.time()
        # current_learning_rate_0 = optimizer.state_dict()['param_groups'][0]['lr']
        # current_learning_rate_1 = optimizer.state_dict()['param_groups'][1]['lr']
        # current_learning_rate_2 = optimizer.state_dict()['param_groups'][2]['lr']
        # current_learning_rate_3 = optimizer.state_dict()['param_groups'][3]['lr']
        # current_learning_rate_4 = optimizer.state_dict()['param_groups'][4]['lr']
        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            print(inf)
            # f.write('Current learning rate: ' + str(current_learning_rate_0) + ' ' + str(current_learning_rate_1) + ' ' + '\n')
            # print('Current learning rate: ', current_learning_rate_0, current_learning_rate_1)

        # train for one epoch
        train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path)

        # evaluate on validation set(if not merge, modify the code here)
        val_acc, val_los = validate(test_loader, model, criterion, args, log_txt_path, epoch)

        scheduler.step()

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best,
                        checkpoint_path,
                        best_checkpoint_path)

        # print and save log
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder.plot_curve(log_curve_path)

        print('The best accuracy: {:.3f}'.format(best_acc.item()))
        print('An epoch time: {:.2f}s'.format(epoch_time))
        with open(log_txt_path, 'a') as f:
            f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
            f.write('An epoch time: ' + str(epoch_time) + 's' + '\n')
    
   
    # best_checkpoint_path = 'checkpoint/EmotiW-2411231001InceptionNet在EmotiW上的表现-set1-model_best.pth'

    with torch.no_grad():

        for i, (images, target, gap) in enumerate(test_loader):

      
            images = images.cuda()
            target = target.cuda()
            # gap = gap.cuda()

            # compute output
            # output = model(images, gap)
            output = model(images)
            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1, 1))
    uar, war, acc = computer_uar_war(test_loader, model, best_checkpoint_path, log_confusion_matrix_path, log_txt_path, data_set)

    return uar, war, acc
 

def train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch),
                             log_txt_path=log_txt_path)

    # switch to train mode
    model.train()

    for i, (images, target, gap) in enumerate(train_loader):

        images = images.cuda()#[B, T, 3, 224, 224]
        target = target.cuda()
        # gap = gap.cuda()
        # compute output
        
        # output = model(images, gap)        #[B, 4]
        output = model(images)        #[B, 4]
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 1))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg



def validate(val_loader, model, criterion, args, log_txt_path, epoch):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix="Epoch: [{}][Valid]".format(epoch),
                             log_txt_path=log_txt_path)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target, gap) in enumerate(val_loader):
            
            images = images.cuda() 
            target = target.cuda()
            # gap = gap.cuda()

            # compute output
            
            # output = model(images, gap)
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1, 1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)


        print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        with open(log_txt_path, 'a') as f:
            f.write('Current Accuracy: {top1.avg:.3f}'.format(top1=top1) + '\n')
    return top1.avg, losses.avg


def test(test_loader, model, args, log_txt_path):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(test_loader),
                             [losses, top1],
                             prefix='Test: ',
                             log_txt_path=log_txt_path)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target, gap) in enumerate(test_loader):

            images = images.cuda()
            target = target.cuda()
            # gap = gap.cuda()

            # compute output
            # output = model(images, gap)
            output = model(images)

            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1, 1))
            top1.update(acc1[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        with open(log_txt_path, 'a') as f:
            f.write('Test Accuracy: {top1.avg:.3f}'.format(top1=top1) + '\n')
    return top1.avg, losses.avg



def save_checkpoint(state, is_best, checkpoint_path, best_checkpoint_path):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(self.log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # print('Curve was saved')
        plt.close(fig)


def plot_confusion_matrix(cm, classes, normalize=False, title='confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=12,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

#加载预训练模型进行测试，获取结果
def computer_uar_war(val_loader, model, best_checkpoint_path, log_confusion_matrix_path, log_txt_path, data_set):
    
    pre_trained_dict = torch.load(best_checkpoint_path)['state_dict']
    model.load_state_dict(pre_trained_dict)
    
    model.eval()

    correct = 0
    with torch.no_grad():
        for i, (images,target, gap) in enumerate(tqdm(val_loader)):
            
            images = images.cuda()
            target = target.cuda()
            # gap = gap.cuda()
            # output = model(images, gap)
            output = model(images)


            predicted = output.argmax(dim=1, keepdim=True)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

            if i == 0:
                all_predicted = predicted
                all_targets = target
            else:
                all_predicted = torch.cat((all_predicted, predicted), 0)
                all_targets = torch.cat((all_targets, target), 0)

            
    war = 100. * correct / len(val_loader.dataset)
    
    # Compute confusion matrix
    _confusion_matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    np.set_printoptions(precision=4)
    normalized_cm = _confusion_matrix.astype('float') / _confusion_matrix.sum(axis=1)[:, np.newaxis]
    normalized_cm = normalized_cm * 100
    list_diag = np.diag(normalized_cm)
    uar = list_diag.mean()
        
    print("Confusion Matrix Diag:", list_diag)
    print("UAR: %0.2f" % uar)
    print("WAR: %0.2f" % war)

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    title_ = ""

    if args.dataset == "DAiSEE":
        title_ = "Confusion Matrix on DAiSEE fold "+str(data_set)
    plot_confusion_matrix(normalized_cm, classes=class_names, normalize=True, title=title_)
    plt.savefig(os.path.join(log_confusion_matrix_path))
    plt.close()

    acc = correct / len(val_loader.dataset)

    with open(log_txt_path, 'a') as f:
        f.write('************************' + '\n')
        f.write("Confusion Matrix Diag:" + '\n')
        f.write(str(list_diag.tolist()) + '\n')
        f.write('UAR: {:.2f}'.format(uar) + '\n')        
        f.write('WAR: {:.2f}'.format(war) + '\n')
        f.write('ACC: {:.2f}'.format(acc) + '\n')
        f.write('************************' + '\n')

    return uar, war, acc

if __name__ == '__main__':

    UAR = 0.0
    WAR = 0.0
    print(args.dataset)

    all_fold = 1
    for set in range(all_fold):
        uar, war ,acc = main(set)
        UAR += float(uar)
        WAR += float(war)
        print(f'Final Acc = {acc}')

    print('********* Final Results *********')   
    print("UAR: %0.2f" % (UAR/all_fold))
    print("WAR: %0.2f" % (WAR/all_fold))
    print('*********************************')







# import argparse
# import os

# import time
# import shutil
# from logging import debug

# import torch
# import torch.distributed
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim
# import torch.utils.data
# import torch.utils.data.distributed
# # from models.Generate_Model import ClipModel, EfficientConformer, ClipModel2, ClipMode3, ClipModelWOGAP, EfficentLstm, EfficentBiLSTM, InceptionNetV1, InceptionNetV3
# from models.SwinTransformer import SwinFace
# from models.RestNetTCN import ResNet50TCN
# from models.Loss import SmoothLoss
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import numpy as np
# import itertools
# import datetime
# from dataloader.swin_dataloader import train_data_loader, test_data_loader,val_data_loader
# from sklearn.metrics import confusion_matrix
# from models.clip import clip
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# from models.Text import *
# import random
# from tqdm import tqdm
# from torch.utils.data.distributed import DistributedSampler
# from torch.utils.data.sampler import WeightedRandomSampler
# from baselines.ResNetTCN.ResTCN import ResTCN
# # import debugpy
# # try:
# #     debugpy.listen(('localhost', 12323))
# #     print('Waiting for debugger attach')
# #     debugpy.wait_for_client()
# # except Exception as e:
# #     print(e)

    
# def main(args):
#     dataset = args.dataset
#     print(f"*********************Exp On {dataset}*********************")
#     print('The training name: ' + time_str)
#     log_txt_path = './log/' + f'{dataset}-' + time_str + '-log.txt'
#     log_curve_path = './log/' + f'{dataset}-' + time_str + '-log.png'
#     log_confusion_matrix_path = './log/' + f'{dataset}-' + time_str + '-set' + '-cn.png' 
#     checkpoint_path = './checkpoint/' + f'{dataset}-' + time_str + '-model.pth'
#     best_checkpoint_path = './checkpoint/' + f'{dataset}-' + time_str + '-model_best.pth'
#     train_annotation_file_path = f"./annotation/{dataset}_Train.txt"
#     test_annotation_file_path = f"./annotation/{dataset}_Test.txt"
#     valid_annotation_file_path = f"./annotation/{dataset}_Validation.txt"

#     best_acc = 0
#     recorder = RecorderMeter(args.epochs)

#     model = None
#     optimizer = None
#     #Swin-T
#     if args.model =='swin':
#         model = SwinFace(img_size=args.img_size, patch_size=4,in_chans=3,num_classes=4, embed_dim=96, window_size=7,depths=args.depths, num_heads=args.num_heads, use_flow=args.flow, fuse_type=args.fuse_type)
#     elif args.model == 'resnet':
#         model = ResTCN()     
#     #load SwinTransformer checkpoint or train from scratch
#     if args.model == 'swin':
#         if args.run_type == 'train':
#             if args.fine_tune == 'fine_tune' or args.fine_tune == 'fully_fine_tune':
#                 #模型字典
#                 model_dict = model.state_dict() #[a,b,c,d_,e]
#                 #预训练模型字典
#                 pretrain_ckpt = torch.load(args.pretrain_path)['model'] #[a,b,c,d]
#                 # if args.local_rank == 0 :
#                 print(f'Loading ckpt from {args.pretrain_path}')
#                 for key in  list(pretrain_ckpt.keys()):
#                     if key not in model_dict: #[a,b,c,d]
#                         pretrain_ckpt.pop(key)
#                 model.load_state_dict(pretrain_ckpt, strict=False)  #允许加载不完全匹配的参数 [a,b,c]
#                 if args.fine_tune == 'fine_tune': #部分微调而不是全量微调
#                     #尝试一下全量微调
#                     for name, param in model.named_parameters():
#                         if name in pretrain_ckpt:
#                             param.requires_grad = False
#                 # if args.local_rank == 0:
#                 print('Finished loading parameters from checkpoint')
#             else:
#                 # if args.local_rank == 0:
#                 print(f'train from scratch')
#                     # print(model.parameters())
    
#     # if args.local_rank == 0:
#     if args.run_type == 'train':
#         total_params = sum(p.numel() for p in model.parameters())
#         trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad==True)
#         print('************************')
#         print(f'Total Params : {total_params}')
#         print(f'Trainable Params : {trainable_params}')
#         # print(p for p in model.parameters() if p.requires_grad == True)

#         for name, param in model.named_parameters():
#             print(name, param.requires_grad)
#         print('************************')

#         with open(log_txt_path, 'a') as f:
#             f.write('************************\n')
#             f.write('Total Params : ' + f"{total_params}\n")
#             f.write('Trainable Params : ' + f"{trainable_params}\n")
#             # for name, param in model.named_parameters():
#             #     f.write(name + '' + f'{param.requires_grad}' +  '\n')
#             # f.write('************************\n')



#     model = torch.nn.DataParallel(model).cuda()

#     # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
#     #SGD or Adan
#     # optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
#     if args.model == 'swin':
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#     if args.model == 'resnet':
#         optimizer = torch.optim.SGD([{"params": model.module.tcn.parameters(), "lr": args.lr},
#                 {"params": model.module.linear.parameters(), "lr": args.lr},
#                 ],
#                 lr=args.lr,
#                 momentum=args.momentum)
#     # optimizer=torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay, amsgrad=False)

#     # model = torch.nn.DataParallel(model).cuda()
#     # if args.local_rank == 0:
#     if args.run_type == 'train':
#         with open(log_txt_path, 'a') as f:
#             for k, v in vars(args).items():
#                 f.write(str(k) + '=' + str(v) + '\n')
        
#     # define loss function (criterion)
     
#     criterion = nn.CrossEntropyLoss().cuda() 
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                      milestones=args.milestones,
#                                                      gamma=0.1)
    
#     cudnn.benchmark = True

#     # Data loading code

#     train_data = train_data_loader(list_file=train_annotation_file_path,
#                                    num_segments=16,
#                                    duration=1,
#                                    image_size=args.img_size,
#                                    args=args)
#     # train_sampler = DistributedSampler(train_data)
#     test_data = test_data_loader(list_file=test_annotation_file_path,
#                                  num_segments=16,
#                                  duration=1,
#                                  image_size=args.img_size,
#                                  args=args)
#     # test_sampler = DistributedSampler(test_data)
#     valid_data = test_data_loader(list_file=valid_annotation_file_path,
#                                   num_segments=16,
#                                   duration=1,
#                                   image_size=args.img_size,
#                                 args=args)

#     train_data = torch.utils.data.ConcatDataset([train_data, valid_data])
#     if args.weighted_sampler:
#         print('Use  Weighted Sampler')
#         labels = [sample[1] for sample in train_data]
#         class_counts = torch.bincount(torch.tensor(labels), minlength=4)
#         print(f'Class counts: {class_counts}')

#         weights = 1.0 / class_counts.float()
#         # weights = weights / weights.sum()
#         weights = torch.tensor([weights[label] for label in labels])
#         print(f'weights: {weights}')

#         weighted_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
#         train_sampler = DistributedSampler(train_data, shuffle=True)
        
#         train_loader = torch.utils.data.DataLoader(train_data,
#                                                 sampler=weighted_sampler,
#                                             batch_size=args.batch_size,
#                                             num_workers=args.workers,
#                                             pin_memory=True,
#                                             drop_last=True)
#     else:
#         # train_sampler = DistributedSampler(train_data, shuffle=True)
#     # valid_sampler = DistributedSampler(valid_data)
#         # train_sampler = 
#         train_loader = torch.utils.data.DataLoader(train_data,
#                                             #    sampler=train_sampler,
#                                             batch_size=args.batch_size,
#                                             shuffle=True,
#                                             num_workers=args.workers,
#                                             pin_memory=True,
#                                             drop_last=True)
#     # valid_loader = torch.utils.data.DataLoader(valid_data,
#     #                                     batch_size=args.batch_size,
#     #                                     shuffle=False,
#     #                                     num_workers=args.workers,
#     #                                     pin_memory=True,
#     #                                     drop_last=False)
#     test_loader = torch.utils.data.DataLoader(test_data,
#                                         # shuffle=True,
#                                         batch_size=args.batch_size,
#                                         shuffle=False,
#                                         num_workers=args.workers,
#                                         pin_memory=True)


#     if args.run_type == 'train':
#         for epoch in range(0, args.epochs):
#             # train_sampler.set_epoch(epoch)

#             inf = '********************Epoch:' + str(epoch) + '********************'
#             start_time = time.time()
#             # current_learning_rate_0 = optimizer.state_dict()['param_groups'][0]['lr']
#             # if args.local_rank == 0 :
#             with open(log_txt_path, 'a') as f:
#                 f.write(inf + '\n')
#                 print(inf)
#             # train for one epoch
#             train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path)
#             # evaluate on validation set
#             val_acc, val_los = validate(test_loader, model, criterion, args, log_txt_path, epoch)
#             scheduler.step()
#             # remember best acc and save checkpoint
#             is_best = val_acc > best_acc
            
#             best_acc = max(val_acc, best_acc)
#             # if args.local_rank == 0:
#             save_checkpoint({'epoch': epoch + 1,
#                             'state_dict': model.state_dict(),
#                             'best_acc': best_acc,
#                             'optimizer': optimizer.state_dict(),
#                             'recorder': recorder}, is_best,
#                             checkpoint_path,
#                             best_checkpoint_path)

#             # print and save log
#             epoch_time = time.time() - start_time
#             recorder.update(epoch, train_los, train_acc, val_los, val_acc)
#             recorder.plot_curve(log_curve_path)
        
#             print('The best accuracy: {:.3f}'.format(best_acc.item()))
#             print('An epoch time: {:.2f}s'.format(epoch_time))
#             with open(log_txt_path, 'a') as f:
#                 f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
#                 f.write('An epoch time: ' + str(epoch_time) + 's' + '\n')
    
#     elif args.run_type=='test':
#         best_checkpoint_path = args.best_checkpoint_path
#     else:
#         raise ValueError("No such run type")
#     # best_checkpoint_path = 'checkpoint/EmotiW-2411231001InceptionNet在EmotiW上的表现-set1-model_best.pth'
#     # if args.local_rank == 0:
#     uar, war, acc = computer_uar_war(test_loader, model, best_checkpoint_path, log_confusion_matrix_path, log_txt_path)

#     return uar, war, acc
 

# def train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path):
#     losses = AverageMeter('Loss', ':.4f')
#     top1 = AverageMeter('Accuracy', ':6.3f')
#     progress = ProgressMeter(len(train_loader),
#                              [losses, top1],
#                              prefix="Epoch: [{}]".format(epoch),
#                              log_txt_path=log_txt_path)

#     # switch to train mode
#     model.train()

#     for i, (images, target, flow) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train:epoch {epoch}'):

#         images = images.cuda()#[B, T, 3, 224, 224]
#         target = target.cuda()
#         if args.flow:
#             flow = flow.cuda()
#         # compute output
#         if args.model == 'swin':
#             output = model(images, flow)        #[B, 4]
#         else:
#             output = model(images)
#         loss = criterion(output, target)

#         # measure accuracy and record loss
#         acc1, _ = accuracy(output, target, topk=(1, 1))
#         losses.update(loss.item(), images.size(0))
#         top1.update(acc1[0], images.size(0))

#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # print loss and accuracy
#         if i % args.print_freq == 0:
#             progress.display(i)

#     return top1.avg, losses.avg



# def validate(val_loader, model, criterion, args, log_txt_path, epoch):
#     losses = AverageMeter('Loss', ':.4f')
#     top1 = AverageMeter('Accuracy', ':6.3f')
#     progress = ProgressMeter(len(val_loader),
#                              [losses, top1],
#                              prefix="Epoch: [{}][Valid]".format(epoch),
#                              log_txt_path=log_txt_path)

#     # switch to evaluate mode
#     model.eval()

#     with torch.no_grad():
#         for i, (images, target, flow) in enumerate(val_loader):
            
#             images = images.cuda() 
#             target = target.cuda()
#             if args.flow:
#                 flow = flow.cuda()
#             # flow = flow.cuda()
#             if args.model == 'swin':
#                 output = model(images, flow)        #[B, 4]
#             else:
#                 output = model(images)
#             # compute output
            
#             loss = criterion(output, target)

#             # measure accuracy and record loss
#             acc1, _ = accuracy(output, target, topk=(1, 1))
#             losses.update(loss.item(), images.size(0))
#             top1.update(acc1[0], images.size(0))

#             if i % args.print_freq == 0:
#                 progress.display(i)
#         # if args.local_rank == 0:
#         print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
#         with open(log_txt_path, 'a') as f:
#             f.write('Current Accuracy: {top1.avg:.3f}'.format(top1=top1) + '\n')
#     return top1.avg, losses.avg


# def test(test_loader, model, args, log_txt_path):
#     losses = AverageMeter('Loss', ':.4f')
#     top1 = AverageMeter('Accuracy', ':6.3f')
#     progress = ProgressMeter(len(test_loader),
#                              [losses, top1],
#                              prefix='Test: ',
#                              log_txt_path=log_txt_path)

#     # switch to evaluate mode
#     model.eval()

#     with torch.no_grad():
#         for i, (images, target, flow) in enumerate(test_loader):

#             images = images.cuda()
#             target = target.cuda()
#             if args.flow:
#                 flow = flow.cuda()
#             # flow = flow.cuda()
#             if args.model == 'swin':
#                 output = model(images, flow)        #[B, 4]
#             else:
#                 output = model(images)


#             # measure accuracy and record loss
#             acc1, _ = accuracy(output, target, topk=(1, 1))
#             top1.update(acc1[0], images.size(0))

#             if i % args.print_freq == 0 and args.local_rank==0:
#                 progress.display(i)

#         # TODO: this should also be done with the ProgressMeter
#         # if args.local_rank == 0:
#         print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
#         with open(log_txt_path, 'a') as f:
#             f.write('Test Accuracy: {top1.avg:.3f}'.format(top1=top1) + '\n')
#     return top1.avg, losses.avg



# def save_checkpoint(state, is_best, checkpoint_path, best_checkpoint_path):
#     torch.save(state, checkpoint_path)
#     if is_best:
#         print(f'best changed, model have save in :{best_checkpoint_path}')
#         shutil.copyfile(checkpoint_path, best_checkpoint_path)

# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self, name, fmt=':f'):
#         self.name = name
#         self.fmt = fmt
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count

#     def __str__(self):
#         fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
#         return fmtstr.format(**self.__dict__)


# class ProgressMeter(object):
#     def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
#         self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
#         self.meters = meters
#         self.prefix = prefix
#         self.log_txt_path = log_txt_path

#     def display(self, batch):
#         entries = [self.prefix + self.batch_fmtstr.format(batch)]
#         entries += [str(meter) for meter in self.meters]
#         print_txt = '\t'.join(entries)
#         print(print_txt)
#         with open(self.log_txt_path, 'a') as f:
#             f.write(print_txt + '\n')

#     def _get_batch_fmtstr(self, num_batches):
#         num_digits = len(str(num_batches // 1))
#         fmt = '{:' + str(num_digits) + 'd}'
#         return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#         res = []
#         for k in topk:
#             correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res


# class RecorderMeter(object):
#     """Computes and stores the minimum loss value and its epoch index"""
#     def __init__(self, total_epoch):
#         self.reset(total_epoch)

#     def reset(self, total_epoch):
#         self.total_epoch = total_epoch
#         self.current_epoch = 0
#         self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
#         self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

#     def update(self, idx, train_loss, train_acc, val_loss, val_acc):
#         self.epoch_losses[idx, 0] = train_loss * 50
#         self.epoch_losses[idx, 1] = val_loss * 50
#         self.epoch_accuracy[idx, 0] = train_acc
#         self.epoch_accuracy[idx, 1] = val_acc
#         self.current_epoch = idx + 1

#     def plot_curve(self, save_path):

#         title = 'the accuracy/loss curve of train/val'
#         dpi = 80
#         width, height = 1600, 800
#         legend_fontsize = 10
#         figsize = width / float(dpi), height / float(dpi)

#         fig = plt.figure(figsize=figsize)
#         x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
#         y_axis = np.zeros(self.total_epoch)

#         plt.xlim(0, self.total_epoch)
#         plt.ylim(0, 100)
#         interval_y = 5
#         interval_x = 1
#         plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
#         plt.yticks(np.arange(0, 100 + interval_y, interval_y))
#         plt.grid()
#         plt.title(title, fontsize=20)
#         plt.xlabel('the training epoch', fontsize=16)
#         plt.ylabel('accuracy', fontsize=16)

#         y_axis[:] = self.epoch_accuracy[:, 0]
#         plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
#         plt.legend(loc=4, fontsize=legend_fontsize)

#         y_axis[:] = self.epoch_accuracy[:, 1]
#         plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
#         plt.legend(loc=4, fontsize=legend_fontsize)

#         y_axis[:] = self.epoch_losses[:, 0]
#         plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
#         plt.legend(loc=4, fontsize=legend_fontsize)

#         y_axis[:] = self.epoch_losses[:, 1]
#         plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
#         plt.legend(loc=4, fontsize=legend_fontsize)

#         if save_path is not None:
#             fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
#             # print('Curve was saved')
#         plt.close(fig)


# def plot_confusion_matrix(cm, classes, normalize=False, title='confusion matrix', cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title, fontsize=16)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt), fontsize=12,
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.ylabel('True label', fontsize=18)
#     plt.xlabel('Predicted label', fontsize=18)
#     plt.tight_layout()

# #加载预训练模型进行测试，获取结果
# def computer_uar_war(val_loader, model, best_checkpoint_path, log_confusion_matrix_path, log_txt_path):
#     pre_trained_dict = torch.load(best_checkpoint_path)['state_dict']
#     model.load_state_dict(pre_trained_dict)
    
#     model.eval()

#     correct = 0
#     with torch.no_grad():
#         for i, (images,target, flow) in enumerate(tqdm(val_loader)):
            
#             images = images.cuda()
#             target = target.cuda()
#             if args.flow:
#                 flow = flow.cuda()
#             if args.model == 'swin':
#                 output = model(images, flow)        #[B, 4]
#             else:
#                 output = model(images)


#             predicted = output.argmax(dim=1, keepdim=True)
#             correct += predicted.eq(target.view_as(predicted)).sum().item()

#             if i == 0:
#                 all_predicted = predicted
#                 all_targets = target
#             else:
#                 all_predicted = torch.cat((all_predicted, predicted), 0)
#                 all_targets = torch.cat((all_targets, target), 0)

            
#     war = 100. * correct / len(val_loader.dataset)
    
#     # Compute confusion matrix
#     _confusion_matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
#     np.set_printoptions(precision=4)
#     normalized_cm = _confusion_matrix.astype('float') / _confusion_matrix.sum(axis=1)[:, np.newaxis]
#     normalized_cm = normalized_cm * 100
#     list_diag = np.diag(normalized_cm)
#     uar = list_diag.mean()
        
#     print("Confusion Matrix Diag:", list_diag)
#     print("UAR: %0.2f" % uar)
#     print("WAR: %0.2f" % war)

#     # Plot normalized confusion matrix
#     plt.figure(figsize=(10, 8))

#     title_ = f"Confusion Matrix on {args.dataset} fold "
#     class_names=['C1','C2','C3','C4']
#     plot_confusion_matrix(normalized_cm, classes=class_names, normalize=True, title=title_)
#     plt.savefig(os.path.join(log_confusion_matrix_path))
#     plt.close()

#     acc = correct / len(val_loader.dataset)

#     with open(log_txt_path, 'a') as f:
#         f.write('************************' + '\n')
#         f.write("Confusion Matrix Diag:" + '\n')
#         f.write(str(list_diag.tolist()) + '\n')
#         f.write('UAR: {:.2f}'.format(uar) + '\n')        
#         f.write('WAR: {:.2f}'.format(war) + '\n')
#         f.write('ACC: {:.2f}'.format(acc) + '\n')
#         f.write('************************' + '\n')
#     return uar, war, acc

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str)

#     parser.add_argument('--workers', type=int, default=8)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--batch-size', type=int, default=48)
#     parser.add_argument('--weight-decay', type=float, default=1e-4)
#     parser.add_argument('--momentum', type=float, default=0.9)
#     parser.add_argument('--print-freq', type=int, default=10)
#     parser.add_argument('--milestones', nargs='+', type=int)
#     parser.add_argument('--exper-name', type=str)
#     parser.add_argument('--seed', type=int)
#     parser.add_argument("--t", type=int, default=16)
#     parser.add_argument('--lr', type=float, default=-1)
#     parser.add_argument('--flow', action='store_true', default=False)
#     parser.add_argument('--weighted_sampler', action='store_true', default=False)
#     parser.add_argument('--fuse_type', type=str, default='tcn')
#     parser.add_argument('--model', type=str, default='swin')
#     # parser.add_argument('--local-rank', dest='local_rank', type=int, help='node rank for distributed training')
#     parser.add_argument('--run_type', type=str, default='train')
#     parser.add_argument('--best_checkpoint_path', type=str, help='the best model checkpoint')
#     parser.add_argument('--depths', type=str,default=[2,2,6,2], help='the swin architecture-layers')
#     parser.add_argument('--num_heads', type=str, default=[3,6,12,24], help='the swin architecture-attn_heads')
#     parser.add_argument('--fine_tune', type=str, default='fine_tune', help='Train a model from scratch | (partital)fine-tune | fully-tune')
#     parser.add_argument('--pretrain_path', type=str, help='the ckpt path of backbone')
#     parser.add_argument('--img_size', type=int, default=224)

#     args = parser.parse_args()
#     args.depths = [int(d) for d in args.depths.split(',')]
#     args.num_heads = [int(d) for d in args.num_heads.split(',')]
#     print(args)
#     # torch.cuda.set_device(args.local_rank)
#     # torch.distributed.init_process_group(backend='nccl')
#     # device = torch.device('cuda', args.local_rank)
#     random.seed(args.seed)  
#     np.random.seed(args.seed) 
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)


#     now = datetime.datetime.now()
#     time_str = now.strftime("%y%m%d%H%M")
#     time_str = time_str + args.exper_name

#     print('************************')
#     for k, v in vars(args).items():
#         print(k,'=',v)
#     print('************************')

#     UAR = 0.0
#     WAR = 0.0
#     print(args.dataset)
#     uar, war, acc = main(args)

#     # if args.local_rank == 0:
#     print(f'acc={acc}')


