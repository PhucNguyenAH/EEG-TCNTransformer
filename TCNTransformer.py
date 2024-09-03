"""
Authors: Anh Hoang Phuc Nguyen

EEG-TCNTransformer code
"""
# remember to change paths

import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import random
import datetime
import scipy.io
import scipy.io as sio
import scipy.signal as signal
import pandas as pd

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.autograd import Variable
from torchsummary import summary
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
from torch.nn.utils import weight_norm

from base import EEGModuleMixin, deprecated_args

from PIL import Image
import torchvision.transforms as transforms

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from PIL import Image
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# writer = SummaryWriter('./TensorBoardX/')
now = datetime.datetime.now()

# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 22, (1, 25), (1, 1)),
            nn.Conv2d(22, 22, (22, 1), (1, 1)),
            nn.BatchNorm2d(22),
            nn.ELU(),
            # nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(22, 22, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            # Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class TCN(EEGModuleMixin, nn.Module):
    """Temporal Convolutional Network (TCN) from Bai et al 2018.

    See [Bai2018]_ for details.

    Code adapted from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py

    Parameters
    ----------
    n_filters: int
        number of output filters of each convolution
    n_blocks: int
        number of temporal blocks in the network
    kernel_size: int
        kernel size of the convolutions
    drop_prob: float
        dropout probability
    n_in_chans: int
        Alias for `n_chans`.

    References
    ----------
    .. [Bai2018] Bai, S., Kolter, J. Z., & Koltun, V. (2018).
       An empirical evaluation of generic convolutional and recurrent networks
       for sequence modeling.
       arXiv preprint arXiv:1803.01271.
    """

    def __init__(
        self,
        n_chans=None,
        n_outputs=None,
        n_blocks=3,
        n_filters=30,
        kernel_size=22,
        drop_prob=0.5,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        sfreq=None,
        n_in_chans=None,
        add_log_softmax=False,
    ):
        (n_chans,) = deprecated_args(
            self,
            ("n_in_chans", "n_chans", n_in_chans, n_chans),
        )
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            add_log_softmax=add_log_softmax,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del n_in_chans

        self.mapping = {
            "fc.weight": "final_layer.fc.weight",
            "fc.bias": "final_layer.fc.bias",
        }
        self.ensuredims = Ensure4d()
        t_blocks = nn.Sequential()
        for i in range(n_blocks):
            n_inputs = self.n_chans if i == 0 else n_filters
            dilation_size = 2**i
            t_blocks.add_module(
                "temporal_block_{:d}".format(i),
                TemporalBlock(
                    n_inputs=n_inputs,
                    n_outputs=n_filters,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size -1) * dilation_size,
                    drop_prob=drop_prob,
                ),
            )
        self.temporal_blocks = t_blocks
        self.avgPooling = nn.AvgPool2d((1, 75), (1, 15))


    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        x = self.ensuredims(x)
        # x is in format: B x C x T x 1
        (batch_size, _, time_size, _) = x.size()
        # assert time_size >= self.min_len
        # remove empty trailing dimension
        x = x.squeeze(3)
        x = self.temporal_blocks(x)
        x = self.avgPooling(x)
        x = x.permute(0,2,1)

        return x

class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, drop_prob
    ):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout1d(drop_prob)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout1d(drop_prob)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.elu = nn.ELU()

        init.normal_(self.conv1.weight, 0, 0.01)
        init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            init.normal_(self.downsample.weight, 0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.elu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.elu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.elu(out + res)

class Ensure4d(nn.Module):
    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x
    
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def extra_repr(self):
        return "chomp_size={}".format(self.chomp_size)

    def forward(self, x):
        return x[:,:,: -self.chomp_size].contiguous()


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class EEGNet(nn.Module): 
    def __init__(self,F1=8,kernel_size_1=32, D=2):
        super().__init__()
        # layer 1
        self.conv2d = nn.Conv2d(1, F1, (1,kernel_size_1), padding=(0,int(round((kernel_size_1-1)/2))))
        self.Batch_normalization_1 = nn.BatchNorm2d(F1)
        # layer 2
        self.Depthwise_conv2D = nn.Conv2d(F1, D*F1, (22,1), groups= F1)
        self.Batch_normalization_2 = nn.BatchNorm2d(D*F1)
        self.Elu = nn.ELU()
        self.Average_pooling2D_1 = nn.AvgPool2d((1,2))
        self.Dropout = nn.Dropout2d(0.2)
        # layer 3
        self.Separable_conv2D_depth = nn.Conv2d(D*F1, D*F1, (1,16),
                                                padding=(0,int(round((kernel_size_1-1)/4))), groups= D*F1)
        self.Separable_conv2D_point = nn.Conv2d(D*F1, D*F1, (1,1))
        self.Batch_normalization_3 = nn.BatchNorm2d(D*F1)
        self.Average_pooling2D_2 = nn.AvgPool2d((1,2))
        
        
    def forward(self, x):
        # layer 1
        y = self.Batch_normalization_1(self.conv2d(x)) #.relu()
        # layer 2
        y = self.Batch_normalization_2(self.Depthwise_conv2D(y))
        y = self.Elu(y)
        y = self.Dropout(self.Average_pooling2D_1(y))
        # layer 3
        y = self.Separable_conv2D_depth(y)
        y = self.Batch_normalization_3(self.Separable_conv2D_point(y))
        y = self.Elu(y)
        y = self.Dropout(self.Average_pooling2D_2(y))
        
        return y

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            # nn.Linear(2440, 256),
            nn.LazyLinear(256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class TCN_Transformer(nn.Sequential):
    def __init__(self, tcn_block=3,emb_size=70, depth=6, n_classes=4, **kwargs):
        super().__init__(
            # PatchEmbedding(emb_size), 
            EEGNet(),
            nn.Sequential(
                Rearrange('b c e t -> b c t e'),
            ),
            TCN(n_chans=16,n_blocks=tcn_block,n_filters=emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


class ExP():
    def __init__(self, nsub,result_path,tcn_block=3, emb_size=70):
        super(ExP, self).__init__()
        self.batch_size = 64
        self.n_epochs = 5000
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub
        self.result_path=result_path

        self.start_epoch = 0
        self.root = 'BCIIV2a/'
        
        self.log_write = open(os.path.join(self.result_path,"log_subject%d.txt" % self.nSub), "w")


        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = TCN_Transformer(tcn_block=tcn_block,emb_size=emb_size).cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()


    # Segmentation and Reconstruction (S&R) data augmentation
    def interaug(self, timg, label):  
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 22, 1000))
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label-1).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label
    
    def get_data(self,path, highpass = False):
        '''	Loads the dataset 2a of the BCI Competition IV
        available on http://bnci-horizon-2020.eu/database/data-sets

        Keyword arguments:
        subject -- number of subject in [1, .. ,9]
        training -- if True, load training data
                    if False, load testing data
        
        Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1750
                class_return 	numpy matrix 	size = NO_valid_trial
        '''
        NO_channels = 22
        NO_tests = 6*48 	
        Window_Length = 4*250 

        class_return = np.zeros(NO_tests)
        data_return = np.zeros((NO_tests,NO_channels,Window_Length))

        NO_valid_trial = 0
        a = scipy.io.loadmat(path)
        
        a_data = a['data']
        for ii in range(0,a_data.size):
            a_data1 = a_data[0,ii]
            a_data2= [a_data1[0,0]]
            a_data3= a_data2[0]
            a_X 		= a_data3[0]
            a_trial 	= a_data3[1]
            a_y 		= a_data3[2]
            a_fs 		= a_data3[3]
            a_classes 	= a_data3[4]
            a_artifacts = a_data3[5]
            a_gender 	= a_data3[6]
            a_age 		= a_data3[7]

            for trial in range(0,a_trial.size):
                if(a_artifacts[trial]==0):
                    data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]+500):(int(a_trial[trial]+1500)),:22])
                    class_return[NO_valid_trial] = int(a_y[trial])
                    NO_valid_trial +=1


        return data_return[0:NO_valid_trial,:,:], class_return[0:NO_valid_trial]
    
    def get_data_filter(self,path, N, aStop):
        '''	Loads the dataset 2a of the BCI Competition IV
        available on http://bnci-horizon-2020.eu/database/data-sets

        Keyword arguments:
        subject -- number of subject in [1, .. ,9]
        training -- if True, load training data
                    if False, load testing data
        
        Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1750
                class_return 	numpy matrix 	size = NO_valid_trial
        '''
        NO_channels = 22
        NO_tests = 6*48 	
        Window_Length = 4*250 

        class_return = np.zeros(NO_tests)
        data_return = np.zeros((NO_tests,NO_channels,Window_Length))

        NO_valid_trial = 0
        a = scipy.io.loadmat(path)
        a_data = a['data']
        for ii in range(0,a_data.size):
            a_data1 = a_data[0,ii]
            a_data2= [a_data1[0,0]]
            a_data3= a_data2[0]
            a_X 		= a_data3[0]
            a_trial 	= a_data3[1]
            a_y 		= a_data3[2]
            a_fs 		= a_data3[3]
            a_classes 	= a_data3[4]
            a_artifacts = a_data3[5]
            a_gender 	= a_data3[6]
            a_age 		= a_data3[7]

            for trial in range(0,a_trial.size):
                if(a_artifacts[trial]==0):
                    data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial])+500:(int(a_trial[trial])+1500),:22])
                    class_return[NO_valid_trial] = int(a_y[trial])
                    NO_valid_trial +=1

        data_return = data_return[0:NO_valid_trial,:,:]
        ws = np.array([4*2/250, 40*2/250])
        sos = signal.cheby2(N, aStop, ws, 'bandpass', output='sos')
        data_return = signal.sosfilt(sos, data_return, axis=2)
        return data_return, class_return[0:NO_valid_trial]

    def get_source_data(self,order=2,rs=10): 

        # train data
        path = self.root + 'A0%dT.mat' % self.nSub
        self.train_data, self.train_label = self.get_data(path)
        # self.train_data, self.train_label = self.get_data_filter(path, order, rs)

        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)
        self.allData = self.train_data
        self.allLabel = self.train_label

        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        # test data
        path = self.root + 'A0%dE.mat' % self.nSub
        self.test_data, self.test_label = self.get_data(path)
        # self.test_data, self.test_label = self.get_data_filter(path, order, rs)

        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label


        # standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        return self.allData, self.allLabel, self.testData, self.testLabel


    def train(self):

        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, div_factor=10.0, pct_start=0.04,final_div_factor=100.0, steps_per_epoch=len(self.dataloader),epochs=self.n_epochs)
        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Train the cnn model
        total_step = len(self.dataloader)
        curr_lr = self.lr

        for e in range(self.n_epochs):
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):

                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                # data augmentation
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))
                # img=img.permute(0,1,3,2)
                tok, outputs = self.model(img)
                loss = self.criterion_cls(outputs, label) 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()

            # test process
            if (e + 1) % 1 == 0:
                self.model.eval()
                Tok, Cls = self.model(test_data)

                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

                print('Epoch:', e,
                      'lr: %.6f' % self.optimizer.param_groups[0]["lr"],
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test accuracy is %.6f' % acc)

                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred
                    torch.save(self.model.module.state_dict(), os.path.join(self.result_path, f'model{self.nSub}.pth'))
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred
        # writer.close()


def main():
    seed_n_lst = [481,343,222,1215,1817,1278,940,1067,806]
    result_path = f'result'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    best = 0
    aver = 0
    result_write = open(os.path.join(result_path,"sub_result.txt"), "w")
    for i in range(9):
        starttime = datetime.datetime.now()
        seed_n = seed_n_lst[i]
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)


        print('Subject %d' % (i+1))
        exp = ExP(i + 1,result_path)
        bestAcc, averAcc, Y_true, Y_pred = exp.train()
        print('THE BEST ACCURACY IS ' + str(bestAcc))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")

        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))
        best = best + bestAcc
        aver = aver + averAcc

    best = best / 9
    aver = aver / 9
    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":
    main()
