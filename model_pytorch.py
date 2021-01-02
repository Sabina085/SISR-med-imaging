import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_


# source of the SRCNN code: https://github.com/yjn870/SRCNN-pytorch/blob/master/models.py
class SRCNN(nn.Module):
    def __init__(self, FLAGS, num_channels=1):
        super(SRCNN, self).__init__()
        if not FLAGS.original_SRCNN:
            self.conv1 = nn.Conv2d(num_channels, FLAGS.inner_channel, kernel_size=9, padding=9 // 2)
            self.conv2 = nn.Conv2d(FLAGS.inner_channel, FLAGS.inner_channel, kernel_size=5, padding=5 // 2)
            self.conv3 = nn.Conv2d(FLAGS.inner_channel, num_channels, kernel_size=5, padding=5 // 2)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
            self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
            self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
            self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class Regression_layer(nn.Module):
    def __init__(self, num_channels, k, dim, num, flag_shortcut):
        super(Regression_layer, self).__init__()
        self.num_channels = num_channels
        self.k = k
        self.dim = dim
        self.num = num
        self.flag_shortcut = flag_shortcut
        self.conv1 = nn.Conv2d(num_channels, num*dim, kernel_size = k, padding = k // 2)
        self.conv2 = nn.Conv2d(num_channels, num, [k, k], padding = k // 2)


    def forward(self, input_feature):
        result = self.conv1(input_feature)
        result = result.permute(0, 2, 3, 1).contiguous()
        result = result.view(result.shape[0], result.shape[1], result.shape[2], self.num, self.dim) #result.view(r[0], r[1], r[2], num, dim)

        alpha = self.conv2(input_feature)
        alpha  = alpha.permute(0, 2, 3, 1).contiguous()
        alpha = F.softmax(alpha, dim = -1)
        alpha = torch.unsqueeze(alpha, 4)

        output_feature = torch.sum(result * alpha, axis=3)
        output_feature = output_feature.permute(0, 3, 1, 2).contiguous() 

        if self.flag_shortcut:
            return output_feature + input_feature
        else:
            return output_feature


# code inspired from the tensorflow implementation: https://github.com/ofsoundof/CARN/blob/master/model.py
class CARN(nn.Module):
    def __init__(self, FLAGS, num_channels):
        super(CARN, self).__init__()
        self.num_anchor = FLAGS.num_anchor
        self.inner_channel = FLAGS.inner_channel
        self.deep_kernel = FLAGS.deep_kernel
        self.deep_layer = FLAGS.deep_layer
        self.upscale = FLAGS.upscale

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size = 3, padding = 3 // 2)
        self.prelu1 = torch.nn.PReLU(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size = 3, padding = 3 // 2)
        self.prelu2 = torch.nn.PReLU(64)

        self.conv3 =  nn.Conv2d(64, self.inner_channel, kernel_size = 3, padding = 3 // 2)
        self.prelu3= torch.nn.PReLU(self.inner_channel)

        self.reg1 = Regression_layer(self.inner_channel, self.deep_kernel,  self.upscale * self.upscale, self.num_anchor, self.inner_channel == (self.upscale * self.upscale))
        self.reg2 = Regression_layer(self.inner_channel, self.deep_kernel,  self.inner_channel, self.num_anchor, True)
        
        self.reg3_list = []
        for i in range(self.deep_layer-2):
            self.reg3_list.append(Regression_layer(self.inner_channel,  self.deep_kernel,  self.inner_channel, self.num_anchor, True))
        self.reg3 = nn.Sequential(*self.reg3_list)
        
        self.reg4 = Regression_layer(self.inner_channel, self.deep_kernel, self.upscale * self.upscale, self.num_anchor, self.inner_channel == self.upscale * self.upscale)

        self.pixel_shuffle = nn.PixelShuffle(self.upscale)


    def forward(self, lr, bic):
        feature = self.prelu1(self.conv1(lr)) 
        feature = self.prelu2(self.conv2(feature)) 
        feature = self.prelu3(self.conv3(feature))

        reshape_size = feature.shape
        kernel_size = self.deep_kernel

        if self.deep_layer == 1:
            regression = self.reg1(feature)
            
        else:
            regression = self.reg2(feature)
            regression = self.reg3(regression)
            regression = self.reg4(regression)
           
        sr_space = self.pixel_shuffle(regression)

        sr_out = torch.add(sr_space, bic) 

        return sr_out

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class newRegression_layer(nn.Module):
    def __init__(self, num_channels, k, dim, num, flag_shortcut):
        super(newRegression_layer, self).__init__()
        self.num_channels = num_channels
        self.k = k
        self.dim = dim
        self.num = num
        self.flag_shortcut = flag_shortcut

        self.conv1 = nn.utils.weight_norm(nn.Conv2d(num_channels, num*dim, kernel_size = 1, padding = 1 // 2)) #1x1 convolution
        self.conv2 = nn.utils.weight_norm(nn.Conv2d(num_channels, num, [k, k], padding = k // 2))
        self.conv3 = nn.utils.weight_norm(nn.Conv2d(dim, dim, [k, k], padding = k // 2))


    def forward(self, input_feature):
        result = self.conv1(input_feature)
        result = result.permute(0, 2, 3, 1).contiguous()
        result = result.view(result.shape[0], result.shape[1], result.shape[2], self.num, self.dim) #result.view(r[0], r[1], r[2], num, dim)

        alpha = self.conv2(input_feature)
        alpha  = alpha.permute(0, 2, 3, 1).contiguous()
        alpha = F.softmax(alpha, dim = -1)
        alpha = torch.unsqueeze(alpha, 4)

        output_feature = torch.sum(result * alpha, axis=3)
        output_feature = output_feature.permute(0, 3, 1, 2).contiguous()
        output_feature = self.conv3(output_feature)

        if self.flag_shortcut:
            return output_feature + input_feature
        else:
            return output_feature


class newCARN(nn.Module):
    def __init__(self, FLAGS, num_channels):
        super(newCARN, self).__init__()
        self.num_anchor = FLAGS.num_anchor
        self.inner_channel = FLAGS.inner_channel
        self.deep_kernel = FLAGS.deep_kernel
        self.deep_layer = FLAGS.deep_layer
        self.upscale = FLAGS.upscale

        self.conv1 = nn.utils.weight_norm(nn.Conv2d(num_channels, self.inner_channel, kernel_size = 3, padding = 3 // 2))
        self.prelu1 = torch.nn.PReLU(self.inner_channel)

        self.reg1 = newRegression_layer(self.inner_channel, self.deep_kernel,  self.upscale * self.upscale, self.num_anchor, self.inner_channel == (self.upscale * self.upscale))
        self.reg2 = newRegression_layer(self.inner_channel, self.deep_kernel,  self.inner_channel, self.num_anchor, True)
        self.reg3_list = []

        for i in range(self.deep_layer-2):
            self.reg3_list.append(Regression_layer(self.inner_channel,  self.deep_kernel,  self.inner_channel, self.num_anchor, True))

        self.reg3 = nn.Sequential(*self.reg3_list)
        self.reg4 = newRegression_layer(self.inner_channel, self.deep_kernel, self.upscale * self.upscale, self.num_anchor, self.inner_channel == (self.upscale * self.upscale))
        self.pixel_shuffle = nn.PixelShuffle(self.upscale)


    def forward(self, lr, bic):
        feature = self.prelu1(self.conv1(lr))
        reshape_size = feature.shape
        kernel_size = self.deep_kernel

        if self.deep_layer == 1:
            regression = self.reg1(feature)
        else:
            regression = self.reg2(feature)
            regression = self.reg3(regression)
            regression = self.reg4(regression)

        sr_space = self.pixel_shuffle(regression)

        sr_out = torch.add(sr_space, bic)

        return sr_out        
