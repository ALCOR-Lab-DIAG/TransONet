'''
Codes are from:
https://github.com/jaxony/unet-pytorch/blob/master/model.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
from src.utils.dsconv import DSConv2d
import numpy as np
from ..models.tinyVit import TinyViT
from ..models.mvit_model import MViT
from ..models.dyvit import VisionTransformerDiffPruning, VisionTransformerTeacher
from ..models.losses import DistillDiffPruningLoss_dynamic
from ..models.fast_quant import fast_quant
from ..models.generic_transformer import Transformer
import pdb

def ds_conv3d(in_channels, out_channels, depth = None, stride=1,
            padding=1, bias=True, groups=1):
    ##print('ds conv3d')
    depth_conv =nn.Conv2d(in_channels,
        in_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)
    ##print(depth_conv.weight)
    point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    if depth:
        name1 = 'conv2d({})'.format(depth)
        name2 = 'conv1d({})'.format(depth)
        depthwise_separable_conv = nn.Sequential(OrderedDict([
                                                    (name1,depth_conv),
                                                    (name2,point_conv)]))
    else:
        depthwise_separable_conv = nn.Sequential(depth_conv,point_conv)
    return depthwise_separable_conv

def cp_conv(in_channels, out_channels, depth = None, stride=1,
            padding=1, bias=True, groups=1):

    first = nn.Conv2d(in_channels, in_channels, kernel_size=(1,1),
                      stride=stride,padding=0, bias=bias)
    last = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1),
                      stride=stride,padding=0, bias=bias)
    expanded = nn.Unflatten(1,(in_channels,1))
    mid1 = nn.Conv3d(in_channels,in_channels,kernel_size=(3,1,1), stride = stride,
                     padding = (padding,0,0), bias = bias, groups=groups)
    mid2 = nn.Conv3d(in_channels,in_channels,kernel_size=(1,3,1), stride=stride, bias = bias,
                     padding = (0,padding,0), groups=groups)
    mid3 = nn.Conv3d(in_channels,in_channels,kernel_size=(1,1,3), stride=stride, bias = bias,
                     padding = (0,0,padding), groups=groups)
    squeezed = nn.Flatten(1,2)

    if depth:
        first_name = 'conv2d1({})'.format(depth)
        last_name = 'conv2d2({})'.format(depth)
        mid1_name = 'conv3d1({})'.format(depth)
        mid2_name = 'conv3d2({})'.format(depth)
        mid3_name = 'conv3d3({})'.format(depth)
        cp_decomposed_conv = nn.Sequential(OrderedDict([
                    (first_name,first),
                    ('expansion',expanded),
                    (mid1_name,mid1),
                    (mid2_name,mid2),
                    (mid3_name,mid3),
                    ('squeeze',squeezed),
                    (last_name,last)]))
    else:
        cp_decomposed_conv = nn.Sequential(first,
                      expanded,
                      mid1,
                      mid2,
                      mid3,
                      squeezed,
                      last)
        #cp_decomposed_conv = Winograd(in_channels,out_channels,groups=groups)
    return cp_decomposed_conv

def dsconv2d(in_channels, out_channels, kernel_size=3, depth=None, block_size=32, stride=1,
                 padding=0, groups=1, bias=False):
    dsconv = DSConv2d(in_channels,out_channels,kernel_size,block_size,stride,padding,groups,bias)
    if depth:
        name1 = 'dsconv2d({})'.format(depth)
        DS_conv = nn.Sequential(OrderedDict([(name1,dsconv)]))
    else:
        DS_conv = nn.Sequential(dsconv)
    return DS_conv


def conv3x3(in_channels, out_channels, depth = None, stride=1,
            padding=1, bias=True, groups=1, upconv = False):

    #print(depth)
    #depthwise_separable_conv = cp_conv(64,
    #        32,
    #        stride=stride,
    #        padding=padding,
    #        bias=bias,
    #        groups=groups,
    #        depth = depth)
    #params_depthwise = sum(p.numel() for p in depthwise_separable_conv.parameters() if p.requires_grad)
    #conv = nn.Conv2d(
    #    64,
    #    32,
    #    kernel_size=3,
    #    stride=stride,
    #    padding=padding,
    #    bias=bias,
    #    groups=groups)
    ##print(in_channels,out_channels)
    #params = sum(p.numel() for p in conv.parameters() if p.requires_grad)
    #x = torch.rand(64, 2048, 3)
    #out = conv(x)
    #params_depthwise = sum(p.numel() for p in depthwise_separable_conv.parameters() if p.requires_grad)

    #out_depthwise = depthwise_separable_conv(x)
    ##print(f"The standard convolution uses {params} parameters.")
    ##print(f"The depthwise separable convolution uses {params_depthwise} parameters.")

    #assert out.shape == out_depthwise.shape, "Size mismatch"

    if in_channels < out_channels:
        return ds_conv3d(in_channels,
            out_channels,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
            depth = depth)
        #return nn.Conv2d(
        #    in_channels=in_channels,
        #    out_channels=out_channels,
        #    kernel_size=3,
        #    stride=stride,
        #    padding=padding,
        #    bias=bias,
        #    groups=groups)
    #elif ((upconv and (depth == 0 or depth == 2)) or (not upconv and depth == 3)):
    elif (upconv and depth == 0):# or (not upconv and depth == 3):
        #print("here")
        return conv1x1(in_channels=in_channels,out_channels=out_channels, groups=groups)

    else:
        #return nn.Conv2d(in_channels,
        #                out_channels,
        #                kernel_size=3,
        #                stride=stride,
        #                padding=padding,
        #                bias=bias,
        #                groups=groups)
        return cp_conv(in_channels,
                        out_channels,
                        stride = stride,
                        bias=bias,
                        groups = groups,
                        depth=depth)






def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(

        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)
# def conv3x3(in_channels, out_channels, stride=1,
#             padding=1, bias=True, groups=1):
#     return nn.Conv2d(
#         in_channels,
#         out_channels,
#         kernel_size=3,
#         stride=stride,
#         padding=padding,
#         bias=bias,
#         groups=groups)
#
# def upconv2x2(in_channels, out_channels, mode='transpose'):
#     if mode == 'transpose':
#         return nn.ConvTranspose2d(
#             in_channels,
#             out_channels,
#             kernel_size=2,
#             stride=2)
#     else:
#         # out_channels is always going to be the same
#         # as in_channels
#         return nn.Sequential(
#             nn.Upsample(mode='bilinear', scale_factor=2),
#             conv1x1(in_channels, out_channels))
#
# def conv1x1(in_channels, out_channels, groups=1):
#     return nn.Conv2d(
#         in_channels,
#         out_channels,
#         kernel_size=1,
#         groups=groups,
#         stride=1)

class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, depth=None, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels, depth=depth)
        self.conv2 = conv3x3(self.out_channels, self.out_channels, depth=depth)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        ##print('x '+str(x.shape))
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels,  depth=None,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, 
            mode=self.up_mode)

        if self.merge_mode == 'concat':
            ##print(2*out_channels)
            self.conv1 = conv3x3(
                3*self.out_channels, self.out_channels, depth=depth, upconv=True)
        else:
            # num of input channels to conv2 is same
            ##print(out_channels)
            self.conv1 = conv3x3(self.out_channels, self.out_channels, depth=depth, upconv=True)
        self.conv2 = conv3x3(self.out_channels, self.out_channels, depth=depth, upconv=True)


    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        #print('from up ' + str(from_up.shape))
        #print('from down ' + str(from_down.shape))
        from_up = self.upconv(from_up)
        ##print('in channels ' + str(self.in_channels ))
        ##print('out channels ' + str(self.out_channels))

        #print('merge mode:'+self.merge_mode)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        #print("after merge:" + str(x.shape))
        x = F.relu(self.conv1(x))
        #print('x after conv 1' + str(x.shape))
        x = F.relu(self.conv2(x))
        #print('x after conv 2' + str(x.shape))
        return x


class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=3, depth=5, 
                 start_filts=64, up_mode='transpose', 
                 merge_mode='concat', **kwargs):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        #print('depth: '+ str(depth))
        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.trans = Transformer(vis=False).cuda()


        self.down_convs = []
        self.up_convs = []


        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False
            
            down_conv = DownConv(ins, outs, i, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        #outs=256#384
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs,depth=i, up_mode=up_mode,
                merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)

        self.up_convs = nn.ModuleList(self.up_convs)

        self.conv_final = conv1x1(outs, self.num_classes)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        #print(m.bias)
        if (isinstance(m, nn.Conv2d) and m.bias is not None) : # and m.bias == True:
            #print(type(m.bias))
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x):
        #print('inside unet')
        encoder_outs = []
        # encoder pathway, save outputs for merging
        ##print(self.down_convs)
        #x = torch.tensor(0)
        for i, module in enumerate(self.down_convs):
            #print("out channels:"+ str(module.out_channels))
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        #print("tensor beofre trans:" + str(x.size()))
        #self.pl_size = x.shape[2]
        #print("net beofre trans:" + str(encoder_outs))
        ##print(self.down_convs)
        bchw = encoder_outs[-2]
        ##print("x size:"+ str(x.size()))
        ##print("x type:"+str(x.type()))
        #print("bchw shape:"+str(bchw.shape))
        #trans = MViT(x.shape[1])
        #print('UNET')
        #self.pl_size = 224
        #print(self.pl_size)
        # self.trans = MViT().to('cuda:0')
        # before_trans = x
        #print('before trans {}'.format(before_trans))
        '''dyvit'''
        # PRUNING_LOC = [3, 6, 9]
        # self.KEEP_RATE = [0.9, 0.9 ** 2, 0.9 ** 3]
        # trans = VisionTransformerDiffPruning(
        #     img_size=x.shape[2], patch_size=2, embed_dim=128, depth=3, in_chans=128, num_classes=128*64, num_heads=8, mlp_ratio=4, qkv_bias=True,
        #     pruning_loc=PRUNING_LOC, token_ratio=self.KEEP_RATE, distill=True, drop_path_rate=0.0
        # ).cuda()
        #
        # loss = torch.nn.CrossEntropyLoss()
        # teacher_model = VisionTransformerTeacher(
        #    img_size=x.shape[2], patch_size=2, embed_dim=128, depth=3, in_chans=128, num_classes=128*64, num_heads=8, mlp_ratio=4, qkv_bias=True).cuda()
        #
        # criterion = DistillDiffPruningLoss_dynamic(
        #     self.teacher_model, self.loss, clf_weight=0.0, keep_ratio=self.KEEP_RATE, mse_token=True,
        #     ratio_weight=2.0, distill_weight=0.5
        # )

        #print(trans)

        #print("x after down conv:"+str(x.shape))
        #x = torch.unsqueeze(x, 0)
        #x= trans(x)#[0])

        x = x.flatten(2)
        #print("x after flatten:" + str(x.shape))
        x = x.transpose(-1,-2)
        #print("x after transpose:" + str(x.shape))
        x, _ = self.trans(x)
        # x, token_pred, mask, out_pred_score = outputs[0], outputs[1], outputs[2], outputs[3]
        B, N, C = x.shape
        #H*W = N
        x = x.permute(0,2,1)

        x = x.contiguous().view(B,C,16,16)
        #print("x after trans:" + str(x.shape))
        # x = x.reshape(B,int(H/2),int(W/2),C).permute(0,3,1,2).contiguous()
        '''dyvit'''
        # trans = TinyViT(img_size= x.shape[2], embed_dims = [128, 128, 128], in_chans = 128, num_classes = 128*64,
        #                 depths = [3,3], num_heads = [1, 2, 4, 8], window_sizes = [8, 8, 16, 8], mlp_ratio = 4. ).cuda()

        #before_pool = x
        #trans = fast_quant(trans, with_noisy_quant=True, percentile=True, search_mean=True, search_noisy=True)
        #x = trans(x)
        #print("x after trans:" + str(x.shape))
        #x = x.reshape(B,int(H/2),int(W/2),C).permute(0,3,1,2).contiguous()

        '''dyvit'''
        # loss = self.criterion(before_trans,outputs)
        '''dyvit'''
        encoder_outs.append(x)
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        x = pool(x)
        #x = pool(x)
        ##print("x after pool:" + str(x.shape))
        ##print("net after mvit:" + str(encoder_outs))
        #print("tensor after mvit:" + str(x.shape))
        for i, module in enumerate(self.up_convs):
            ##print(-(i+2))
            #print("out channels:" + str(module.out_channels))
            #print("in channels:" + str(module.in_channels))
            #print(-(i+2))
            before_pool = encoder_outs[-(i+2)]
            #print('from down:{}, from up:{} '.format(before_pool.shape,x.shape))
            x = module(before_pool, x)

        ##print(x.size())
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        ##print('x: '+str(x))
        return x#, loss

if __name__ == "__main__":
    """
    testing
    """
    ##print('main')
    model = UNet(1, depth=2, merge_mode='concat', in_channels=1, start_filts=32)
    #print(model)
    #print(sum(p.numel() for p in model.parameters()))

    reso = 176
    x = np.zeros((1, 1, reso, reso))
    x[:,:,int(reso/2-1), int(reso/2-1)] = np.nan
    x = torch.FloatTensor(x)

    out, loss = model(x), loss
    #print('%f'%(torch.sum(torch.isnan(out)).detach().cpu().numpy()/(reso*reso)))
    
    # loss = torch.sum(out)
    # loss.backward()
