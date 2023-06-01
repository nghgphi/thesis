import sys
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from networks.blocks import ConvBNBlock, convspectralnorm_wrapper
from networks.activation import ParametricSoftplus

class Learner(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Learner, self).__init__()
        print("Initialize alexnet_spectral network.")
        self.args = args
        n_ch, size, _ = n_inputs

        if args.use_conv_wrapper:
            print("Use conv_wrapper = convspectralnorm_wrapper")
            self.conv_wrapper = convspectralnorm_wrapper

        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
        self.kernel_size = []

        self.conv1 = ConvBNBlock(
            in_channels=n_ch,
            out_channels=64,
            kernel_size=size // 8,
            init_lipschitz=10.0,
            conv_wrapper=self.conv_wrapper,
            clip_bn=args.clip_bn
        )

        # self.conv1 = torch.nn.Conv2d(n_ch, 64, kernel_size=size // 8, bias=False)
        self.kernel_size.append(size // 8)
        s = self.compute_conv_output_size(size, size // 8)
        s = s // 2

        self.conv2 = ConvBNBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=size // 10,
            init_lipschitz=10.0,
            conv_wrapper=self.conv_wrapper,
            clip_bn=args.clip_bn
        )
        # self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=size // 10, bias=False)
        # if self.args.use_track:
        #     self.bn2 = torch.nn.BatchNorm2d(128, momentum=0.1)
        # else:
        #     self.bn2 = torch.nn.BatchNorm2d(128, track_running_stats=False)
        self.kernel_size.append(size // 10)
        s = self.compute_conv_output_size(s, size // 10)
        s = s // 2

        self.conv3 = ConvBNBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=2,
            init_lipschitz=10.0,
            conv_wrapper=self.conv_wrapper,
            clip_bn=args.clip_bn
        )
        # self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=2, bias=False)
        self.kernel_size.append(2)
        s = self.compute_conv_output_size(s, 2)
        s = s // 2
        self.maxpool = torch.nn.MaxPool2d(2)

        # self.relu = torch.nn.ReLU()
        self.activation_fn_1 = ParametricSoftplus(init_beta=2.0, threshold=10)
        self.activation_fn_2 = ParametricSoftplus(init_beta=2.0, threshold=10)
        self.activation_fn_3 = ParametricSoftplus(init_beta=2.0, threshold=10)


        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(256 * s * s, 2048, bias=False)
        self.fc2 = torch.nn.Linear(2048, 2048, bias=False)
        
        self.head = torch.nn.Linear(2048, self.n_outputs, bias=False)
        
        # transform linear layers by spectral_norm operation
        self.fc1 = torch.nn.utils.parametrizations.spectral_norm(self.fc1)
        self.fc2 = torch.nn.utils.parametrizations.spectral_norm(self.fc2)
        # self.head = torch.nn.utils.parametrizations.spectral_norm(self.head)
        # number of representation matrix
        self.n_rep = 5
        self.multi_head = True
        self.freeze_bn = args.freeze_bn

    def forward(self, x, vars=None, svd=False):
        if svd:
            y = []
            fs, h = self.conv_to_linear(x, self.conv1)
            y.append(fs)

            h = self.maxpool(self.drop1(self.activation_fn_1(h)))
            fs, h = self.conv_to_linear(h, self.conv2)
            y.append(fs)

            h = self.maxpool(self.drop1(self.activation_fn_2(h)))
            fs, h = self.conv_to_linear(h, self.conv3)
            y.append(fs)

            h = self.maxpool(self.drop2(self.activation_fn_3(h)))
            h = h.reshape(x.size(0), -1)
            y.append(h)
            h = self.drop2((self.fc1(h)))
            y.append(h)
        elif vars is not None:
            assert len(vars) == 6, f"len(vars) = {len(vars)} != 6"
            h = self.maxpool(self.drop1(self.activation_fn_1(
                self.conv1(x, freeze_bn=False, weight=vars[0])
            )))
            h = self.maxpool(self.drop1(self.activation_fn_2(
                self.conv2(h, weight=vars[1])
            )))
            h = self.maxpool(self.drop2(self.activation_fn_3(
                self.conv3(h, weight=vars[2])
            )))
            h = h.reshape(x.size(0), -1)
            h = self.drop2(F.linear(h, vars[3]))
            h = self.drop2((F.linear(h, vars[4])))
            y = F.linear(h, vars[5])
            # if self.freeze_bn and len(vars) == 6:
            #     h = self.maxpool(self.drop1(self.relu(self.F_conv(self.conv1, x, vars[0]))))
            #     h = self.maxpool(self.drop1(self.relu(self.bn2(self.F_conv(self.conv2, h, vars[1])))))
            #     h = self.maxpool(self.drop2(self.relu(self.F_conv(self.conv3, h, vars[2]))))

            #     h = h.reshape(x.size(0), -1)

            #     h = self.drop2(self.relu(F.linear(h, vars[3])))
            #     h = self.drop2(self.relu(F.linear(h, vars[4])))

            #     y = F.linear(h, vars[5])
            # else:
            #     assert len(vars) == 8

            #     h = self.maxpool(self.drop1(self.relu(self.F_conv(self.conv1, x, vars[0]))))
            #     h = self.F_conv(self.conv2, h, vars[1])
            #     h = self.F_batch_norm(h, weight=vars[2], bias=vars[3])
            #     h = self.maxpool(self.drop1(self.relu(h)))
            #     h = self.maxpool(self.drop2(self.relu(self.F_conv(self.conv3, h, vars[4]))))

            #     h = h.reshape(x.size(0), -1)

            #     h = self.drop2(self.relu(F.linear(h, vars[5])))
            #     h = self.drop2(self.relu(F.linear(h, vars[6])))
            #     y = F.linear(h, vars[7])
        else:
            h = self.maxpool(self.drop1(self.activation_fn_1(self.conv1(x))))
            h = self.maxpool(self.drop1(self.activation_fn_2(self.conv2(h))))
            h = self.maxpool(self.drop2(self.activation_fn_3(self.conv3(h))))
            h = h.reshape(x.size(0), -1)
            h = self.drop2(self.fc1(h))
            h = self.drop2(self.fc2(h))
            y = self.head(h)

        return y

    def compute_conv_output_size(self, size, kernel_size, stride=1, padding=0, dilation=1):
        return int(np.floor((size + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))

    def conv_to_linear(self, x, conv: ConvBNBlock, batchsize=None):
        kernel = conv.conv.kernel_size
        stride = conv.conv.stride
        padding = conv.conv.padding

        if batchsize is None:
            batchsize = x.shape[0]
        else:
            batchsize = min(batchsize, x.shape[0])

        assert batchsize > 0

        if padding[0] > 0 or padding[1] > 0:
            y = torch.zeros((batchsize, x.shape[1], x.shape[2] + 2 * padding[0], x.shape[3] + 2 * padding[1]))
            y[:, :, padding[0]: x.shape[2] + padding[0], padding[1]: x.shape[2] + padding[1]] = x[:batchsize]
        else:
            y = x[:batchsize]

        h = y.shape[2]
        w = y.shape[3]
        kh = kernel[0]
        kw = kernel[1]

        fs = []

        for i in range(0, h, stride[0]):
            for j in range(0, w, stride[1]):
                if i + kh > h or j + kw > w:
                    break
                f = y[:, :, i:i + kh, j:j + kw]
                f = f.reshape(batchsize, 1, -1)
                if i == 0 and j == 0:
                    fs = f
                else:
                    fs = torch.cat((fs, f), 1)

        fs = fs.reshape(-1, fs.shape[-1])
        h = conv(x)

        if self.args.cuda:
            fs = fs.cuda()
            h = h.cuda()

        assert fs.shape[0] == batchsize * h.shape[2] * h.shape[3]
        assert fs.shape[1] == x.shape[1] * conv.conv.kernel_size[0] * conv.conv.kernel_size[1]

        return fs, h

    def get_params(self):
        self.vars = []
        for p in list(self.parameters()):
            if p.requires_grad and p.dim() > 0:
                self.vars.append(p)
        return self.vars

    def F_conv(self, conv, x, weight):
        return F.conv2d(x, weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)

    def F_batch_norm(self, bn, x, weight, bias):
        if bn.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = bn.momentum

        if bn.training and bn.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if bn.num_batches_tracked is not None:
                bn.num_batches_tracked = bn.num_batches_tracked + 1
                if bn.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = bn.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if bn.training:
            bn_training = True
        else:
            bn_training = (bn.running_mean is None) and (bn.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            bn.running_mean if not bn.training or bn.track_running_stats else None,
            bn.running_var if not bn.training or bn.track_running_stats else None,
            weight, bias, bn_training, exponential_average_factor, bn.eps)
