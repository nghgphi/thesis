import sys
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from networks.blocks import ConvBNBlock, ConvSpectralNorm, convspectralnorm_wrapper
from networks.activation import get_activation_obj, ParametricSoftplus

class Learner(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Learner, self).__init__()

        self.args = args
        n_channels, size, _ = n_inputs

        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
        self.kernel_size = []
        self.clip_bn = args.clip_bn
        if args.use_conv_wrapper:
            print("Use conv_wrapper = convspectralnorm_wrapper")
            self.conv_wrapper = convspectralnorm_wrapper
        
        # self.conv1 = nn.Conv2d(n_ch, 20, 5, bias=False, padding=2)
        # if self.args.use_track:
        #     self.bn1 = nn.BatchNorm2d(20, momentum=0.1)
        # else:
        #     self.bn1 = nn.BatchNorm2d(20, track_running_stats=False)
        self.conv1 = ConvBNBlock(
            in_channels=n_channels,
            out_channels=20,
            padding=2,
            bias=False,
            init_lipschitz=5.0,
            conv_wrapper=self.conv_wrapper,
            clip_bn=self.clip_bn
        )
        

        s = self.compute_conv_output_size(size, 5, 1, 2)
        s = self.compute_conv_output_size(s, 3, 2, 1)

        # self.conv2 = nn.Conv2d(20, 50, 5, bias=False, padding=2)
        # if self.args.use_track:
        #     self.bn2 = nn.BatchNorm2d(50, momentum=0.1)
        # else:
        #     self.bn2 = nn.BatchNorm2d(50, track_running_stats=False)
        self.conv2 = ConvBNBlock(
            in_channels=20,
            out_channels=50,
            init_lipschitz=5.0,
            conv_wrapper=self.conv_wrapper,
            clip_bn=self.clip_bn,
            bias=False
        )

        s = self.compute_conv_output_size(s, 5, 1, 2)
        s = self.compute_conv_output_size(s, 3, 2, 1)

        self.maxpool = torch.nn.MaxPool2d(3, 2, padding=1)
        self.relu = torch.nn.ReLU()

        self.drop1 = torch.nn.Dropout(0)
        self.drop2 = torch.nn.Dropout(0)

        self.fc1 = nn.Linear(50 * s * s, 800, bias=False)
        self.fc2 = nn.Linear(800, 500, bias=False)

        if args.activation_name is not None:
            self.fc1 = torch.nn.utils.parametrizations.spectral_norm(self.fc1)
            self.fc2 = torch.nn.utils.parametrizations.spectral_norm(self.fc2)


        self.head = torch.nn.Linear(500, self.n_outputs, bias=False)

        self.activation_fn_1 = ParametricSoftplus()
        self.activation_fn_2 = ParametricSoftplus()
        self.activation_fn_3 = ParametricSoftplus()
        self.activation_fn_4 = ParametricSoftplus()


        # number of representation matrix
        self.n_rep = 4
        self.multi_head = True
        self.freeze_bn = args.freeze_bn

    def forward(self, x, vars=None, svd=False):
        """
            - x: training data, used to calculate SVD
            - vars: variables, If not None, it will calculate based on vars, used to inner update
            - feature: used to get representation matrix
                        If true, return is the representation matrix for each layer;
                        Otherwise, return is the output for last layer
            """
        if svd:
            y = []
            fs, h = self.conv_to_linear(x, self.conv1)
            y.append(fs)

            h = self.maxpool(self.drop1(self.activation_fn_2(h)))
            fs, h = self.conv_to_linear(h, self.conv2)
            y.append(fs)

            h = self.maxpool(self.drop1(self.activation_fn_3(h)))
            h = h.reshape(x.size(0), -1)

            y.append(h)
            h = self.drop2(self.activation_fn_4(self.fc1(h)))
            y.append(h)
        elif vars is not None:
            # print(f"len(vars) = {len(vars)}")

            # if self.freeze_bn and len(vars) == 5:  
            #     print("Case 1")
            #     # h = self.maxpool(self.drop1(self.activation_fn(self.bn1(self.F_conv(self.conv1, x, vars[0])))))
            #     # h = self.maxpool(self.drop1(self.activation_fn(self.bn2(self.F_conv(self.conv2, h, vars[1])))))
            #     h = self.maxpool(self.drop1(self.activation_fn(
            #         self.conv1(x, freeze_bn=True, weight=vars[0])
            #     )))
            #     h = self.maxpool(self.drop1(self.activation_fn(
            #         self.conv2(h, freeze_bn=True, weight=vars[1])
            #     )))
                
            #     h = h.reshape(x.size(0), -1)

            #     h = self.drop2(self.activation_fn(F.linear(h, vars[2])))
            #     h = self.drop2(self.activation_fn(F.linear(h, vars[3])))

            #     y = F.linear(h, vars[4])
            # else:
            # print("Case 2")
            assert len(vars) == 5, f"len(vars) = {len(vars)} != 5" # in this case, batch_norm weights are update
            
            # h = self.F_conv(self.conv1, x, vars[0])
            # h = self.F_batch_norm(self.bn1, h, weight=vars[1], bias=vars[2])
            # h = self.maxpool(self.drop1(self.activation_fn(h)))
            # h = self.F_conv(self.conv2, h, vars[3])
            # h = self.F_batch_norm(self.bn2, h, weight=vars[4], bias=vars[5])
            # h = self.maxpool(self.drop1(self.activation_fn  (h)))
            h = self.maxpool(self.drop1(self.activation_fn_1(
                self.conv1(x, freeze_bn=False, weight=vars[0])
            )))
            h = self.maxpool(self.drop2(self.activation_fn_2(
                self.conv2(h, weight=vars[1])
            )))

            h = h.reshape(x.size(0), -1)

            h = self.drop2(self.activation_fn_3(F.linear(h, vars[2])))
            h = self.drop2(self.activation_fn_4(F.linear(h, vars[3])))

            y = F.linear(h, vars[4])
        else:
            # print("Case 3")
            h = self.maxpool(self.drop1(self.activation_fn_1(self.conv1(x))))
            h = self.maxpool(self.drop1(self.activation_fn_2(self.conv2(h))))

            h = h.reshape(x.size(0), -1)

            h = self.drop2(self.activation_fn_3(self.fc1(h)))
            h = self.drop2(self.activation_fn_4(self.fc2(h)))

            y = self.head(h)
        return y
    
    #    def forward(self, x):
    #     if self.conv_wrapper is not None:
    #         # initialize conv spectral norm taking into account the size of the input
    #         if self.init_convspectralnorm is True:
    #             if self.conv_wrapper is models.conv_spectral_norm.convspectralnorm_wrapper:
    #                 self.conv = self.conv_wrapper(self.conv, im_size=x.size(2))
    #             self.init_convspectralnorm = False
    #     x = self.conv(x)
    #     if self.clip_bn:
    #         scale = torch.min((self.bn.running_var + self.bn.eps) ** .5)
    #         one_lipschitz_part = self.bn(x) * scale
    #         x = one_lipschitz_part * torch.minimum(1 / scale, self.log_lipschitz.exp())
    #     else:
    #         x = self.bn(x)
    #     return x

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
        # print("Using l/enet_spectral")
        for name, p in list(self.named_parameters()):
            if p.requires_grad and p.dim() > 0:
                # print(f"name: {name}, {p.data.shape}")
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
