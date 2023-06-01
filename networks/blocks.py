import warnings

from typing import Callable, Optional, Tuple, Union
import math

import torch
import torch.nn.utils.parametrize as P
import torch.nn.functional as F



def _normalize(x: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    norm = max(float(torch.sqrt(torch.sum(x.pow(2)))), eps)
    return torch.div(x, norm, out=x)

class ConvSpectralNorm(torch.nn.Module):
    def __init__(self, 
                weight: torch.Tensor,
                stride: Union[int, tuple] = 1,
                dilation: Union[int, tuple] = 1,
                padding: Union[int, tuple] = 1,
                im_size: int = 10,
                n_power_iterations: int = 1,
                eps: float = 1e-12):
        super().__init__()
        num_iterations_initial = 15

        self.im_size = im_size
        self.n_power_iterations = n_power_iterations
        self.eps = eps

        self.padding = padding[0]
        self.stride = stride[0]
        self.dilation = dilation[0]

        # Left singular vectors
        u = _normalize(weight.new_empty((1, weight.size(0), self.im_size, self.im_size)).normal_(0, 1), eps=self.eps)
        v = _normalize(weight.new_empty((1, weight.size(1), self.im_size, self.im_size)).normal_(0, 1), eps=self.eps)

        self.register_buffer('_u', u)
        self.register_buffer('_v', v)

        self._power_method(weight, num_iterations_initial)

    @torch.autograd.no_grad()
    def _power_method(self, weight: torch.Tensor, num_iterations: int) -> None:
        for _ in range(num_iterations):
            # Spectral norm of weight equals to `u^T * W * v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            # print('Before:', self._v.size(), self._u.size())
            self._v = _normalize(torch.nn.functional.conv_transpose2d(self._u,weight, 
                                padding=self.padding, 
                                stride=self.stride,
                                dilation=self.dilation),
                                eps=self.eps)
            self._u = _normalize(torch.nn.functional.conv2d(self._v, weight, 
                                padding=self.padding,
                                stride=self.stride,
                                dilation=self.dilation), 
                                eps=self.eps)
            # print('After:', self._v.size(), self._u.size())

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._power_method(weight, self.n_power_iterations)
        u = self._u.clone(memory_format=torch.contiguous_format)
        v = self._v.clone(memory_format=torch.contiguous_format)

        spectral_norm = torch.sum(u * torch.nn.functional.conv2d(v, weight, 
                                                        padding=self.padding,
                                                        stride=self.stride,
                                                        dilation=self.dilation))
        if spectral_norm < self.eps:
            return weight 
        else:
            return weight / spectral_norm


def convspectralnorm_wrapper(module: torch.nn.Module,
                             im_size: int=10,
                             n_power_iterations: int=1):
    return P.register_parametrization(module, "weight", 
                                ConvSpectralNorm(weight=module.weight, 
                                                 stride=module.stride,
                                                 dilation=module.dilation,
                                                 padding=module.padding,
                                                 im_size=im_size,
                                                 n_power_iterations=n_power_iterations))

class ConvBNBlock(torch.nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                padding: int=0,
                bias: bool=False,
                stride: int = 1,
                kernel_size: int = 3,
                init_lipschitz: float=math.inf,
                conv_wrapper: Optional[Callable]=None,
                clip_bn: Optional[bool]=None
                ):
        super(ConvBNBlock, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        if (conv_wrapper is not None) and (conv_wrapper is not convspectralnorm_wrapper):
            self.conv = conv_wrapper(self.conv)
        
        self.bn = torch.nn.BatchNorm2d(out_channels, affine=False)
        self.log_lipschitz = torch.nn.Parameter(torch.tensor(init_lipschitz).log())
        # print(f"self.log_lipschitz: {self.log_lipschitz}")
        self.init_convspectralnorm = True
        self.conv_wrapper = conv_wrapper
        if clip_bn is None:
            clip_bn = self.conv_wrapper is not None
        self.clip_bn = True
    def forward(self, x, freeze_bn=False, weight=None):
        if self.conv_wrapper is not None:
            if self.init_convspectralnorm:
                if self.conv_wrapper is convspectralnorm_wrapper:
                    self.conv = self.conv_wrapper(self.conv, im_size=x.size(2))
                self.init_convspectralnorm = False # NOTE: can modify this command 
        if weight is None:
            x = self.conv(x)
            if self.clip_bn:
                scale = torch.min((self.bn.running_var + self.bn.eps) ** 0.5)
                one_lipschitz_part = self.bn(x) * scale 
                x = one_lipschitz_part * torch.minimum(1 / scale, self.log_lipschitz.exp())
            else:
                x = self.bn(x)
        else:
            # if freeze_bn:
            x = self.F_conv(x, weight)
            if self.clip_bn:
                scale = torch.min((self.bn.running_var + self.bn.eps) ** 0.5)
                one_lipschitz_part = self.bn(x) * scale
                x = one_lipschitz_part * torch.minimum(1 / scale, self.log_lipschitz.exp())
            else:
                x = self.bn(x)
            # else:
            #     x = self.F_conv(x, weight)
            #     if self.clip_bn:
            #         scale = torch.min((self.bn.running_var + self.bn.eps) ** 0.5)
            #         one_lipschitz_part = self.F_batch_norm(self.bn, x, weight_bn, bias_bn) * scale
            #         x = one_lipschitz_part * torch.minimum(1 / scale, self.log_lipschitz.exp())
            #     else:
            #         x = self.bn(x)
        return x
    # def forward(self, x):
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

    def F_conv(self, x, weight):
        return F.conv2d(
            input=x,
            weight=weight,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups
        )
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

