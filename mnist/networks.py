import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

import sys
from options import args
torch.manual_seed(0)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)
class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)
#-********************************************
# CNN
class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = norm(in_planes)
        self.conv2 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = norm(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        return out + x

class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        downsampling_layers = [
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True)
            ]
        feature_layers = [ResBlock(64, 64) for _ in range(1)]
        fc_layers = [nn.AdaptiveAvgPool2d((1, 1)), 
            Flatten(), nn.Linear(64, 10)
            ]
        self.net = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers)

    def forward(self, x):
        return self.net(x)

#-**********************************************************************************-#
# ODE models
sys.path.append('..')
import torchdiffeq._impl.odeint as odeint

class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class ODEBlock(nn.Module):
    def __init__(self, odefunc, t=[0,1]):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor(t).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, 
            rtol=args.rtol, atol=args.atol, 
            method='euler', options={'step_size':args.step_size})
        return out[1] # Two time points are stored in [0,1], here the state at time 1 is output.
           
class ODEfunc(nn.Module):
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm1 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
    def forward(self, t, x):
        out = self.conv1(t, x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        return out

# 原来的版本
class ODENet_MNIST(nn.Module):
    def __init__(self):
        super(ODENet_MNIST, self).__init__()
        downsampling_layers = [
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True)
            ]
        feature_layers = [ODEBlock(ODEfunc(64), args.TimePeriod)]
        fc_layers = [nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(), nn.Linear(64, 10)
            ]
        self.net = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers)

    def forward(self, x):
        return self.net(x)

# 自己修改的版本
# class ODENet_MNIST(nn.Module):
#     def __init__(self):
#         super(ODENet_MNIST, self).__init__()
#         self.conv0 = nn.Conv2d(1, 64, 3, 1)
#         self.norm0 = norm(64)
#         self.relu0 = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(64, 64, 4, 2, 1)
#         self.norm1 = norm(64)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.odeblock = ODEBlock(ODEfunc(64), args.TimePeriod)
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.flat = Flatten()
#         self.linear = nn.Linear(64, 10)
#
#     def forward(self, x):
#         out = self.conv0(x)
#         out = self.norm0(out)
#         out = self.relu0(out)
#
#         out = self.conv1(out)
#         out = self.norm1(out)
#         out = self.relu1(out)
#
#         out = self.odeblock(out)
#
#         out = self.pool(out)
#         out = self.flat(out)
#         out = self.linear(out)
#
#         return out


# 添加傅里叶变换之后的版本
class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

# class ODENet_MNIST(nn.Module):
#     def __init__(self):
#         super(ODENet_MNIST, self).__init__()
#         self.fconv0 = SpectralConv2d_fast(1, 64, 15, 15)
#         self.conv0 = nn.Conv2d(1, 64, 1, 1)
#         self.norm0 = norm(64)
#         self.relu0 = nn.ReLU(inplace=True)
#         self.fconv1 = SpectralConv2d_fast(64, 64, 15, 15)
#         self.conv1 = nn.Conv2d(64, 64, 1, 1)
#         self.norm1 = norm(64)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.odeblock = ODEBlock(ODEfunc(64), args.TimePeriod)
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.flat = Flatten()
#         self.linear = nn.Linear(64, 10)
#
#     def forward(self, x):
#         out00 = self.fconv0(x)
#         out01 = self.conv0(x)
#         out = out00 + out01
#         out = self.norm0(out)
#         out = self.relu0(out)
#
#         out10 = self.fconv1(out)
#         out11 = self.conv1(out)
#         out = out10 + out11
#         out = self.norm1(out)
#         out = self.relu1(out)
#
#         out = self.odeblock(out)
#
#         out = self.pool(out)
#         out = self.flat(out)
#         out = self.linear(out)
#
#         return out



#-********************************************
# TisODE - new version codes
class ODEfunc_tisode(nn.Module):
    def __init__(self, dim):
        super(ODEfunc_tisode, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = norm(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = norm(dim)
    def forward(self, t, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        return out
    
class ODENet_tisode(nn.Module):
    def __init__(self):
        super(ODENet_tisode, self).__init__()
        self._set_integration_params_(args.TimePeriod)

        self.downsampling_layers = nn.Sequential(*[
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True)
            ])
        self.feature_layers = ODEfunc_tisode(64)
        self.fc_layers = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), 
            Flatten(), nn.Linear(64, 10)
            ])
    def _set_integration_params_(self, t):
        assert len(t) == 4
        self.integration_time = torch.tensor(t).to(torch.float32)
        self.rtol = args.rtol
        self.atol = args.atol
        self.step_size = args.step_size
        self.method = 'euler'
        self.method_abs = 'euler_abs'
    def forward(self, x):
        # prediction
        x_d = self.downsampling_layers(x)
        _, x_f_1, x_f = odeint(self.feature_layers, x_d, self.integration_time[0:3], self.rtol, self.atol, 
                    method=self.method, options={'step_size':self.step_size})
        pred = self.fc_layers(x_f)
        # # steady state
        _, _, x_f_2 = odeint(self.feature_layers, x_f_1, self.integration_time[1:4], self.rtol, self.atol, 
                    method=self.method_abs, options={'step_size':self.step_size})
        x_steady_state_target = torch.zeros_like(x_f, requires_grad=False)
        x_steady_diff = x_f_2 - x_f_1
        return pred, x_steady_state_target, x_steady_diff


if __name__ == '__main__':
    def test():
        net = ODEfunc_tiv(64)
        x = torch.randn(1,64,3,3)
        import pdb; pdb.set_trace()
        print('#param:', count_parameters(net))

    test()
    pass
