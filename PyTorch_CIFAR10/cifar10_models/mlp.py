import torch
from torch import nn

class MLP(nn.Module):
    def __init__(
        self,
        in_dim=784, # MNIST
        num_classes=10,
        out_channels=[10000, 512, 64],
        split_layer=-1,
        bottleneck_dim=-1
    ):
        super(MLP, self).__init__()
        layers = []
        in_channel = in_dim
        for out_channel in out_channels:
            layers.append(nn.Linear(in_channel, out_channel))
            layers.append(nn.ReLU(inplace=True))
            in_channel = out_channel
        layers.append(nn.Linear(in_channel, num_classes))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
        self.split_layer = split_layer

    def set_split_layer(self, layer):
        self.split_layer = layer

    def split_forward(self, x, sigma=None, return_act=False):
        # When calculating jacobian, this is called w/o the batch dim
        # if len(x.shape) == 4:
        #     x = x.reshape([x.shape[0], -1])
        # elif len(x.shape) == 3:
        #     x = x.reshape([1, -1])
        # else:
        #     assert(False)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == self.split_layer:
                if sigma is not None:
                    x = x + torch.stack([torch.normal(torch.zeros_like(x[j]), sigma[j]) for j in range(len(sigma))])
                if return_act:
                    return x

        return x

    def forward(self, x):
        return self.split_forward(x)
    
    # def forward_for_jacobian(self, x):
    #     return self.split_forward(x, return_act=True)

    # def set_bn_training(self, training):
    #     pass
