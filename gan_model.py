import copy

import torch
import torch.nn as nn
from object_dictionary import *
from utils import *


class Generator(nn.Module):
    def __init__(self, latent_dim=16, batchnorm=True):
        """A generator for mapping a latent space to a sample space.
        The sample space for this generator is single-channel, 28x28 images
        with pixel intensity ranging from -1 to +1.
        Args:
            latent_dim (int): latent dimension ("noise vector")
            batchnorm (bool): Whether or not to use batch normalization
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.batchnorm = batchnorm
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        self.linear1 = nn.Linear(self.latent_dim, 136 * 6 * 5, bias=False)
        self.bn1d1 = nn.BatchNorm1d(136 * 6 * 5)
        self.leaky_relu = nn.LeakyReLU()
        # 6X5
        self.bn2d1 = nn.BatchNorm2d(136) if self.batchnorm else None

        self.conv2 = nn.ConvTranspose2d(
            in_channels=136,
            out_channels=68,
            kernel_size=2,
            stride=1,
            padding=0,
            bias=False)
        # 7X6
        self.bn2d2 = nn.BatchNorm2d(68) if self.batchnorm else None

        self.conv3 = nn.ConvTranspose2d(
            in_channels=68,
            out_channels=17,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        # 13X11
        self.tanh = nn.Tanh()

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
        intermediate = self.linear1(input_tensor)

        intermediate = intermediate.view((-1, 136, 6, 5))

        if self.batchnorm:
            intermediate = self.bn2d1(intermediate)
        intermediate = self.leaky_relu(intermediate)

        intermediate = self.conv2(intermediate)

        if self.batchnorm:
            intermediate = self.bn2d2(intermediate)
        intermediate = self.leaky_relu(intermediate)

        intermediate = self.conv3(intermediate)

        intermediate = intermediate.narrow(3, 0, 10)
        output_tensor = self.tanh(intermediate)

        return output_tensor


def tensor_to_json_gan(output_tensor):
    # result_tensor = result_tensor[0]
    # object_dict = {
    #     0: 'WB',
    #     1: 'PR',
    #     2: 'T2',
    #     3: 'TV',
    #     4: 'B4',
    #     5: 'PP',
    #     6: 'MB',
    #     7: 'CC',
    #     8: 'CS',
    #     9: 'T3',
    #     10: 'B2',
    #     11: 'LP',
    #     12: 'MP',
    #     13: 'LB',
    #     14: 'DC'
    # }

    room_json = {
        "generator": "gan",
        "room": []
    }

    threshold = 0.7
    enough = False

    while not enough:
        threshold -= 0.05
        num = 0
        for row in range(13):
            for col in range(10):
                max_value = 0
                max_index = -1
                for depth in range(15):
                    data = output_tensor[depth][row][col]

                    if data >= threshold and max_value < data:
                        max_index = depth
                        max_value = data

                if max_index != -1:
                    num += 1
                    object_name = object_name_dict[max_index]
                    orientation = decode_orientation(max_index, output_tensor[15][row][col],
                                                     output_tensor[16][row][col])
                    print("orien:" + str(orientation) + " name:" + object_name)
                    object = copy.deepcopy(object_dict[object_name][orientation])
                    object["x"] = shift_object(max_index, col, orientation)
                    object["y"] = row
                    room_json["room"].append(object)

        if num > 10:
            enough = True

    return room_json
