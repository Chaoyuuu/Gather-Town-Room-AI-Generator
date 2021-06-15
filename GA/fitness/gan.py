from torch import nn


# TODO: While it's recommended to export params of model,
#       this code is the same as one in GAN/, not good.
class Discriminator(nn.Module):
    def __init__(self):
        """A discriminator for discerning real from generated images.
        Images must be single-channel and 28x28 pixels.
        Output activation is Sigmoid.
        """
        super(Discriminator, self).__init__()
        self._init_modules()

    def _init_modules(self):
        """Initialize the modules."""
        self.conv1 = nn.Conv2d(
            in_channels=17,
            out_channels=68,
            kernel_size=4,
            stride=2,
            padding=2,
            bias=False)
        # 7X6
        self.leaky_relu = nn.LeakyReLU()
        self.dropout_2d = nn.Dropout2d(0.3)

        self.conv2 = nn.Conv2d(
            in_channels=68,
            out_channels=136,
            kernel_size=2,
            stride=1,
            padding=0,
            bias=False)
        # 6X5
        self.linear1 = nn.Linear(136 * 6 * 5, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """Forward pass; map samples to confidence they are real [0, 1]."""
        intermediate = self.conv1(input_tensor)

        # print(intermediate.size())
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)

        intermediate = self.conv2(intermediate)

        # print(intermediate.size())
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout_2d(intermediate)

        intermediate = intermediate.view((-1, 136 * 6 * 5))
        intermediate = self.linear1(intermediate)
        output_tensor = self.sigmoid(intermediate)
        # print("in dis forward")
        # print(output_tensor.size())
        # print(output_tensor)
        # print("finished")
        return output_tensor