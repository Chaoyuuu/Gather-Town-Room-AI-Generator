import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, x_dim))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


VAE_PATH = './VAE/model/vae-test'
def load_vae_model():
    loaded = torch.load(VAE_PATH, map_location=torch.device('cpu'))
    model = loaded['model']
    return model


object_name_dict = {
    0: 'Whiteboard',
    1: 'Projector Screen',
    2: 'Chippendale Table (2x3)',
    3: 'TV (Flatscreen)',
    4: 'Bookshelf (2x4)',
    5: 'Potted Plant (Spikey)',
    6: 'Mod Chair',
    7: 'Captain’s Chair',
    8: 'Chair (Simple)',
    9: 'Chippendale Table (3x3)',
    10: 'Bookshelf [Tall] (1x2)',
    11: 'Laptop',
    12: 'Microphone',
    13: 'Lucky Bamboo',
    14: 'Dining Chair (Square)'
}


def tensor_to_json(sample, threshold):
    output_tensor = torch.zeros((17, 13, 10))

    for k, batch in enumerate(sample):
        i = 0
        for depth in range(17):
            for col in range(13):
                for row in range(10):
                    data = batch[i]
                    i += 1
                    if data >= threshold:
                        output_tensor[depth][col][row] = data

    return encode_to_json(output_tensor, threshold)


def encode_to_json(output_tensor, threshold):
    room_json = {
        "generator": "vae",
        "room": []
    }

    for col in range(13):
        for row in range(10):
            max_value = 0
            max_index = -1
            for depth in range(15):
                data = output_tensor[depth][col][row]

                if data >= threshold and max_value < data:
                    max_index = depth

            if max_index != -1:
                object = {
                    "x": col,
                    "y": row,
                    "name": object_name_dict[max_index],
                    "orientation": 0
                }
                room_json["room"].append(object)

    return room_json
