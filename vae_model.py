import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from object_dictionary import *
from utils import *


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


def load_vae_model(vae_model_path):
    loaded = torch.load(vae_model_path, map_location=torch.device('cpu'))
    model = loaded['model']
    z_dim = loaded['z_dim']
    threshold = loaded['threshold']
    return model, z_dim, threshold


def tensor_to_json(sample, threshold):
    output_tensor = torch.zeros((17, 13, 10))

    for k, batch in enumerate(sample):
        i = 0
        for depth in range(17):
            for col in range(13):
                for row in range(10):
                    data = batch[i]
                    i += 1
                    if depth == 15 or depth == 16:
                        output_tensor[depth][col][row] = data
                    elif data >= threshold:
                        output_tensor[depth][col][row] = data

    return encode_to_json(output_tensor, threshold)


def tensor_to_json_special_size(sample, threshold, size):
    output_tensor = torch.zeros((size, 13, 10))

    for k, batch in enumerate(sample):
        i = 0
        for depth in range(size):
            for col in range(13):
                for row in range(10):
                    data = batch[i]
                    i += 1
                    if depth == size - 2 or depth == size - 1:
                        output_tensor[depth][col][row] = data
                    elif data >= threshold:
                        output_tensor[depth][col][row] = data

    return output_tensor


def encode_to_json(output_tensor, threshold):
    room_json = {
        "generator": "vae",
        "room": []
    }

    for row in range(13):
        for col in range(10):
            max_value = 0
            max_index = -1
            for depth in range(15):
                data = output_tensor[depth][row][col]

                if data >= threshold and max_value < data:
                    max_index = depth

            if max_index != -1 and ignore_chair(max_index):
                object_name = object_name_dict[max_index]
                orientation = decode_orientation(max_index, output_tensor[15][row][col], output_tensor[16][row][col])
                object = copy.deepcopy(object_dict[object_name][orientation])
                object["x"] = shift_object(max_index, col, orientation)
                object["y"] = row
                room_json["room"].append(object)

    return room_json


def encode_to_json_special_size(output_tensor, threshold, size):
    room_json = {
        "generator": "vae",
        "room": []
    }

    special_object_name_dict = object_name_dict_room
    if size == 11:
        special_object_name_dict = object_name_dict_room
    elif size == 8:
        special_object_name_dict = object_name_dict_table
    else:
        print("error")
        return {}


    for row in range(13):
        for col in range(10):
            max_value = 0
            max_index = -1
            for depth in range(size - 2):
                data = output_tensor[depth][row][col]

                if data >= threshold and max_value < data:
                    max_index = depth

            if max_index != -1:
                object_name = special_object_name_dict[max_index]
                orientation = decode_orientation(max_index, output_tensor[size - 2][row][col],
                                                 output_tensor[size - 1][row][col])
                object = copy.deepcopy(object_dict[object_name][orientation])
                object["x"] = shift_object(max_index, col, orientation)
                object["y"] = row
                room_json["room"].append(object)

    return room_json
