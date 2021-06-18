import time
from argparse import ArgumentParser
from flask import Flask, jsonify

from gan_model import *
from vae_model import *
from GA.ga_rooms_output import ga_rooms

app = Flask(__name__)

ga_counter = 0

@app.route('/generator/gan', methods=['GET'])
def get_gan():
    myGen = Generator()
    myGen.load_state_dict(torch.load("./GAN/saveG.pt"))

    noise_fn = lambda x: torch.randn((x, 16))
    latent_vec = noise_fn(1)
    result_tensor = myGen(latent_vec)

    output = tensor_to_json_gan(result_tensor[0])
    output['time'] = time.time()
    return jsonify(output)


@app.route('/generator/ga', methods=['GET'])
def get_ga():
    global ga_counter
    ga_counter = (ga_counter + 1) % 5
    print(ga_counter)
    return jsonify(ga_rooms[ga_counter])


def generate_room_from_vae(model_path):
    model, z_dim, threshold = load_vae_model(model_path)
    z = torch.randn(1, z_dim)
    sample = model.decoder(z)
    output = tensor_to_json(sample, threshold)
    output['time'] = time.time()
    return output


@app.route('/generator/vae', methods=['GET'])
def get_vae():
    model_path = './VAE/model/simple-vae'
    return generate_room_from_vae(model_path)


@app.route('/generator/vae/table', methods=['GET'])
def get_vae_table():
    model_path = 'VAE/model/vae-table'
    model, z_dim, threshold = load_vae_model(model_path)
    z = torch.randn(1, z_dim)
    sample = model.decoder(z)
    output_tensor = tensor_to_json_special_size(sample, threshold, 8)
    output = encode_to_json_special_size(output_tensor, threshold, 8)
    output['time'] = time.time()
    return output


@app.route('/generator/vae/room', methods=['GET'])
def get_vae_room():
    model_path = 'VAE/model/vae-room'
    model, z_dim, threshold = load_vae_model(model_path)
    z = torch.randn(1, z_dim)
    sample = model.decoder(z)
    output_tensor = tensor_to_json_special_size(sample, threshold, 11)
    output = encode_to_json_special_size(output_tensor, threshold, 11)
    output['time'] = time.time()
    return output


@app.route('/generator/vae/chao', methods=['GET'])
def get_vae_chao():
    model_path = './VAE/model/personal/vae-chaoyu'
    return generate_room_from_vae(model_path)


@app.route('/generator/vae/johnny', methods=['GET'])
def get_vae_johnny():
    model_path = './VAE/model/personal/vae-johnny'
    return generate_room_from_vae(model_path)


@app.route('/generator/vae/han', methods=['GET'])
def get_vae_han():
    model_path = './VAE/model/personal/vae-han'
    return generate_room_from_vae(model_path)


@app.route('/generator/vae/ning', methods=['GET'])
def get_vae_ning():
    model_path = './VAE/model/personal/vae-ning'
    return generate_room_from_vae(model_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=8888, help='port')
    arg_parser.add_argument('-d', '--debug', default=8888, help='debug')
    options = arg_parser.parse_args()

    app.run(host='0.0.0.0', port=options.port, debug=options.debug)
