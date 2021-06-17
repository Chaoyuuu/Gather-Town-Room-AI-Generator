import time
from argparse import ArgumentParser
from flask import Flask, jsonify
from vae_model import *

app = Flask(__name__)


@app.route('/generator/gan', methods=['GET'])
def get_gan():
    return 'GET GAN'


@app.route('/generator/ga', methods=['GET'])
def get_ga():
    return 'GET GA'


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
    model_path = './VAE/model/personal/vae-table'
    return generate_room_from_vae(model_path)


@app.route('/generator/vae/room', methods=['GET'])
def get_vae_room():
    model_path = './VAE/model/vae-room'
    return generate_room_from_vae(model_path)


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
