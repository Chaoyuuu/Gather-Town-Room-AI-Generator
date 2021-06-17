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


@app.route('/generator/vae', methods=['GET'])
def get_vae():
    model, z_dim, threshold = load_vae_model()
    z = torch.randn(1, z_dim)
    sample = model.decoder(z)
    output = tensor_to_json(sample, threshold)
    output['time'] = time.time()
    return jsonify(output)


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=8888, help='port')
    arg_parser.add_argument('-d', '--debug', default=8888, help='debug')
    options = arg_parser.parse_args()

    app.run(host='0.0.0.0', port=options.port, debug=options.debug)
