from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import corrector

server = Flask(__name__)
cors = CORS(server)
server.config['CORS_HEADERS'] = 'Content-Type'


@server.route('/', methods=['GET'])
def upload():
    return 'OK'


@server.route('/result', methods=['POST'])
@cross_origin()
def process():
    result = corrector.eval_img(request.files['file'].read())
    return jsonify(result)


if __name__ == '__main__':
    server.run(debug=True)
