from Model.model import pred
import os
from flask import Flask, request#,render_template,  make_response, jsonify, send_file
import logging
#from flask_cors import CORS, cross_origin

app = Flask(__name__, static_url_path='/', static_folder='./build')
#CORS(app,supports_credentials = True)

@app.route('/', methods=['GET', 'POST'])
#@cross_origin(supports_credentials = True)
def index():
    return app.send_static_file('index.html')

@app.errorhandler(404)
#@cross_origin(supports_credentials = True)
def not_found(e):
    return app.send_static_file('index.html')

@app.route('/predict', methods=['GET', 'POST'])
#@cross_origin(supports_credentials = True)
def home():
    app.logger.info('Action Initiated')
    review = request.json['review']
    app.logger.info('Reviewed')
    prediction = pred(review)
    app.logger.info('Predicted')
    # for i in prediction:
    #     print(i)
    return prediction

if __name__ == '__main__':
    app.run()
