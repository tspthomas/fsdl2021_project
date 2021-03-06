import os
import logging

from flask import Flask
from flask import render_template

from api.v1 import routes as v1

from pprint import pprint
from mlflow.tracking import MlflowClient

from constants import UPLOAD_FOLDER


log = logging.getLogger('werkzeug')
log.setLevel(logging.DEBUG)

logging.basicConfig(
    level='INFO',
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s')


app = Flask(__name__)
app.config['SECRET_KEY'] = 'happysecreoffsdl'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.debug = True
app.register_blueprint(v1.api_blueprint, url_prefix='/api/v1')


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/intelscenes/")
def intelscene():
    return render_template('models/intelscenes.html')


def run_main():
    port = os.environ['FLASK_PORT']
    app.run(host='0.0.0.0', port=port)


if __name__ == '__main__':
    run_main()
