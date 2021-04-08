import os

from flask import Flask
from flask import render_template

app = Flask(__name__)
app.config['SECRET_KEY'] = 'happysecreoffsdl'
app.debug = True
#app.register_blueprint(v1.api_blueprint, url_prefix='/api/v1')


@app.route("/")
def index():
    return render_template('index.html')


def run_main():
    port = os.environ['FLASK_PORT']
    app.run(host='0.0.0.0', port=port)


if __name__ == '__main__':
    run_main()