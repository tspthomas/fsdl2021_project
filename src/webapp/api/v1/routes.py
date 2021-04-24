import os
import time
import json
import torch
import shutil
import logging
import mlflow

from PIL import Image

from flask import g
from flask import request
from flask import Blueprint
from flask import send_from_directory
from flask import current_app as app

from flask_api import status

from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

from constants import FEEDBACK_FOLDER
from constants import INTELSCENES_FOLDER
from constants import INTELSCENES_RESNET50_LR_MODEL

from fsdl_lib import feature_extraction as fe


api_blueprint = Blueprint('/api/v1', __name__)


def get_resnet50():
    if 'resnet50' not in g:
        g.resnet50 = fe.resnet50_feature_extractor(pretrained=True)

    return g.resnet50


def get_lr_intelscenes():
    stage = 'Production'

    if 'lr_intelscenes' not in g:
        g.lr_intelscenes = mlflow.pyfunc.load_model(
            model_uri=f"models:/{INTELSCENES_RESNET50_LR_MODEL}/{stage}"
        )

    return g.lr_intelscenes


@api_blueprint.route('/intelscenes/', methods=['POST'])
def intelscenes():

    logging.info(request.files)

    try:
        image_query = request.files['upload_image']
        filename = __save_to_disk(image_query)

        # Do feature extraction
        img = Image.open(image_query.stream).convert('RGB')
        transform = fe.get_transform()
        x = transform(img)
        x = torch.unsqueeze(x, dim=0)

        resnet50_model = get_resnet50()
        features = resnet50_model(x)
        logging.info(features.shape)

        # classify
        lr = get_lr_intelscenes()
        prediction = lr.predict(features.detach().numpy())
        prediction_class = fe.int2cat[int(prediction)]
        logging.info(prediction_class)

        # build results
        results = {
            'prediction': prediction_class,
            'classes': [*fe.int2cat.values()],
            'uploaded_image_id': filename
        }
        return json.dumps(results), status.HTTP_200_OK
    except Exception as e:
        logging.info(str(e))
        return json.dumps({'error_message': str(e)}
                          ), status.HTTP_500_INTERNAL_SERVER_ERROR


@api_blueprint.route('/feedback/', methods=['POST'])
def feedback():

    logging.info(request.form)

    try:
        feedback_class = request.form['feedback_class']
        image_id = request.form['uploaded_image_id']

        # TODO - make it generic
        created_dir = __create_feedback_folder(
            INTELSCENES_FOLDER, feedback_class)

        # move image to feedback folder
        src = os.path.join(app.config['UPLOAD_FOLDER'], image_id)
        dest = os.path.join(created_dir, image_id)
        shutil.move(src, dest)

        return json.dumps({'image_folder': dest}), status.HTTP_200_OK
    except Exception as e:
        logging.info(str(e))
        return json.dumps({'error_message': str(e)}
                          ), status.HTTP_500_INTERNAL_SERVER_ERROR


def __save_to_disk(image_file):
    filename = secure_filename(image_file.filename)

    filename = '{}'.format(time.time()) + filename
    complete_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    FileStorage(stream=image_file).save(complete_filename)
    return filename


def __create_feedback_folder(dataset, class_name):
    dest = os.path.join(FEEDBACK_FOLDER, dataset, class_name)
    os.makedirs(dest, exist_ok=True)
    return dest
