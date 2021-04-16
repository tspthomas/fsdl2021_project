import os
import time
import json
import torch
import shutil
import logging
import mlflow.pyfunc

from PIL import Image

from flask import g
from flask import request
from flask import Blueprint
from flask import send_from_directory
from flask import current_app as app

from flask_api import status

from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

from utils import create_feedback_folder
from constants import INTELSCENES_FOLDER


#TODO improve this, move to common library
from api.v1.fe import transform, int2cat
from api.v1.fe import resnet50_feature_extractor

api_blueprint = Blueprint('/api/v1', __name__)

#TODO improve this, preload
resnet_model = resnet50_feature_extractor(pretrained=True)

model_name = 'intel_scenes_train_resnet50_lr'
stage = 'Production'

lr = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{stage}"
)


@api_blueprint.route('/intelscenes/', methods=['POST'])
def intelscenes():

    logging.info(request.files)

    try:       
        image_query = request.files['upload_image']
        filename = __save_to_disk(image_query)

        # Do feature extraction
        img = Image.open(image_query.stream).convert('RGB')
        x = transform(img)
        x = torch.unsqueeze(x, dim=0)
        
        features = resnet_model(x)
        logging.info(features.shape)
    
        # classify
        prediction = lr.predict(features.detach().numpy())
        prediction_class = int2cat[int(prediction)]
        logging.info(prediction_class)

        # build results
        results = {
            'prediction': prediction_class,
            'classes': [*int2cat.values()],
            'uploaded_image_id': filename
        }
        return json.dumps(results), status.HTTP_200_OK
    except Exception as e:
        logging.info(str(e))
        return json.dumps({'error_message': str(e)}), status.HTTP_500_INTERNAL_SERVER_ERROR


@api_blueprint.route('/feedback/', methods=['POST'])
def feedback():

    logging.info(request.form)

    try:
        feedback_class = request.form['feedback_class']
        image_id = request.form['uploaded_image_id']

        #TODO - make it generic
        created_dir = create_feedback_folder(INTELSCENES_FOLDER, feedback_class)

        # move image to feedback folder
        src = os.path.join(app.config['UPLOAD_FOLDER'], image_id)
        dest = os.path.join(created_dir, image_id)
        shutil.move(src, dest)

        return json.dumps({'image_folder': dest}), status.HTTP_200_OK
    except Exception as e:
        logging.info(str(e))
        return json.dumps({'error_message': str(e)}), status.HTTP_500_INTERNAL_SERVER_ERROR


def __save_to_disk(image_file):
    filename = secure_filename(image_file.filename)
    
    filename = '{}'.format(time.time()) + filename
    complete_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    FileStorage(stream=image_file).save(complete_filename)
    return filename