import os
import json
import torch
import urllib
import logging

from PIL import Image

from flask import g
from flask import jsonify
from flask import request
from flask import Blueprint
from flask import send_from_directory

from flask_api import status


#TODO improve this, move to common library
from api.v1.fe import transform
from api.v1.fe import resnet50_feature_extractor

api_blueprint = Blueprint('/api/v1', __name__)

#TODO improve this, preload
resnet_model = resnet50_feature_extractor(pretrained=True)


@api_blueprint.route('/intelscenes/', methods=['POST'])
def intelscenes():

    logging.info(request.files)

    try:       
        image_query = request.files['upload_image']
        img = Image.open(image_query.stream).convert('RGB')

        # Do feature extraction
        x = transform(img)
        x = torch.unsqueeze(x, dim=0)
        
        features = resnet_model(x)
        logging.info(features.shape)
    
        
        # classify

        # build results

        return json.dumps({'features': str(features)}), status.HTTP_200_OK
    except Exception as e:
        logging.info(str(e))
        return json.dumps({'error_message': str(e)}), status.HTTP_500_INTERNAL_SERVER_ERROR


@api_blueprint.route('/image/<folder_id>/<path:filename>')
def get_image(folder_id, filename):
    src_path = os.path.join(DOCUMENTS_PATH, folder_id, 'images')
    return send_from_directory(src_path, filename)


@api_blueprint.route('/document/<path:filepath>')
def get_document(filepath):
    folder = os.path.join('/', os.path.dirname(filepath))
    filename = os.path.basename(filepath)
    filename = urllib.parse.unquote(urllib.parse.unquote(filename))
    print(folder, filename)
    return send_from_directory(folder, filename)