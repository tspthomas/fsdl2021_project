import os
import numpy as np
import argparse

import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_TRAIN = os.environ.get("X_TRAIN")
Y_TRAIN = os.environ.get("Y_TRAIN")
X_TEST = os.environ.get("X_TEST")
Y_TEST = os.environ.get('Y_TEST')

MODEL_NAME = os.environ.get("MODEL_NAME")
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--X_train", type=str, default=X_TRAIN)
    parser.add_argument("--y_train", type=str, default=Y_TRAIN)
    parser.add_argument("--X_test", type=str, default=X_TEST)
    parser.add_argument("--y_test", type=str, default=Y_TEST)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--save_dir", type=str, default=CURRENT_DIR)
    return parser

def main():

    parser = _setup_parser()
    args = parser.parse_args()

    # load training data from local dir
    X_train = np.load(args.X_train)
    y_train = np.load(args.y_train)
    X_test = np.load(args.X_test)
    y_test = np.load(args.y_test)

    # fit model
    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)

    # check prediction report
    predictions = logmodel.predict(X_test)
    print(classification_report(y_test, predictions))

    # save model
    with open(CURRENT_DIR+'/'+args.model_name, 'wb') as f:
        pickle.dump(logmodel, f)


if __name__ == "__main__":
    main()