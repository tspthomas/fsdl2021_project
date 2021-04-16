import os
from constants import FEEDBACK_PATH


def create_feedback_folder(dataset, class_name):
    dest = os.path.join(FEEDBACK_PATH, dataset, class_name)
    os.makedirs(dest, exist_ok=True)
    return dest
