import os
import pickle


class FeaturesData(object):
    def __init__(self):
        self.X = []
        self.y = []
        self.paths = set()

    def add(self, features, label, path):
        self.X.append(features)
        self.y.append(label)
        self.paths.add(path)

    def path_exists(self, path):
        return path in self.paths

    def get_hash(self):
        return hash(frozenset(self.paths))


def __get_features_path(dest_path, data_folder):
    filename = f'{data_folder}.pkl'
    features_path = os.path.join(dest_path, filename)
    return features_path


def load_or_create_features(dest_path, data_folder):
    features_path = __get_features_path(dest_path, data_folder)

    if os.path.isfile(features_path):
        features_data = load_features(dest_path, data_folder)
    else:
        features_data = FeaturesData()

    return features_data


def save_features(dest_path, data_folder, features_data):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    features_path = __get_features_path(dest_path, data_folder)

    with open(features_path, 'wb') as f:
        pickle.dump(features_data, f)


def load_features(dest_path, data_folder):
    features_path = __get_features_path(dest_path, data_folder)

    with open(features_path, 'rb') as f:
        features_data = pickle.load(f)

    return features_data