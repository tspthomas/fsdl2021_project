from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


models_config = {
    'train_test_lr_A': {
        'model': LogisticRegression,
        'hparams': {
            'penalty': 'l2',
            'C': 1.0,
            'random_state': 33
        }
    },
    'train_test_lr_B': {
        'model': LogisticRegression,
        'hparams': {
            'penalty': 'l2',
            'C': 2.0,
            'random_state': 33
        }
    },
    'train_test_lr_C': {
        'model': LogisticRegression,
        'hparams': {
            'penalty': 'l1',
            'C': 1.0,
            'solver': 'liblinear',
            'random_state': 33
        }
    },
    'train_test_svm': {
        'model': svm.SVC,
        'hparams': {
            'kernel': 'rbf',
            'C': 1.0,
            'random_state': 33
        }
    },
    'train_test_knn_A': {
        'model': KNeighborsClassifier,
        'hparams': {
            'n_neighbors': 7
        }
    },
    'train_test_knn_B': {
        'model': KNeighborsClassifier,
        'hparams': {
            'n_neighbors': 3
        }
    }
}