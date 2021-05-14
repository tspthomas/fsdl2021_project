from sklearn.linear_model import LogisticRegression, SGDClassifier

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
    }
}