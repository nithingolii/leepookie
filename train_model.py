
"""
Train a simple classifier on the collected dataset and save it as gesture_model.pkl
Usage:
  python train_model.py
Requirements:
  pip install numpy scikit-learn joblib

"""
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

DATAFILE = 'dataset.npz'  # created by data_capture.py when pressing 'c'
OUTMODEL = 'gesture_model.pkl'


def load_dataset():
    if not os.path.exists(DATAFILE):
        raise FileNotFoundError(f'{DATAFILE} not found. Run data_capture.py and press c to combine samples')
    data = np.load(DATAFILE, allow_pickle=True)
    X, y = data['X'], data['y']
    labels = data.get('labels', None)
    return X, y, labels


def main():
    X, y, labels = load_dataset()
    print('Loaded', X.shape[0], 'samples, dim', X.shape[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    ypred = clf.predict(X_test)
    print(classification_report(y_test, ypred))

    joblib.dump({'model': clf, 'labels': labels}, OUTMODEL)
    print('Saved model to', OUTMODEL)

if __name__ == '__main__':
    main()