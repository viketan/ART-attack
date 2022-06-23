import pickle
import numpy as np
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import HopSkipJump
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

model_endpoint = "../Model/model_lr.pkl"
X_train_path = "../Data/X_train.npy"
y_train_path = "../Data/y_train.npy"
X_test_path = "../Data/X_test.npy"
y_test_path = "../Data/y_test.npy"


def load():
    with open(model_endpoint, "rb") as infile:
        model = pickle.load(infile)
    X_test = np.load(X_test_path, allow_pickle=True)
    y_test = np.load(y_test_path, allow_pickle=True)
    return model, X_test, y_test


def attack(model, X_test, y_test):
    # Step 3: Create the ART classifier with the trained model.
    score_b = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    print("Accuracy before attack: ", score_b)
    print(classification_report(y_test, y_pred))
    art_classifier = SklearnClassifier(model=model)
    # Step 5: Initialize the projected gradient descent object with ART classifier.
    attack = HopSkipJump(classifier=art_classifier,
                         targeted=False, max_iter=0, max_eval=1000, init_eval=10)
# Step 6: Generate the adversial data with test data.
    x_test_adv = attack.generate(X_test)
# Step 7: Compute the score.
    score = model.score(x_test_adv, y_test)
    y_pred = model.predict(x_test_adv)
    print(classification_report(y_test, y_pred))
    art_classifier = SklearnClassifier(model=model)
    print("Accuracy after attack: ", score)
    np.savetxt("x_test_adv.csv", x_test_adv, delimiter=",")


def main():
    model, X_test, y_test = load()
    attack(model, X_test, y_test)


if __name__ == "__main__":
    main()
