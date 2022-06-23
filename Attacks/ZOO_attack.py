import pickle
import numpy as np
import matplotlib.pyplot as plt
from art.attacks.evasion import ZooAttack
from art.estimators.classification import XGBoostClassifier

model_endpoint = "../Model/model_xgb.pkl"
X_train_path = "../Data/X_train.npy"
y_train_path = "../Data/y_train.npy"
X_test_path = "../Data/X_test.npy"
y_test_path = "../Data/y_test.npy"


def load():
    with open(model_endpoint, "rb") as infile:
        model = pickle.load(infile)
    X_train = np.load(X_train_path, allow_pickle=True)
    y_train = np.load(y_train_path, allow_pickle=True)
    X_test = np.load(X_test_path, allow_pickle=True)
    y_test = np.load(y_test_path, allow_pickle=True)
    return model, X_train, y_train, X_test, y_test


def attack(model, X_test, y_test):
    # Step 3: Create the ART classifier with the trained model.
    classifier = XGBoostClassifier(model=model,
                                   clip_values=(0, 1),
                                   nb_features=X_test.shape[1],
                                   nb_classes=2)
# Step 4: Initialize the zoo attack object with ART classifier.
    attack = ZooAttack(
        classifier=classifier,
        confidence=0.0,
        targeted=False,
        learning_rate=1e-1,
        max_iter=200,
        binary_search_steps=10,
        initial_const=1e-3,
        abort_early=True,
        use_resize=False,
        use_importance=False,
        nb_parallel=5,
        batch_size=1,
        variable_h=0.01,
    )
# Step 5: Generate the adversarial data with test data.
    x_test_adv = attack.generate(x=X_test, y=y_test)
# Step 6: Compute the score.
    score = model.score(X_test, y_test)
    print("Benign Training Score: %.4f" % score)
    attack_score = model.score(x_test_adv, y_test)
    print("Adversarial Training Score: %.4f" % attack_score)


def main():
    model, X_train, y_train, X_test, y_test = load()
    attack(model, X_test, y_test)


if __name__ == "__main__":
    main()
