import pickle
import numpy as np
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ProjectedGradientDescent
import matplotlib.pyplot as plt

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
    art_classifier = SklearnClassifier(model=model)
# Step 4: Define the epsilon values in a list which defines the attack strength.
    eps_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1,
                0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    score_list = list()

    for eps in eps_list:
        # Step 5: Initialize the projected gradient descent object with ART classifier.
        pgd = ProjectedGradientDescent(estimator=art_classifier,
                                       norm=np.inf, eps=eps,
                                       eps_step=0.05,
                                       max_iter=20,
                                       targeted=False,
                                       num_random_init=0,
                                       batch_size=128)
# Step 6: Generate the adversial data with test data.
        x_test_adv = pgd.generate(X_test)
# Step 7: Compute the score.
        score = model.score(x_test_adv, y_test)
        score_list.append(score)
# Step 8: Plot the result
    plt.plot(eps_list, score_list)
    plt.xlabel('attack strength(eps)')
    plt.ylabel('Test Accuracy')
    plt.ylim((0, 1))
    plt.savefig('../Result/PGD_attack_result.png')
    plt.show()


def main():
    model, X_test, y_test = load()
    attack(model, X_test, y_test)


if __name__ == "__main__":
    main()
