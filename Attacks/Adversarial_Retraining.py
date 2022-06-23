import pickle
from turtle import color
import numpy as np
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ProjectedGradientDescent
import matplotlib.pyplot as plt
from copy import copy

model_endpoint = "../Model/model_lr.pkl"
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


def Adversarial_Retraining(model, X_train, y_train):
    eps_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1,
                0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    robust_model = copy(model)
    art_classifier = SklearnClassifier(model=robust_model)
    for eps in eps_list:
        pgd = ProjectedGradientDescent(estimator=art_classifier, norm=np.inf, eps=eps,
                                       eps_step=0.05, max_iter=20, targeted=False, num_random_init=0, batch_size=128)
        x_train_adv = pgd.generate(X_train)
        robust_model.fit(x_train_adv, y_train)
    return robust_model


def Test_Robustness(model, X_test, y_test):
    eps_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1,
                0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    score_list = list([])
    art_classifier = SklearnClassifier(model=model)

    for eps in eps_list:
        pgd = ProjectedGradientDescent(estimator=art_classifier, norm=np.inf, eps=eps,
                                       eps_step=0.05, max_iter=20, targeted=False, num_random_init=0, batch_size=128)
        x_test_adv = pgd.generate(X_test)
        score = model.score(x_test_adv, y_test)
        score_list.append(score)

    return eps_list, score_list


def show_figure(eps_list, score_list, robust_score_list):
    plt.plot(eps_list, score_list, color='red')
    plt.plot(eps_list, robust_score_list, color='green')
    plt.xlabel('attack strength(eps)')
    plt.ylabel('Test Accuracy')
    legend_drawn_flag = True
    plt.legend(["Original Model", "Robust Model"],
               loc=0, frameon=legend_drawn_flag)
    plt.ylim((0, 1))
    plt.savefig('../Result/Test_Robustness_Result.png')
    plt.show()


def main():
    model, X_train, y_train, X_test, y_test = load()
    eps_list, score_list = Test_Robustness(model, X_test, y_test)
    robust_model = copy(Adversarial_Retraining(model, X_train, y_train))
    eps_list_robust, score_list_robust = Test_Robustness(
        robust_model, X_test, y_test)
    show_figure(eps_list, score_list, score_list_robust)


if __name__ == "__main__":
    main()
