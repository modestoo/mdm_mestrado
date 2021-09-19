import pickle

import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss


def train_lr_model(
    name_model, best_alpha, penalty, train_data, train_labels, test_data, test_labels
):
    sgd_clf = SGDClassifier(
        loss="log", alpha=best_alpha, penalty=penalty, n_jobs=-1, random_state=42
    )

    sgd_clf.fit(train_data, train_labels)
    cal_clf = CalibratedClassifierCV(sgd_clf)
    cal_clf.fit(train_data, train_labels)

    predict_test = cal_clf.predict_proba(test_data)
    predict_metrics_test = cal_clf.predict(test_data)

    model_log_loss = log_loss(test_labels, predict_test)
    model_accuracy_score = accuracy_score(test_labels, predict_metrics_test)
    model_f1_score = f1_score(test_labels, predict_metrics_test)

    filename = "models/" + name_model + ".sav"
    pickle.dump(cal_clf, open(filename, "wb"))

    return model_log_loss, model_accuracy_score, model_f1_score


def train_xgboost_model(
    name_model,
    n_estimators,
    max_depth,
    train_data,
    train_labels,
    test_data,
    test_labels,
):

    gbdt_clf = xgb.XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        objective="binary:logistic",
        n_jobs=-1,
        random_state=42,
    )

    gbdt_clf.fit(train_data, train_labels)
    cal_clf = CalibratedClassifierCV(gbdt_clf)
    cal_clf.fit(train_data, train_labels)

    predict_test = cal_clf.predict_proba(test_data)
    predict_metrics_test = cal_clf.predict(test_data)

    model_log_loss = log_loss(test_labels, predict_test)
    model_accuracy_score = accuracy_score(test_labels, predict_metrics_test)
    model_f1_score = f1_score(test_labels, predict_metrics_test)

    filename = "models/" + name_model + ".sav"
    pickle.dump(cal_clf, open(filename, "wb"))

    return model_log_loss, model_accuracy_score, model_f1_score
