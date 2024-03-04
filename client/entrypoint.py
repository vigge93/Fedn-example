from dataclasses import dataclass
import json
import os

import fire
from fedn.utils.helpers.helpers import get_helper, save_metadata, save_metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import keras

@dataclass
class Metrics:
    Time: float
    Accuracy: float
    F_score: float
    ROC_AUC: float | None
    Precision: float
    Recall: float
    TP: int
    TN: int
    FP: int
    FN: int

    def to_dict(self, prefix="") -> dict[str, float | int | None]:
        return {
            f"{prefix}ROC_AUC": self.ROC_AUC,
            f"{prefix}Accuracy": self.Accuracy,
            f"{prefix}Time": self.Time,
            f"{prefix}Precision": self.Precision,
            f"{prefix}Recall": self.Recall,
            f"{prefix}F_score": self.F_score,
            f"{prefix}TN": int(self.TN),
            f"{prefix}TP": int(self.TP),
            f"{prefix}FN": int(self.FN),
            f"{prefix}FP": int(self.FP),
        }

def save_metrics(model: str, dataset: str, metrics: Metrics):
    with open(os.path.join("metrics", f"{dataset}_{model}.json"), "w") as fd:
        json.dump(metrics.to_dict(), fd)

helper = get_helper("numpyhelper")

clients = os.getenv("FEDN_CLIENTS")
if clients:
    clients = int(clients)
else:
    clients = 1
client_id = os.getenv("FEDN_CLIENT_ID")
if client_id:
    client_id = int(client_id)
else:
    client_id = 1


def load_data(partition, *, train_only=False, test_only=False):
    if train_only and test_only:
        raise ValueError("train_only and test_only can't both be True")
    X, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    n = len(x_train)
    client_n = (n)//clients
    start = client_n*partition
    x_train = x_train[start:start+client_n]
    y_train = y_train[start:start+client_n]
    if train_only:
        return x_train, y_train
    if test_only:
        return x_test, y_test
    return x_train, y_train, x_test, y_test


def compile_model():
    input_shape = (4,)

    # Define model
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model


def init_seed(out_path="seed.npz"):
    model = compile_model()
    weights = model.get_weights()
    helper.save(weights, out_path)


def train(in_model_path, out_model_path):
    model = compile_model()
    weights = helper.load(in_model_path)
    model.set_weights(weights)

    x_train, y_train = load_data(client_id, train_only=True)

    model.fit(x_train, y_train, epochs=2)

    metadata = {
        # num_examples are mandatory
        "num_examples": len(x_train),
        'epochs': 2,
    }

    # Save JSON metadata file (mandatory)
    save_metadata(metadata, out_model_path)

    # Save JSON metadata file (mandatory)
    weights = model.get_weights()
    helper.save(weights, out_model_path)


def validate(in_model_path, out_json_path):
    x_train, y_train, x_test, y_test = load_data(client_id)

    model = compile_model()
    weights = helper.load(in_model_path)
    model.set_weights(weights)

    y_pred = model.predict(x_test)
    roc_auc = roc_auc_score(y_test, y_pred[:, 1])
    accuracy = accuracy_score(y_test, y_pred.argmax(axis=1))
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_test, y_pred.argmax(axis=1), average="binary"
    )
    CM = confusion_matrix(y_test, y_pred.argmax(axis=1))

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    test_metrics = Metrics(
        0,
        accuracy,
        fscore,
        roc_auc,
        precision,
        recall,
        TP,
        TN,
        FP,
        FN,
    ).to_dict("test_")

    y_pred = model.predict_proba(x_train)
    roc_auc = roc_auc_score(y_train, y_pred[:, 1])
    accuracy = accuracy_score(y_train, y_pred.argmax(axis=1))
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_train, y_pred.argmax(axis=1), average="binary"
    )
    CM = confusion_matrix(y_train, y_pred.argmax(axis=1))

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    train_metrics = Metrics(
        0,
        accuracy,
        fscore,
        roc_auc,
        precision,
        recall,
        TP,
        TN,
        FP,
        FN,
    ).to_dict("train_")

    metrics = test_metrics | train_metrics
    save_metrics(metrics, out_json_path)


def infer(in_model_path, out_json_path):
    x_test, _ = load_data(0, test_only=True)

    model = compile_model()
    weights = helper.load(in_model_path)
    model.set_weights(weights)

    y_pred = model.predict(x_test)

    # Save JSON
    with open(out_json_path, "w") as fh:
        fh.write(json.dumps({"predictions": y_pred.tolist()}))


if __name__ == "__main__":
    fire.Fire(
        {
            "init_seed": init_seed,
            "train": train,
            "validate": validate,
            "infer": infer,
        }
    )
