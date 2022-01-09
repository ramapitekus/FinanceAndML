from classification.nn_solver_classification import (
    train_classification,
    evaluate_classification,
)
from utils import prepare_dataset, create_loaders, pick_nn_model
from regression.nn_solver_regression import train_regression, evaluate_regression


INTERVALS = {
    1: "01-04-2013-19-07-2016",
    2: "01-04-2013-01-01-2017",
    3: "01-04-2013-31-12-2020",
}
PERIODS = [1, 7, 30, 90]
EPOCHS = 1000
# LSTM LR 0.01
LR = 0.01
DROPOUT = 0.3


def main(
    prediction_type: str,
    interval: int,
    nn_type: str,
    epochs: int,
    lr: float,
    shuffle: bool,
    eval_only: bool = False,
    dropout: float = 0.0,
    loss: str = None,
) -> None:
    """
    For every period in global variable periods list, creates datasets and loaders for training,
    testing and validation. NN Model is initialized, depending on the model specified in argument.
    By setting eval_only to true, model will not be trained, instead pre-trained model will be evaluated.
    Please keep in mind that if shuffle=True, different results may appear, since the train/test dataset
    is randomly shuffled. Dropout is considered only for nn_type dropout. For regression For binary classification only
    binary cross entropy - thus changing loss does not have an effect.
    Parameters:
            prediction_type: regression or classification
            interval: time period to train on (1, 2 or 3 in our setting)
            nn_type: Neural network type to use (possibilites: "ANN", "dropout", "LSTM")
            epochs: number of epochs
            lr: learning rate
            shuffle: True for shuffling the data when splitting, False for not shuffling
            eval_only: Use pre-trained model to evalute it and perform statistics
            dropout: if nn_type is "dropout", specify the probability of dropout. Do not specify for other NN architectures
            loss: in regression use either logcosh or MSE.
                  For binary classification, only binary cross entropy (BCELoss) is implemented, thus do not specify empty in this case.
    """

    for period in PERIODS:
        interval_str = INTERVALS[interval]
        train_dataset, test_dataset, val_dataset = prepare_dataset(
            prediction_type, interval_str, period, nn_type, shuffle
        )

        train_loader, val_loader, test_loader = create_loaders(
            train_dataset, val_dataset, test_dataset, batch_size=64
        )

        n_features = train_dataset.x.shape[1]

        net = pick_nn_model(prediction_type, nn_type, n_features, interval, dropout)

        if not eval_only:
            print(
                f"{prediction_type} {nn_type} network, interval {interval} for {period} day(s) period. Learning Rate is {lr}. {epochs} epochs"
            )
        model_path = f"./predictions/neural_nets/models/{prediction_type}/{nn_type}/interval_{interval}/model_interval_{interval}_period_{period}.pth"

        if prediction_type == "regression":
            if not eval_only:
                train_regression(
                    net,
                    train_loader,
                    val_loader,
                    model_path,
                    epochs=epochs,
                    loss_type=loss,
                    lr=lr,
                )

            evaluate_regression(
                n_features,
                test_loader,
                interval,
                period,
                model_path,
                nn_type,
                hidden_units=n_features,
                dropout=dropout,
            )

        elif prediction_type == "classification":
            if not eval_only:
                train_classification(
                    net,
                    train_loader,
                    val_loader,
                    model_path,
                    epochs=epochs,
                    loss_type=loss,
                    lr=lr,
                )

            evaluate_classification(
                n_features,
                test_loader,
                interval,
                period,
                model_path,
                nn_type,
                hidden_units=n_features,
                dropout=dropout,
            )


# Currently, set up for evaluation of regression NN models
for nn_type in ["dropout", "ANN", "LSTM"]:
    for interval in INTERVALS.keys():
        main(
            prediction_type="regression",
            interval=interval,
            nn_type=nn_type,
            epochs=EPOCHS,
            lr=LR,
            shuffle=False,
            eval_only=True,
            dropout=DROPOUT,
            loss="mse",
        )
