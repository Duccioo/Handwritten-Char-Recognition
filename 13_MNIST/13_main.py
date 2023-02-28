import numpy as np
import os

import itertools
from rich import print
from rich.progress import track
import time

from data_manager import (
    load_data,
    load_labels,
)
from model import relu, softmax, sigmoid
import train_test

from utils import save_results_to_csv, save_model, save_figure, generate_unique_id, init_folder


def main():
    np.random.seed(42)
    input_size = 784

    # Definiamo i parametri da testare
    learning_rates = (0.1, 0.2)
    num_epochs = (10,)
    hidden_sizes = (100, 50)
    loss_functions = ("cross-entropy", "quadratic")
    data_encodings = ("one_hot", "binary")
    batch_sizes = (1)
    weights_inits = ("Xavier", "random", 0)
    hidden_activ_funcs = (sigmoid, relu)
    last_activ_funcs = (sigmoid,)

    # Load the data
    X_train = load_data("data/train-images-idx3-ubyte.gz")
    y_train = load_labels("data/train-labels-idx1-ubyte.gz")
    X_test = load_data("data/t10k-images-idx3-ubyte.gz")
    y_test = load_labels("data/t10k-labels-idx1-ubyte.gz")

    # Generiamo tutte le possibili combinazioni di iperparametri
    hyperparameters = list(
        itertools.product(
            batch_sizes,
            learning_rates,
            loss_functions,
            data_encodings,
            weights_inits,
            last_activ_funcs,
            hidden_activ_funcs,
            num_epochs,
            hidden_sizes,
        )
    )

    best_loss = 10000
    best_accuracy = 0
    best_parameters = [0]
    log_data = []
    n = 0

    # get the parametes with string:

    # initialize folder and load the hyperparam saved
    (full_path, accuracy_path, loss_path, loaded_lines) = init_folder("out", hyperparameters)

    # Iteriamo su tutte le combinazioni di iperparametri
    for hyperparameter in track((hyperparameters), description="training..."):
        # if loaded skip hyperparameters
        if n < loaded_lines and loaded_lines != 0:
            (
                batch_size,
                learning_rate,
                loss_function,
                data_encoding,
                weights_init,
                last_activ_func,
                hidden_activ_func,
                num_epoch,
                hidden_size,
            ) = hyperparameter

            n += 1
            continue

        (
            batch_size,
            learning_rate,
            loss_function,
            data_encoding,
            weights_init,
            last_activ_func,
            hidden_activ_func,
            num_epoch,
            hidden_size,
        ) = hyperparameter

        print(
            f"[bold magenta]Testing hyperparameter[/bold magenta] :{[str(p) if not callable(p) else p.__name__ for p in hyperparameter]}",
        )

        # train the model
        start_time = time.time()
        model, array_accuracy, array_loss = train_test.train(
            X_train,
            y_train,
            input_size,
            hidden_size,
            learning_rate,
            num_epoch,
            batch_size,
            loss_function,
            encode_type=data_encoding,
            weights_init=weights_init,
            last_activ_func=last_activ_func,
            hidden_activ_func=hidden_activ_func,
        )
        end_time = time.time()
        execution_time = end_time - start_time

        # test the model
        (test_accuracy, _, _, _, _) = train_test.test(model, X_test, y_test, loss_function, data_encoding)

        # logging the information of the model
        log_data = [
            n,
            num_epoch,
            hidden_size,
            learning_rate,
            loss_function,
            data_encoding,
            batch_size,
            weights_init,
            hidden_activ_func,
            last_activ_func,
            test_accuracy,
            execution_time,
        ]
        save_results_to_csv(log_data, os.path.join(full_path, f"result_{generate_unique_id(hyperparameters)}.csv"))

        # Saving accuracy andament
        save_figure(
            array_accuracy["epoch"],
            array_accuracy["value"],
            f"{n}_figure.jpg",
            accuracy_path,
        )

        save_figure(
            array_loss["epoch"],
            array_loss["value"],
            f"{n}_figure.jpg",
            loss_path,
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_parameters = hyperparameter

            # salvo il modello se è il migliore
            save_model(model.get_weights(), "best_model")

        n += 1

    print("Best Accuracy: ", best_accuracy)
    print("With Parameters:", best_parameters)


if __name__ == "__main__":
    main()
