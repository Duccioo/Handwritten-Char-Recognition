import numpy as np
from matplotlib import pyplot as plt
import os
import hashlib
import glob
import csv

from model import relu, softmax, sigmoid


def load_csv(filename, row_name):
    # Apri il file CSV in modalità lettura
    with open(filename, "r") as f:
        # Crea un oggetto reader CSV
        reader = csv.DictReader(f)

        # Inizializza il massimo valore a None
        max_value = None

        # Loop attraverso le righe del file CSV
        for row in reader:
            # Ottieni il valore della colonna di interesse
            col_value = float(row[row_name])

            # Se il massimo valore è ancora None, inizializzalo al valore corrente
            if max_value is None:
                max_value = col_value

            # Altrimenti, se il valore corrente è maggiore del massimo valore, aggiorna il massimo valore
            elif col_value > max_value:
                max_value = col_value

    return max_value


def generate_unique_id(params):
    input_str = ""

    # Concateniamo le stringhe dei dati di input
    for param in params:
        param_1 = [str(p) if not callable(p) else p.__name__ for p in param]
        input_str += str(param_1)

    # Calcoliamo il valore hash SHA-256 della stringa dei dati di input
    hash_obj = hashlib.sha256(input_str.encode())
    hex_dig = hash_obj.hexdigest()

    # Restituiamo i primi 8 caratteri del valore hash come ID univoco
    return hex_dig[:8]


def init_folder(name, params):
    absolute_path = os.path.dirname(__file__)
    relative_path = name

    full_path = os.path.join(absolute_path, relative_path)
    accuracy_path = os.path.join(full_path, "accuracy_figures")
    loss_path = os.path.join(full_path, "loss_figures")

    loaded_lines = 0

    if os.path.exists(full_path):
        # check if there is a csv file in the folder

        if len(glob.glob(os.path.join(full_path, "*.csv"))) >= 1:
            # get the name of the csv file
            full_file_name = glob.glob(os.path.join(full_path, "*.csv"))[0]
            file_name = os.path.basename(full_file_name)
            # get the hashi code
            hashi_name = file_name.split(".")[0].split("_")[-1]

            # check if the hashi code is the same
            if hashi_name == generate_unique_id(params):
                # print("The hashi code is the same")
                print("try to load the csv file...")
                loaded_lines = load_csv(full_file_name, "Number")
                print("loaded lines: ", loaded_lines + 1)

                loaded_lines = loaded_lines + 1

                # get the last number on the csv file

    else:
        os.mkdir(full_path)
        os.mkdir(accuracy_path)
        os.mkdir(loss_path)

    return (full_path, accuracy_path, loss_path, loaded_lines)


def save_model(weights, filename):
    np.savez(filename, W1=weights["W1"], b1=weights["b1"], W2=weights["W2"], b2=weights["b2"])


def save_figure(X, Y, name, path):
    absolute_path = os.path.dirname(__file__)
    full_path = path
    # check if path is a dir
    if os.path.isdir(full_path) is False:
        # make a dir
        os.makedirs(full_path)

    plt.figure(figsize=(10, 8))
    plt.ylim(0.1, 1.05)
    plt.plot(X, Y)
    plt.xlabel("STEP")
    plt.ylabel("ACCURACY")
    plt.savefig(os.path.join(full_path, name))
    plt.close()


def save_results_to_csv(results, filename):
    # check if file exists
    if os.path.exists(filename):
        with open(filename, "a") as f:
            if results[7] == -1:
                results[7] = "random"

            if results[8] == relu:
                results[8] = "ReLu"
            elif results[8] == softmax:
                results[8] = "Softmax"
            elif results[8] == sigmoid:
                results[8] = "Sigmoid"

            if results[9] == relu:
                results[9] = "ReLu"
            elif results[9] == softmax:
                results[9] = "Softmax"
            elif results[9] == sigmoid:
                results[9] = "Sigmoid"

            f.write(
                "{}, {}, {}, {},{},{},{},{},{},{},{},{}\n".format(
                    results[0],
                    results[1],
                    results[2],
                    results[3],
                    results[4],
                    results[5],
                    results[6],
                    results[7],
                    results[8],
                    results[9],
                    results[10],
                    results[11],
                )
            )

    else:
        with open(filename, "w") as f:
            f.write(
                "Number,Epoch, Hidden Neurons, Learning Rate, Loss Function, Data Encoding, Batch Size, Weights Init,Hidden Activation Function, Last Activation Function, Accuracy, Time\n"
            )

            if results[7] == -1:
                results[7] = "random"

            if results[8] == relu:
                results[8] = "ReLu"
            elif results[8] == softmax:
                results[8] = "Softmax"
            elif results[8] == sigmoid:
                results[8] = "Sigmoid"

            if results[9] == relu:
                results[9] = "ReLu"
            elif results[9] == softmax:
                results[9] = "Softmax"
            elif results[9] == sigmoid:
                results[9] = "Sigmoid"

            f.write(
                "{}, {}, {}, {},{},{},{},{},{},{},{},{}\n".format(
                    results[0],
                    results[1],
                    results[2],
                    results[3],
                    results[4],
                    results[5],
                    results[6],
                    results[7],
                    results[8],
                    results[9],
                    results[10],
                    results[11],
                )
            )
