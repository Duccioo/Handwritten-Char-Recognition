import numpy as np
from rich import print

from data_manager import (
    data_normalization,
    binary_encoding,
    one_hot_encoding,
)
from model import NN_MNIST, relu, softmax, sigmoid


def LOSS(Y_pred, Y_true, type="cross-entropy"):
    if type == "cross-entropy" or type == "cross-entropy-1" or type == "cross-entropy-0":
        epsilon = 1e-8
        num_examples = Y_true.shape[0]
        Y_pred = np.maximum(Y_pred, epsilon)
        log_probs = -np.log(Y_pred[range(num_examples), np.argmax(Y_true, axis=1)])
        loss = np.sum(log_probs) / num_examples
        return loss

    elif type == "quadratic":
        return 0.5 * np.sum(np.square(Y_true - Y_pred))
    else:
        print("Loss function not implemented")


def train(
    X_train,
    y_train,
    input_size,
    hidden_size,
    learning_rate=0.02,
    num_epochs=10,
    batch_size=32,
    loss_type="cross-entropy",
    encode_type=one_hot_encoding,
    last_activ_func=softmax,
    weights_init=-1,
    normalization=True,
    hidden_activ_func=relu,
    verbose=0,
):
    if encode_type == binary_encoding:
        output_size = 8
        y_train = binary_encoding(y_train)

    else:
        output_size = 10
        y_train = one_hot_encoding(y_train)

    if normalization == True:
        X_train, _ = data_normalization(train=X_train)

    model = NN_MNIST(
        input_size,
        hidden_size,
        output_size,
        loss_type=loss_type,
        last_func_activation=last_activ_func,
        hidden_activation_func=hidden_activ_func,
        weights_init=weights_init,
    )

    array_loss = {"epoch": [], "value": []}
    array_accuracy = {"epoch": [], "value": []}
    step = 0

    for epoch in range(num_epochs):
        # Shuffle the training data
        permutation = np.random.permutation(X_train.shape[0])
        X_train = X_train[permutation]
        y_train = y_train[permutation]

        # Split the training data into batches
        num_batches = X_train.shape[0] // batch_size
        for batch in range(num_batches):
            # Get a mini-batch of training data
            start = batch * batch_size
            end = (batch + 1) * batch_size
            X = X_train[start:end]
            Y = y_train[start:end]

            # Forward pass
            output_layer = model.forward(X)

            # Compute loss and accuracy
            loss = LOSS(output_layer, Y, type=loss_type)
            accuracy = np.mean(np.argmax(Y, axis=1) == np.argmax(output_layer, axis=1))

            # Backward pass
            model.backward(X, Y)

            # Update weights and biases
            model.update(learning_rate)

            step += 1

        # Print progress

        array_accuracy["epoch"].append(epoch)
        array_accuracy["value"].append(accuracy)

        array_loss["epoch"].append(epoch)
        array_loss["value"].append(loss)

        if epoch % 1 == 0 and verbose == 1:
            print(f"Epoch {epoch+1}/{num_epochs}, loss = {loss:.4f}, accuracy = {accuracy:.4f}")

    return model, array_accuracy, array_loss


def precision_recall_f1(y_true, y_pred):
    # Calcola il numero di true positives (tp), false positives (fp) e false negatives (fn)
    tp = np.sum(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))

    # Calcola la precision, recall e f1-score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def test(model, x_test, y_test, loss_type, data_encoding, normalization=True):
    if data_encoding == binary_encoding:
        y_test = binary_encoding(y_test)

    else:
        y_test = one_hot_encoding(y_test)

    if normalization == True:
        _, x_test = data_normalization(test=x_test)

    # Evaluate the model on the test set
    output_test = model.forward(x_test.reshape(-1, 784))
    test_loss = LOSS(output_test, y_test, type=loss_type)
    test_accuracy = np.mean(np.argmax(y_test, axis=1) == np.argmax(output_test, axis=1))
    precision, recall, f1 = precision_recall_f1(y_test, output_test)
    print(
        f"[bold italic yellow on red blink]Test loss[/] = {test_loss:.4f}, [bold italic blue on green blink]Test accuracy[/] = {test_accuracy:.4f}\n"
    )
    return test_accuracy, precision, recall, f1, test_loss
