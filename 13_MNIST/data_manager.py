import gzip
import struct
import numpy as np
import os


def load_data(path):
    absolute_path = os.path.dirname(__file__)
    relative_path = path
    full_path = os.path.join(absolute_path, relative_path)

    with gzip.open(full_path, "rb") as f:
        data = f.read()
        offset = 0
        magic, size = struct.unpack_from(">II", data, offset)
        offset += struct.calcsize(">II")
        if magic != 2051:
            raise ValueError("Invalid magic number")
        num_images = size
        rows, cols = struct.unpack_from(">II", data, offset)
        offset += struct.calcsize(">II")
        images = np.empty((num_images, rows, cols))
        for i in range(num_images):
            images[i] = np.array(
                struct.unpack_from(">" + str(rows * cols) + "B", data, offset)
            ).reshape((rows, cols))
            offset += struct.calcsize(">" + str(rows * cols) + "B")
        return images


def one_hot_encoding(labels, LowValue=0.0001, HighValue=1):
    # Redefine the values for the Low and High:

    labels_one_hot = np.zeros((labels.size, labels.max() + 1))
    labels_one_hot[np.arange(labels.size), labels] = 1

    labels_one_hot[labels_one_hot == 0] = LowValue
    labels_one_hot[labels_one_hot == 1] = HighValue

    return labels_one_hot


def binary_encoding(labels):
    # labels_binary = np.zeros(len(labels), dtype=np.int8)
    labels_binary = np.array([np.unpackbits(np.uint8(label)) for label in labels])
    return labels_binary


def load_labels(path):
    absolute_path = os.path.dirname(__file__)
    relative_path = path
    full_path = os.path.join(absolute_path, relative_path)

    with gzip.open(full_path, "rb") as f:
        data = f.read()
        offset = 0
        magic, size = struct.unpack_from(">II", data, offset)
        offset += struct.calcsize(">II")
        if magic != 2049:
            raise ValueError("Invalid magic number")
        labels = np.empty(size, dtype=np.int64)
        for i in range(size):
            labels[i] = struct.unpack_from(">B", data, offset)[0]
            offset += struct.calcsize(">B")

        return labels


def data_normalization(train=0, test=0, offset=0):
    # Normalize the images

    fac = (1 - offset) / 255  # normalization factor

    train = fac * train + offset
    test = fac * test + offset

    return train, test
