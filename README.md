# Handwritten-Char-Recognition Assignment
Project for the Fundamental of Machine Learning Exam 2023.

In this study, we trained a neural network model with **one hidden layer** on the MNIST dataset for
handwritten digit recognition. Through a grid search, we identify the optimal hyperparameters for the model, achieving an accuracy
of 98% on the test set with a single hidden layer network. This study demonstrates the importance of
selecting appropriate hyperparameters, particularly batch size, the loss type and the weights initialization,
in achieving high accuracy and computational efficiency in training neural network models for handwritten
digit recognition.


The original text of the Assignment:

> Based on the Python scripts and on the MNIST data made available write a program for handwritten char recognition whose objectives are the following:
>
> 1. Test the performances by using the Quadratic and the Entropy Loss

> 2.  Test the performance and discuss the efficiency of the following learning mode protocols:
>
> - batch mode
> - on-line mode
> - mini-batch mode

> 3.  Discuss the role of the weight initialization for both the Quadratic and Entropy loss'

> 4. discuss the difference between 1-hot encoding and binary encoding

## Report
[Link to the Report](https://duccioo.github.io/Handwritten-Char-Recognition/report_ML_MNIST.pdf)
## Grid Search

I used grid search to find the parameters that best fit the character recognition problem with the original MNIST dataset.

Parameters used:

- Learning rate `(0.1, 0.2, 0.05)`
- Number of Neurons in the hidden layer `(100, 50)`
- Type of Activation Function for the hidden layer `(sigmoid, relu)`
- Type of Activation Function for the output layer `(sigmoid, relu)`
- Loss type `("cross-entropy", "quadratic")`
- Number of epochs `(10,20)`
- Data encoding `("one_hot", "binary")`
- Batch size `(1,32,256,60000)`
- Different Weight Initialization `("Xavier", "random", 0,1, "He")`

## Dataset

The dataset was taken from the [@fgnt](https://github.com/fgnt/mnist) repo and is the original dataset by Yann Lecun.

## Run Locally

Clone the project

```bash
  git clone https://github.com/Duccioo/Handwritten-Char-Recognition.git
```

Go to the project directory

```bash
  cd Handwritten-Char-Recognition
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the training

```bash
  python Assignments/script/13_main.py
```

## Result
Best model with 10 epoch,	100 hidden neurons,	0.2 learning rate,	cross-entropy loss,	binary encoding,	32 batch size, 	Xavier weight init,	ReLu hidden activation function,	Sigmoid output activation function, 98% accuracy in	49.75 seconds


## Feedback

If you have any feedback, please reach out to us at meconcelliduccio@gmail.com or visit my website [duccio.me](http://www.duccio.me)
