# Handwritten-Char-Recognition Assignment
Project for the Fundamental of Machine Learning Exam 2023.

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

- Learning rate
- Number of Neurons in the hidden layer
- Type of Activation Function for the hidden layer
- Type of Activation Function for the output layer
-Loss type
- Number of epochs
- Data encoding
- Batch size
- Different Weight Initialization

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



## Feedback

If you have any feedback, please reach out to us at meconcelliduccio@gmail.com or visit my website [duccio.me](http://www.duccio.me)
