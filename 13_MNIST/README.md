# 13th Assignment: MNIST Problem Variation

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

## Grid Search

Ho usato la grid search per trovare i parametri che meglio si adattano al problema del riconoscimento dei caratteri con il dataset originale MNIST.

Parametri utilizzati:

- Learning Rate
- Number of Neurons in the hidden layer
- Type of Activation Function for the hidden layer
- Type of Activation Function for the output layer
- Loss type
- Number of epochs
- Data encoding
- Batch size
- Different Weight Initialization

## Dataset

Il dataset è stato preso dalla repo di [@fgnt](https://github.com/fgnt/mnist) ed è il dataset originale di Yann Lecun.

## Osservazioni:

- non c'è tanta differenza tra binary encoding e on-hot encoding, anche se va puntualizzato che il problema in questione è molto semplice e la rete con un singolo layer potrebbe influire su questo risultato.

- c'è una notevole differenza tra inizializzare i pesi tutti a zero e metterli invece casuali

- c'è una leggera differenza ad usare la **cross entropy loss** rispetto alla **quadratic loss** (meglio la cross entropy per problemi di classificazione)

- la più grande differenza si riscontra nella scelta del batch size: infatti con batch size a 1 (online mode) ottengo migliori risultati, simili risultati si otttengono con batch mode inferiore a quella della grandezza dell'intero training set

## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the training

```bash
  python Assignments/script/13_main.py
```

## Running Tests

To run tests, run the following command

```bash
  npm run test
```

## Feedback

If you have any feedback, please reach out to us at meconcelliduccio@gmail.com or visit my website [duccio.me](http://www.duccio.me)
