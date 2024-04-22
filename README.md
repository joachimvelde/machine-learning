# AI/ML in C
This is a repository for exploring and implementing algorithms in AI/ML.

The networks uses the sigmoid activation function, and there are many potential
improvements that could be made. Classifying images is a task better suited for
convolutional neural networks, and ReLU would be interesting to test and compare to the sigmoid.
This program is also single-threaded and only runs on the CPU. Utilizing the GPU would likelygive much shorter runtimes.

Currently, running train will train and test the network on the MNIST dataset.
The parameters will be stored in a binary file which will be loaded when running main.
The main binary will open a window in raylib, where the usercan draw digits. Right click clears the window.
Press space to have the network guess which digit has beendrawn. The results will be printed to the terminal.

<video src='https://github.com/joachimvelde/machine-learning/blob/main/demo.mp4' width=180/>

## Runs
For the whole MNIST dataset uploaded, a network with two hidden layers, with
500 and 100 neurons and a learning rate of 0.01 gave an accuracy of 94 percentfor the test set.
The whole program, including reading the set, had a runtimeof just under nine minutes.
Lowering the learning rate to 0.001 also lowered theaccuracy by about ten percent.

Also, remember to seed srand!

## Future potential optimizations:
Try to parallelise mat_mult, mat_sum and mat_hadamard
Use memcpy where relevant: mat_copy and mat_fill?
Try to apply some loop optimizations, like loop fusion
Don't allocate and copy for the input vectors, just change the pointer at as[0]

## Things to look into:
* Batch processing
* GPU-acceleration

## Bugs
* The initialised raylib window displays garbage, but this can be fixed by asimple right click.

The MNIST-dataset was downloaded from [this website.](http://yann.lecun.com/exdb/mnist/)

This project was partly inspired by [this](https://www.youtube.com/playlist?list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw) playlist on machine learning.

