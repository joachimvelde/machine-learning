# AI/ML in C
This is a repository for exploring and implementing algorithms in AI/ML.

Currently, running train will train and test the network on the MNITS
dataset. The parameters will be stored in a binary file which will be loaded
when running main. The main binary will open a window in raylib, where the user
can draw digits. Right click clears the window. Press space to have the network guess which digit the user has
drawn. The results will be printed to the terminal.

[Add a gif/video showing the network guessing a hand drawn digit]

## Runs
For the whole MNIST dataset uploaded, a network with two hidden layers, with
500 and 100 neurons and a learning rate of 0.01 gave an accuracy of 94 percent
for the test set. The whole program, including reading the set, had a runtime
of just under nine minutes. Lowering the learning rate to 0.001 also lowered the
accuracy by about ten percent.

Also, remember to seed srand!

## Future optimizations:
Try to parallelise mat_mult, mat_sum and mat_hadamard

Use memcpy where relevant: mat_copy and mat_fill?

Try to apply some loop optimizations, like loop fusion

Don't allocate and copy for the input vectors, just change the pointer at as[0]

## Things to look into:
* Batch processing
* GPU-acceleration

## Bugs
* The initialised raylib window displays garbage, but this can be fixed by a
simple right click.