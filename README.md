# AI/ML in C
This is a private repository for exploring and implementing algorithms in AI/ML.

## Runs
For the whole MNIST dataset uploaded, a network with two hidden layers, with
500 and 100 neurons and a learning rate of 0.01 gave an accuracy of 94 percent
for the test set. The whole program, including reading the set, had a runtime
of just under nine minutes. Lowering the learning rate to 0.001 also lowered the
accuracy by about ten percent.

## Ramblings
Note for your idiot brain: Even if the input of a layer is stored as a vector, the weights will still be stored in a matrix.
Example: 5 inputs => triple neuron layer: 3x5 weight matrix and 3x1 bias vector.
A = ø(W x I + b)

Also, remember to seed srand!

## Future optimizations:
Try to parallelise mat_mult, mat_sum and mat_hadamard

Use memcpy where relevant: mat_copy and mat_fill

Try to apply some loop optimizations, like loop fusion

Don't allocate and copy for the input vectors, just change the pointer at as[0]

## Potential features/implementations:
* Batch processing
* GPU-acceleration

## Bugs
* This was not actually a bug, I just forgot to set the input of the network
before the first call to the loss function. Let this be a lesson.
Sometimes calling backprop a single time will actually make the loss greater. This could potentially be an issue with using too high of a learning rate, but I am not sure.
Maybe this is because I have not yet added the bias gradient to the backprop-function.
