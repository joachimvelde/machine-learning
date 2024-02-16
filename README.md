# AI/ML in C
This is a private repository for exploring and implementing algorithms in AI/ML.

Note for your idiot brain: Even if the input of a layer is stored as a vector, the weights will still be stored in a matrix.
Example: 5 inputs => triple neuron layer: 3x5 weight matrix and 3x1 bias vector.
A = ø(W x I + b)

Also, remember to seed srand!

## Future optimizations:
Use memcpy where relevant: mat_copy and mat_fill

Apply loop fusion to most of the matrix loops - should be relatively easy

Apply loop unrolling and/or unroll and jam

Don't allocate and copy for the input vectors, just change the pointer at as[0]

## Potential features/implementations:
* Batch processing
* GPU-acceleration
