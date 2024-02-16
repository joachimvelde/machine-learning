#ifndef ML_H_
#define ML_H_

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>


double sigmoid(double x);


#define MAT_AT(m, i, j) m.data[m.cols * (i) + (j)]

typedef struct Matrix
{
    size_t rows, cols;
    double *data;
} Matrix;

Matrix mat_alloc(size_t rows, size_t cols);
void mat_copy(Matrix dst, Matrix src);
void mat_fill(Matrix m, double x);
void mat_flatten(Matrix *m);
void mat_hadamard(Matrix dst, Matrix a, Matrix b);
void mat_normalise(Matrix m); // Used to normalise the dataset
Matrix mat_transpose(Matrix m);
void mat_rand(Matrix m, double min, double max);
void mat_scale(Matrix m, double x);
void mat_sigmoid(Matrix m);
void mat_sub(Matrix dst, Matrix m);
Matrix mat_sub_from_f(double x, Matrix m); // Allocates a new matrix with values x - m
void mat_sum(Matrix dst, Matrix m);
void mat_mult(Matrix dst, Matrix a, Matrix b);
void mat_print(Matrix m);
void mat_free(Matrix m);

typedef struct Gradient
{
    Matrix *ws;
    Matrix *bs;
    Matrix *ds; // Deltas
} Gradient;

typedef struct Network
{
    size_t layer_count; // Should include the input layer
    Matrix *ws; // Weights
    Matrix *bs; // Biases
    Matrix *as; // Activations
    Gradient g;
} Network;


#define NET_IN(n) n.as[0]
#define NET_OUT(n) n.as[(n.layer_count - 1)]

// The layers array should specify the number of neurons in each layer
Network net_alloc(size_t layer_count, size_t layers[]);
void net_backprop(Network n, Matrix target, double learning_rate);
void net_forward(Network n);
void net_free(Network n);
double net_loss(Network n, Matrix target);
void net_print(Network n);
void net_train(Network n, Matrix in, Matrix target, double learning_rate);
void net_zero_gradient(Network n);

/* Make a train-function that accepts the input and expected output.
   This function will have to be called for every forward/backprop.

   Later improvenemt: Accept the whole dataset at initialization, and have the
   train-function train the network on the dataset a given number of times */

#endif // Ml_H_

#ifndef ML_IMPLEMENTATION


double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}


// A good few of these functions could be optimized by applying loop collapsing

Matrix mat_alloc(size_t rows, size_t cols)
{
    double *data = (double *) calloc(rows * cols, sizeof(double));
    assert(data != NULL);
    Matrix m = { .rows = rows, .cols = cols, .data = data };
    return m;
}

void mat_copy(Matrix dst, Matrix src)
{
    assert(dst.rows == src.rows);
    assert(dst.cols == src.cols);

    for (size_t i = 0; i < src.rows; i++) {
        for (size_t j = 0; j < src.cols; j++) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

void mat_fill(Matrix m, double x)
{
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = x;
        }
    }
}

void mat_flatten(Matrix *m)
{
    m->rows = m->rows * m->cols;
    m->cols = 1;
}

void mat_hadamard(Matrix dst, Matrix a, Matrix b)
{
    assert(dst.rows == a.rows);
    assert(dst.rows == b.rows);
    assert(a.rows == b.rows);
    assert(dst.cols == a.cols);
    assert(dst.cols == b.cols);
    assert(a.cols == b.cols);

    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            MAT_AT(dst, i, j) = MAT_AT(a, i, j) * MAT_AT(b, i, j);
        }
    }
}

Matrix mat_transpose(Matrix m)
{
    Matrix new = mat_alloc(m.cols, m.rows);
    
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(new, j, i) = MAT_AT(m, i, j);
        }
    }

    return new;
}

void mat_rand(Matrix m, double min, double max)
{
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = min + (rand() / (RAND_MAX / (max - min))); // Maybe place in its own function
        }
    }
}

void mat_scale(Matrix m, double x)
{
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) *= x;
        }
    }
}

void mat_sigmoid(Matrix m)
{
    for (size_t i = 0; i < m.rows * m.cols; i++) {
        m.data[i] = sigmoid(m.data[i]);
    }
}

void mat_sub(Matrix dst, Matrix m)
{
    assert(dst.rows == m.rows);
    assert(dst.cols == m.cols);

    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            MAT_AT(dst, i, j) -= MAT_AT(m, i, j);
        }
    }
}

Matrix mat_sub_from_f(double x, Matrix m)
{
    Matrix new = mat_alloc(m.rows, m.cols);

    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(new, i, j) = x - MAT_AT(m, i, j);
        }
    }

    return new;
}

void mat_sum(Matrix dst, Matrix m)
{
    assert(dst.rows == m.rows);
    assert(dst.cols == m.cols);

    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            MAT_AT(dst, i, j) += MAT_AT(m, i, j);
        }
    }
}

// Optimize this for cache hits
void mat_mult(Matrix dst, Matrix a, Matrix b)
{
    assert(a.cols == b.rows);
    assert(dst.rows == a.rows);
    assert(dst.cols == b.cols);

    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < b.cols; j++) {
            MAT_AT(dst, i, j) = 0.0;
            for (size_t k = 0; k < a.cols; k++) {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

void mat_print(Matrix m)
{
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

void mat_free(Matrix m)
{
    free(m.data);
}



Network net_alloc(size_t layer_count, size_t layers[])
{
    Network n;
    n.layer_count = layer_count;

    // Allocate arrays for parameters
    n.ws = (Matrix *) malloc(sizeof(*n.ws) * (n.layer_count - 1));
    n.bs = (Matrix *) malloc(sizeof(*n.bs) * (n.layer_count - 1));
    n.as = (Matrix *) malloc(sizeof(*n.as) * n.layer_count);
    assert(n.ws != NULL && n.bs != NULL && n.as != NULL);

    // Allocate arrays for the gradient
    n.g.ws = (Matrix *) malloc(sizeof(*n.g.ws) * (n.layer_count - 1));
    n.g.bs = (Matrix *) malloc(sizeof(*n.g.bs) * (n.layer_count - 1));
    n.g.ds = (Matrix *) malloc(sizeof(*n.g.ds) * n.layer_count);
    assert(n.g.ws != NULL && n.g.bs != NULL && n.g.ds != NULL);

    // Allocate and initialize architecture
    n.as[0] = mat_alloc(layers[0], 1); // Stick with flattened data for now
    n.g.ds[0] = mat_alloc(layers[0], 1); // Stick with flattened data for now
    for (size_t i = 1; i < n.layer_count; i++) {
        n.ws[i-1] = mat_alloc(layers[i], layers[i-1]);
        n.bs[i-1] = mat_alloc(layers[i], 1);
        n.as[i] = mat_alloc(layers[i], 1);

        mat_rand(n.ws[i-1], -1.0, 1.0);
        mat_rand(n.bs[i-1], -1.0, 1.0);

        n.g.ws[i-1] = mat_alloc(layers[i], layers[i-1]);
        n.g.bs[i-1] = mat_alloc(layers[i], 1);
        n.g.ds[i] = mat_alloc(layers[i], 1);
    }

    return n;
}

// Check the wikipedia page for backpropagation for further explanation
void net_backprop(Network n, Matrix target, double learning_rate)
{
    // Remember that the first matrix in n.as and g.ds is the input layer
    // This should probably be removed for g.ds, but we will do that later.
    // This means the current activation matrix at an index is as[i+1], not as[i].

    Matrix o = NET_OUT(n);

    // Zero out the gradient
    net_zero_gradient(n);

    // Calculate the deltas for the output neuros first
    // delta = (o_j - t_j) * o_j * (1 - o_j)
    Matrix deltas = n.g.ds[n.layer_count - 1];
    Matrix one_minus_o = mat_sub_from_f(1, o);

    mat_copy(deltas, o);
    mat_sub(deltas, target);
    mat_hadamard(deltas, deltas, o);
    mat_hadamard(deltas, deltas, one_minus_o);
    mat_free(one_minus_o);

    // Iterate backwards through each layer to calculate deltas
    for (int i = n.layer_count - 3; i >= 0; i--) {
        // delta = sum_l_in_L(w_jl * delta_l) * o_j * (1 - o_j)
        Matrix wt = mat_transpose(n.ws[i+1]);
        Matrix one_minus_o = mat_sub_from_f(1, n.as[i+1]);

        Matrix delta_next = n.g.ds[i+2]; // Delta from next layer
        deltas = n.g.ds[i+1];

        mat_mult(deltas, wt, delta_next);
        mat_hadamard(deltas, deltas, n.as[i+1]);
        mat_hadamard(deltas, deltas, one_minus_o);

        mat_free(wt);
        mat_free(one_minus_o);

        // Oopsie, does this compute the gradient for the output layer? check loop index
        // Finalize gradients and scale by learning rate;
        Matrix at = mat_transpose(n.as[i]);

        // printf("MULT ON %zu x %zu and %zu x %zu into %zu x %zu\n",
        //        at.rows, at.cols, n.g.ds[i+1].rows, n.g.ds[i+1].cols,
        //        n.g.ws[i].rows, n.g.ws[i].cols);

        mat_mult(n.g.ws[i], n.g.ds[i+1], at);
        mat_scale(n.g.ws[i], learning_rate);

        mat_free(at);
    }

    // Update parameters
    for (size_t i = 0; i < n.layer_count - 1; i++) {
        mat_sub(n.ws[i], n.g.ws[i]);
    }
}

void net_forward(Network n)
{
    for (size_t i = 0; i < n.layer_count - 1; i++) {
        mat_mult(n.as[i+1], n.ws[i], n.as[i]);
        mat_sum(n.as[i+1], n.bs[i]);
        mat_sigmoid(n.as[i+1]);
    }
}

void net_free(Network n)
{
    mat_free(n.as[0]);
    mat_free(n.g.ds[0]);
    for (size_t i = 1; i < n.layer_count; i++) {
        mat_free(n.ws[i-1]);
        mat_free(n.bs[i-1]);
        mat_free(n.as[i]);

        mat_free(n.g.ws[i-1]);
        mat_free(n.g.bs[i-1]);
        mat_free(n.g.ds[i]);
    }

    free(n.ws);
    free(n.bs);
    free(n.as);

    free(n.g.ws);
    free(n.g.bs);
    free(n.g.ds);
}

double net_loss(Network n, Matrix target)
{
    assert(NET_OUT(n).rows == target.rows);
    assert(NET_OUT(n).cols == target.cols);

    net_forward(n);

    double l = 0;
    for (size_t i = 0; i < NET_OUT(n).rows; i++) {
        l += pow((MAT_AT(NET_OUT(n), i, 0)) - MAT_AT(target, i, 0), 2);
    }

    return l / NET_OUT(n).rows;
}

// For debugging
void net_print(Network n)
{
    printf("Activations:\n");
    for (size_t i = 0; i < n.layer_count; i++) {
        printf("%zu x %zu\n", n.as[i].rows, n.as[i].cols);
    }

    printf("Weights:\n");
    for (size_t i = 0; i < n.layer_count - 1; i++) {
        printf("%zu x %zu\n", n.ws[i].rows, n.ws[i].cols);
    }
    printf("\n");

    printf("Biases:\n");
    for (size_t i = 0; i < n.layer_count - 1; i++) {
        printf("%zu x %zu\n", n.bs[i].rows, n.bs[i].cols);
    }
    printf("\n");
}

void net_train(Network n, Matrix in, Matrix target, double learning_rate)
{
    assert(NET_IN(n).rows == in.rows);
    assert(NET_OUT(n).rows == target.rows);

    // Set the input data
    mat_copy(NET_IN(n), in);

    // Forward pass
    net_forward(n);

    // Call backprop()
    net_backprop(n, target, learning_rate);
}

void net_zero_gradient(Network n)
{
    for (size_t i = 0; i < n.layer_count - 1; i++) {
        mat_fill(n.g.ws[i], 0.0);
        mat_fill(n.g.bs[i], 0.0);
        mat_fill(n.g.ds[i], 0.0);
    }
    mat_fill(n.g.ds[n.layer_count - 1], 0.0);
}

#endif // ML_IMPLEMENTATION
