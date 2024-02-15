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
void mat_rand(Matrix m, double min, double max);
void mat_sigmoid(Matrix m);
void mat_sum(Matrix dst, Matrix m);
void mat_mult(Matrix dst, Matrix a, Matrix b);
void mat_print(Matrix m);
void mat_free(Matrix m);

typedef struct Network
{
    size_t layer_count; // Should include the input layer
    Matrix *ws; // Weights
    Matrix *bs; // Biases
    Matrix *as; // Activations
} Network;


#define NET_IN(n) n.as[0]
#define NET_OUT(n) n.as[(n.layer_count - 1)]

// The layers array should specify the number of neurons in each layer
Network net_alloc(size_t layer_count, size_t layers[]);
void net_forward(Network n);
void net_free(Network n);
double net_loss(Network n, Matrix target);
void net_print(Network n);
void net_train(Network n, Matrix in, Matrix target);
/* Make a train-function that accepts the input and expected output.
   This function will have to be called for every forward/backprop.

   Later improvenemt: Accept the whole dataset at initialization, and have the
   train-function train the network on the dataset a given number of times */

// Set input data (train-function) -> forward -> calculate loss -> backprop

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

void mat_rand(Matrix m, double min, double max)
{
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = min + (rand() / (RAND_MAX / (max - min))); // Maybe place in its own function
        }
    }
}

void mat_sigmoid(Matrix m)
{
    for (size_t i = 0; i < m.rows * m.cols; i++) {
        m.data[i] = sigmoid(m.data[i]);
    }
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
    n.ws = (Matrix *) malloc(sizeof(*n.ws) * (n.layer_count - 1));
    n.bs = (Matrix *) malloc(sizeof(*n.ws) * (n.layer_count - 1));
    n.as = (Matrix *) malloc(sizeof(*n.ws) * n.layer_count);
    assert(n.ws != NULL && n.bs != NULL && n.as != NULL);

    // Allocate and initialize architecture
    n.as[0] = mat_alloc(layers[0], 1); // Stick with flattened data for now
    for (size_t i = 1; i < n.layer_count; i++) {
        n.ws[i-1] = mat_alloc(layers[i], layers[i-1]);
        n.bs[i-1] = mat_alloc(layers[i], 1);
        n.as[i] = mat_alloc(layers[i], 1);

        mat_rand(n.ws[i-1], -1.0, 1.0);
        mat_rand(n.bs[i-1], -1.0, 1.0);
    }

    return n;
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
    for (size_t i = 1; i < n.layer_count; i++) {
        mat_free(n.ws[i-1]);
        mat_free(n.bs[i-1]);
        mat_free(n.as[i]);
    }

    free(n.ws);
    free(n.bs);
    free(n.as);
}

double net_loss(Network n, Matrix target)
{
    assert(NET_IN(n).rows == target.rows);
    assert(NET_IN(n).cols == target.cols);

    net_forward(n);

    double l = 0;
    for (size_t i = 0; i < NET_OUT(n).rows; i++) {
        l += pow((MAT_AT(NET_IN(n), i, 0) - MAT_AT(NET_OUT(n), i, 0)), 2);
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

void net_train(Network n, Matrix in, Matrix target)
{
    // Set the input data
    mat_copy(NET_IN(n), in);

    // Call backprop()
}

#endif // ML_IMPLEMENTATION
