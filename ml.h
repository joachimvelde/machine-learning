#ifndef ML_H_
#define ML_H_

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define MAT_AT(m, i, j) m.data[m.cols * (j) + (i)]

typedef struct Matrix
{
    size_t rows, cols;
    double *data;
} Matrix;

Matrix mat_alloc(size_t rows, size_t cols);
void mat_fill(Matrix m, double x);
void mat_sum(Matrix dst, Matrix m);
void mat_dot(Matrix dst, Matrix a, Matrix b);
void mat_print(Matrix m);
void mat_free(Matrix m);

typedef struct Network
{
    size_t layer_count; // Should include the input layer
    Matrix *ws; // Weights
    Matrix *bs; // Biases
    Matrix *as; // Activations
} Network;



// The layers array should specify the number of neurons in each layer
Network net_alloc(size_t layer_count, size_t layers[]);
void net_free(Network n);

#endif // Ml_H_

#ifndef ML_IMPLEMENTATION

Matrix mat_alloc(size_t rows, size_t cols)
{
    double *data = (double *) calloc(rows * cols, sizeof(double));
    assert(data != NULL);
    Matrix m = { m.rows = rows, m.cols = cols, m.data = data };
    return m;
}

void mat_fill(Matrix m, double x)
{
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = x;
        }
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
void mat_dot(Matrix dst, Matrix a, Matrix b)
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
    n.ws = malloc(sizeof(*n.ws) * n.layer_count - 1);
    n.bs = malloc(sizeof(*n.ws) * n.layer_count - 1);
    n.as = malloc(sizeof(*n.ws) * n.layer_count);
    assert(n.ws != NULL && n.bs != NULL && n.as != NULL);

    n.as[0] = mat_alloc(layers[0], 1); // Stick with flattened data for now
    for (size_t i = 1; i < n.layer_count; i++) {
        n.ws[i] = mat_alloc(layers[i], layers[i-1]);
        n.bs[i] = mat_alloc(layers[i], 1);
        n.as[i] = mat_alloc(layers[i], layers[0]);
    }

    return n;
}

void net_free(Network n)
{
    mat_free(n.as[0]);
    for (size_t i = 1; i < n.layer_count; i++) {
        mat_free(n.ws[i]);
        mat_free(n.bs[i]);
        mat_free(n.as[i]);
    }

    free(n.ws);
    free(n.bs);
    free(n.as);
}

#endif // ML_IMPLEMENTATION
