#ifndef ML_H_
#define ML_H_

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define MAT_AT(m, i, j) m.data[m.cols * (j) + (i)]

// Data structures
typedef struct Matrix
{
    size_t rows, cols;
    double *data;
} Matrix;

// Function definitions
Matrix mat_alloc(size_t rows, size_t cols);
void mat_fill(Matrix m, double x);
void mat_sum(Matrix dst, Matrix m);
void mat_dot(Matrix dst, Matrix a, Matrix b);
void mat_print(Matrix m);
void mat_free(Matrix m);

// Function implementations
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

#endif // ML_IMPLEMENTATION
