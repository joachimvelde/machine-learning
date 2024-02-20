#ifndef ML_H_
#define ML_H_

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>


double sigmoid(double x);


#define MAT_AT(m, i, j) m.data[m.cols * (i) + (j)]

// Matrix conflicted with a type in raylib, so I had to change the name
typedef struct Mat
{
    size_t rows, cols;
    double *data;
} Mat;

Mat mat_alloc(size_t rows, size_t cols);
void mat_copy(Mat dst, Mat src);
void mat_fill(Mat m, double x);
void mat_flatten(Mat *m);
void mat_hadamard(Mat dst, Mat a, Mat b);
Mat mat_transpose(Mat m);
void mat_rand(Mat m, double min, double max);
void mat_scale(Mat m, double x);
void mat_sigmoid(Mat m);
void mat_sub(Mat dst, Mat m);
Mat mat_sub_from_f(double x, Mat m); // Allocates a new matrix with values x - m
void mat_sum(Mat dst, Mat m);
void mat_mult(Mat dst, Mat a, Mat b);
void mat_print(Mat m);
void mat_free(Mat m);

typedef struct Gradient
{
    Mat *ws;
    Mat *bs;
    Mat *ds; // Deltas
} Gradient;

typedef struct Network
{
    size_t layer_count; // Should include the input layer
    Mat *ws; // Weights
    Mat *bs; // Biases
    Mat *as; // Activations
    Gradient g;
} Network;


#define NET_IN(n) n.as[0]
#define NET_OUT(n) n.as[(n.layer_count - 1)]

// The layers array should specify the number of neurons in each layer
Network net_alloc(size_t layer_count, size_t layers[]);
void net_backprop(Network n, Mat target, double learning_rate);
void net_forward(Network n);
void net_free(Network n);
void net_load(Network n, char *filename);
double net_loss(Network n, Mat target);
void net_print(Network n);
void net_save(Network n, char *filename);
void net_train(Network n, Mat in, Mat target, double learning_rate);
void net_zero_gradient(Network n);


#endif // Ml_H_

#ifndef ML_IMPLEMENTATION


double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}


// Try to optimise some of these

Mat mat_alloc(size_t rows, size_t cols)
{
    double *data = (double *) calloc(rows * cols, sizeof(double));
    assert(data != NULL);
    Mat m = { .rows = rows, .cols = cols, .data = data };
    return m;
}

void mat_copy(Mat dst, Mat src)
{
    assert(dst.rows == src.rows);
    assert(dst.cols == src.cols);

    for (size_t i = 0; i < src.rows; i++) {
        for (size_t j = 0; j < src.cols; j++) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

void mat_fill(Mat m, double x)
{
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = x;
        }
    }
}

void mat_flatten(Mat *m)
{
    m->rows = m->rows * m->cols;
    m->cols = 1;
}

void mat_hadamard(Mat dst, Mat a, Mat b)
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

Mat mat_transpose(Mat m)
{
    Mat new = mat_alloc(m.cols, m.rows);
    
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(new, j, i) = MAT_AT(m, i, j);
        }
    }

    return new;
}

void mat_rand(Mat m, double min, double max)
{
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = min + (rand() / (RAND_MAX / (max - min))); // Maybe place in its own function
        }
    }
}

void mat_scale(Mat m, double x)
{
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) *= x;
        }
    }
}

void mat_sigmoid(Mat m)
{
    for (size_t i = 0; i < m.rows * m.cols; i++) {
        m.data[i] = sigmoid(m.data[i]);
    }
}

void mat_sub(Mat dst, Mat m)
{
    assert(dst.rows == m.rows);
    assert(dst.cols == m.cols);

    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            MAT_AT(dst, i, j) -= MAT_AT(m, i, j);
        }
    }
}

Mat mat_sub_from_f(double x, Mat m)
{
    Mat new = mat_alloc(m.rows, m.cols);

    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(new, i, j) = x - MAT_AT(m, i, j);
        }
    }

    return new;
}

void mat_sum(Mat dst, Mat m)
{
    assert(dst.rows == m.rows);
    assert(dst.cols == m.cols);

    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            MAT_AT(dst, i, j) += MAT_AT(m, i, j);
        }
    }
}

// Try to parallelise this?
void mat_mult(Mat dst, Mat a, Mat b)
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

void mat_print(Mat m)
{
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

void mat_free(Mat m)
{
    free(m.data);
}



Network net_alloc(size_t layer_count, size_t layers[])
{
    Network n;
    n.layer_count = layer_count;

    // Allocate arrays for parameters
    n.ws = (Mat *) malloc(sizeof(*n.ws) * (n.layer_count - 1));
    n.bs = (Mat *) malloc(sizeof(*n.bs) * (n.layer_count - 1));
    n.as = (Mat *) malloc(sizeof(*n.as) * n.layer_count);
    assert(n.ws != NULL && n.bs != NULL && n.as != NULL);

    // Allocate arrays for the gradient
    n.g.ws = (Mat *) malloc(sizeof(*n.g.ws) * (n.layer_count - 1));
    n.g.bs = (Mat *) malloc(sizeof(*n.g.bs) * (n.layer_count - 1));
    n.g.ds = (Mat *) malloc(sizeof(*n.g.ds) * n.layer_count);
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
void net_backprop(Network n, Mat target, double learning_rate)
{
    // Remember that the first matrix in n.as and g.ds is the input layer
    // This should probably be removed for g.ds, but we will do that later.
    // This means the current activation matrix at an index is as[i+1], not as[i].

    Mat o = NET_OUT(n);

    // Zero out the gradient
    net_zero_gradient(n);

    // Calculate the deltas for the output neurons first
    // delta = (o_j - t_j) * o_j * (1 - o_j)
    Mat deltas = n.g.ds[n.layer_count - 1];
    Mat one_minus_o = mat_sub_from_f(1, o);

    mat_copy(deltas, o);
    mat_sub(deltas, target);
    mat_hadamard(deltas, deltas, o);
    mat_hadamard(deltas, deltas, one_minus_o);
    mat_free(one_minus_o);

    // Gradient for the weights in the output layer
    Mat at = mat_transpose(n.as[n.layer_count - 2]);
    mat_mult(n.g.ws[n.layer_count - 2], n.g.ds[n.layer_count - 1], at);
    mat_scale(n.g.ws[n.layer_count - 2], learning_rate);
    mat_free(at);

    // Gradient for the biases in the output layer
    mat_copy(n.g.bs[n.layer_count - 2], n.g.ds[n.layer_count - 1]);
    mat_scale(n.g.bs[n.layer_count - 2], learning_rate);

    // Iterate backwards through each layer to calculate deltas
    for (int i = n.layer_count - 3; i >= 0; i--) {
        // delta = sum_l_in_L(w_jl * delta_l) * o_j * (1 - o_j)
        Mat wt = mat_transpose(n.ws[i+1]);
        Mat one_minus_o = mat_sub_from_f(1, n.as[i+1]);

        Mat delta_next = n.g.ds[i+2]; // Delta from next layer
        deltas = n.g.ds[i+1];

        mat_mult(deltas, wt, delta_next);
        mat_hadamard(deltas, deltas, n.as[i+1]);
        mat_hadamard(deltas, deltas, one_minus_o);

        mat_free(wt);
        mat_free(one_minus_o);

        // Finalize gradients and scale by learning rate;
        Mat at = mat_transpose(n.as[i]);
        mat_mult(n.g.ws[i], n.g.ds[i+1], at);
        mat_scale(n.g.ws[i], learning_rate);
        mat_free(at);

        // For the biases
        mat_copy(n.g.bs[i], n.g.ds[i+1]);
        mat_scale(n.g.bs[i], learning_rate);
    } 

    // Update parameters
    for (size_t i = 0; i < n.layer_count - 1; i++) {
        mat_sub(n.ws[i], n.g.ws[i]);
        mat_sub(n.bs[i], n.g.bs[i]);
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

void net_load(Network n, char *filename)
{
    FILE *f = fopen(filename, "rb");
    if (f == NULL) {
        perror("fopen failed");
        exit(EXIT_FAILURE);
    }

    size_t read = 0;
    for (size_t i = 0; i < n.layer_count - 1; i++) {
        read = fread(n.ws[i].data, sizeof(double), n.ws[i].rows * n.ws[i].cols, f);
        read += fread(n.bs[i].data, sizeof(double), n.bs[i].rows * n.bs[i].cols, f);
        if (read != n.ws[i].rows * n.ws[i].cols + n.bs[i].rows * n.bs[i].cols) {
            perror("fread failed while loading network");
            exit(EXIT_FAILURE);
        }
    }

    fclose(f);
}

double net_loss(Network n, Mat target)
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

void net_save(Network n, char *filename)
{
    FILE *f = fopen(filename, "wb");
    if (f == NULL) {
        perror("fopen failed");
        exit(EXIT_FAILURE);
    }

    // Store the weights and biases consecutively for each layer
    size_t written = 0;
    for (size_t i = 0; i < n.layer_count - 1; i++) {
        written = fwrite(n.ws[i].data, sizeof(double), n.ws[i].rows * n.ws[i].cols, f);
        written += fwrite(n.bs[i].data, sizeof(double), n.bs[i].rows * n.bs[i].cols, f);
        if (written != n.ws[i].rows * n.ws[i].cols + n.bs[i].rows * n.bs[i].cols) {
            perror("fwrite failed while saving network");
            exit(EXIT_FAILURE);
        }
    }

    fclose(f);
}

void net_train(Network n, Mat in, Mat target, double learning_rate)
{
    assert(NET_IN(n).rows == in.rows);
    assert(NET_OUT(n).rows == target.rows);

    // Set the input data
    mat_copy(NET_IN(n), in);

    // Forward pass
    net_forward(n);

    // Backprop
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
