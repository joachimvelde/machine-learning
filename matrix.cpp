#include "Matrix.h"
#include <iostream>
#include <assert.h>
#include <cmath>

Matrix::Matrix(size_t r, size_t c) {
    rows = r;
    cols = c;
    data = new float[rows * cols];
}

Matrix::~Matrix() {
    delete[] data;
}

float Matrix::get(size_t r, size_t c) const {
    return data[r * cols + c];
}

void Matrix::set(size_t r, size_t c, float x) {
    data[r * cols + c] = x;
}

void Matrix::fill(float x) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            set(i, j, x);
        }
    }
}

void Matrix::rand() {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            set(i, j, randf());
        }
    }
}

void Matrix::print() {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            std::cout << get(i, j) << ' ';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

Matrix Matrix::transpose() {
    Matrix temp(cols, rows);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            temp.set(j, i, get(i, j));
        }
    }
    return temp;
}

Matrix Matrix::sigmoid() {
    Matrix temp(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            temp.set(i, j, sigmoidf(get(i, j)));
        }
    }
    return temp;
}

Matrix Matrix::sigmoid_diff() {
    Matrix temp(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            temp.set(i, j, sigmoid_difff(get(i, j)));
        }
    }
    return temp;
}

Matrix Matrix::hadamard(const Matrix& x) {
    assert(rows == x.rows && cols == x.cols);

    Matrix temp(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            temp.set(i, j, (get(i, j) * x.get(i, j)));
        }
    }
    return temp;
}

size_t Matrix::get_rows() {
    return rows;
}

size_t Matrix::get_cols() {
    return cols;
}

Matrix Matrix::operator+ (const Matrix& x) {
    assert(rows == x.rows && cols == x.cols);

    Matrix temp(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            float sum = get(i, j) + x.get(i, j);
            temp.set(i, j, sum);
        }
    }
    return temp;
}

Matrix Matrix::operator- (const Matrix& x) {
    assert(rows == x.rows && cols == x.cols);

    Matrix temp(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            float diff = get(i, j) - x.get(i, j);
            temp.set(i, j, diff);
        }
    }
    return temp;
}

Matrix Matrix::operator* (float x) {
    Matrix temp(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            temp.set(i, j, get(i, j) * x);
        }
    }
    return temp;
}

Matrix Matrix::operator* (const Matrix& x) {
    assert(cols == x.rows);

    Matrix temp(rows, x.cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < x.cols; j++) {
            temp.set(i, j, 0);
            for (size_t k = 0; k < cols; k++) {
                float product = get(i, k) * x.get(k, j);
                temp.set(i, j, temp.get(i, j) + product);
            }
        }
    }
    return temp;
}

Matrix Matrix::operator= (const Matrix& x) {
    if (this != &x) {
        float *new_data = new float[x.rows * x.cols];
        std::copy(x.data, x.data + x.rows * x.cols, new_data);

        delete[] data;

        data = new_data;
        rows = x.rows;
        cols = x.cols;
    }
    return *this;
}

// Copy constructor
Matrix::Matrix(const Matrix& x) : rows(x.rows), cols(x.cols), data(new float[rows * cols]) {
    std::copy(x.data, x.data + (rows * cols), data);
}

float randf()
{
    return (float) (rand()) / (float) (RAND_MAX);
}

float sigmoidf(float x) {
    return 1 / (1 + exp(-x));
}

// Good job naming this
float sigmoid_difff(float x) {
    return x * (1 - x);
}
