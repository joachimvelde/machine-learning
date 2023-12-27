#include "matrix.h"
#include <iostream>

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

void Matrix::print() {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            std::cout << get(i, j) << ' ';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

size_t Matrix::get_rows() {
    return rows;
}

Matrix Matrix::operator+ (const Matrix& x) {
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
    Matrix temp(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            float diff = get(i, j) - x.get(i, j);
            temp.set(i, j, diff);
        }
    }
    return temp;
}

Matrix Matrix::operator* (const Matrix& x) {
    Matrix temp(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            temp.set(i, j, 0);
            for (size_t k = 0; k < cols; k++) {
                float product = get(i, k) * x.get(k, j);
                temp.set(i, j, temp.get(i, j) + product);
            }
        }
    }
    return temp;
}
