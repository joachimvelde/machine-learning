#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>

class Matrix {
    private:
        size_t rows;
        size_t cols;
        float *data;

    public:
        Matrix(size_t r, size_t c);
        virtual ~Matrix();
        float get(size_t r, size_t c) const;
        void set(size_t r, size_t c, float x);
        void fill(float x);
        void print();

        size_t get_rows();

        Matrix operator+ (const Matrix& x);
        Matrix operator- (const Matrix& x);
        Matrix operator* (const Matrix& x);

        // Copy constructor
        Matrix(const Matrix& x);
};

#endif
