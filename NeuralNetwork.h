#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Matrix.h"

#include <iostream>
#include <vector>
#include <cmath>

class NeuralNetwork {
    private:
        size_t layers;
        Matrix train_out;

        std::vector<Matrix> ws;
        std::vector<Matrix> bs;
        std::vector<Matrix> as;

    public:
        NeuralNetwork(size_t arch[], size_t layers, Matrix in, Matrix out);
        void forward();
        float cost();
        void backprop();
        void print();
};

#endif
