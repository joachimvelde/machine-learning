#include "matrix.h"

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

NeuralNetwork::NeuralNetwork(size_t arch[], size_t layers, Matrix in, Matrix out) : layers(layers), train_out(out) {
    // Check that the input matches the first layer, and the output the last layer
    assert(in.get_rows() == arch[0]);
    assert(out.get_rows() == arch[layers - 1]);

    as.emplace_back(in);

    size_t rows = in.get_rows();
    size_t cols = in.get_cols();

    for (size_t i = 1; i < layers; i++) {
        size_t n_count = arch[i];
        ws.emplace_back(Matrix(n_count, rows));
        bs.emplace_back(Matrix(n_count, cols));
        as.emplace_back(Matrix(n_count, cols));
        std::cout << "created as with " << n_count << " rows and " << cols << " columns"  << std::endl;

        ws[i - 1].rand();
        bs[i - 1].rand();
        as[i].fill(0.0f); // just to make sure

        rows = as[i].get_rows(); 
    }

    assert(as.back().get_rows() == out.get_rows());
}

void NeuralNetwork::forward() {
    for (size_t i = 1; i < layers; i++) {
        as[i] = (ws[i - 1] * as[i - 1]) + bs[i - 1];
        as[i] = as[i].sigmoid();
    }
}

float NeuralNetwork::cost() {
    forward();

    float c = 0;
    Matrix out = as.back();

    for (size_t i = 0; i < out.get_rows(); i++) {
        c += pow((train_out.get(i, 0) - out.get(i, 0)), 2);
    }

    return c / out.get_rows();
}

void NeuralNetwork::backprop() {
    forward();

    std::vector<Matrix> wd;
    std::vector<Matrix> bd;
    for (size_t i = 0; i < ws.size(); i++) {
        wd.push_back(Matrix(ws[i].get_rows(), ws[i].get_cols()));
        bd.push_back(Matrix(bs[i].get_rows(), bs[i].get_cols()));
    }

    Matrix delta_output = (as.back() - train_out) * 2;

    for (int layer = layers - 2; layer >= 0; layer--) {
        Matrix delta_layer = delta_output.hadamard(as[layer + 1].sigmoid_diff());

        wd[layer] = delta_layer * as[layer].transpose();
        bd[layer] = delta_layer;

        if (layer > 0) {
            delta_output = ws[layer].transpose() * delta_layer;
        }
    }

    float learning_rate = 0.001f; // Should define this somewhere else

    for (size_t layer = 0; layer < layers - 1; layer++) {
        ws[layer] = ws[layer] - wd[layer] * learning_rate;
        bs[layer] = bs[layer] - bd[layer] * learning_rate;
    }
}

void NeuralNetwork::print() {
    std::cout << "Activations\n";
    for (size_t i = 0; i < layers; i++) {
        as.at(i).print();
    }

    std::cout << "Weights\n";
    for (size_t i = 0; i < layers - 1; i++) {
        ws.at(i).print();
    }

    std::cout << "Biases\n";
    for (size_t i = 0; i < layers - 1; i++) {
        bs.at(i).print();
    }
}

int main()
{
    srand(69); // Seeding for pseudorandom numbers

    // Define the training data. The data should only have one column.
    Matrix in(4, 1);
    Matrix out(4, 1);
    in.fill(3.14f);
    out.fill(0.5f);

    // The input and output layers defined in arch have to match the in and out data.
    size_t arch[] = { 4, 3, 4 };
    size_t layers = (size_t) (sizeof(arch) / sizeof(arch[0]));
    NeuralNetwork nn(arch, layers, in, out);

    nn.forward();
    nn.print();
    std::cout << "cost = " << nn.cost() << std::endl;

    std::cout << "---------- TRAINING ----------" << std::endl;
    for (size_t i = 0; i < 10000; i++) {
        nn.backprop();
    }

    nn.print();
    std::cout << "cost = " << nn.cost() << std::endl;

    return 0;
}
