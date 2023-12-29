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

        // Create the gradient here

        rows = as[i].get_rows(); 
    }

    assert(as.back().get_rows() == out.get_rows());
}

void NeuralNetwork::forward() {
    for (size_t i = 1; i < layers; i++) {
        as[i] = (ws[i - 1] * as[i - 1]) + bs[i - 1];
    }
}

float NeuralNetwork::cost() {
    float c = 0;

    Matrix out = as.back();

    for (size_t i = 0; i < out.get_rows(); i++) {
        c += pow((train_out.get(i, 0) - out.get(i, 0)), 2);
    }

    return c / out.get_rows();
}

void NeuralNetwork::backprop() {
    std::cout << "\nbackpropagating" << std::endl;

    forward();

    std::cout << "as.back:" << std::endl;
    as.back().print();
    std::cout << "train_out:" << std::endl;
    train_out.print();

    Matrix delta_output = (as.back() - train_out) * 2;

    for (int i = layers - 1; i >= 0; i--) {
        delta_output.print();
        as[i + 1].print();
        Matrix delta_layer = delta_output * as[i + 1];
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

    Matrix in(4, 1); // This is fucked up, but we need one column
    Matrix out(2, 1);
    in.fill(3.14f);
    out.fill(2.7);

    size_t arch[] = { 4, 3, 2 };
    size_t layers = (size_t) (sizeof(arch) / sizeof(arch[0]));
    NeuralNetwork nn(arch, layers, in, out);

    nn.print();
    std::cout << std::endl << "cost = " << nn.cost() << std::endl;
    nn.forward();
    std::cout << std::endl << "---------- FOWARDING ----------" << std::endl << std::endl;
    nn.print();
    std::cout << std::endl << "cost = " << nn.cost() << std::endl;

    nn.backprop();

    return 0;
}
