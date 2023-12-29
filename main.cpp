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
        NeuralNetwork(size_t arch[], size_t l, Matrix in, Matrix out);
        void forward();
        float cost();
        void backprop();
        void print();
};

NeuralNetwork::NeuralNetwork(size_t arch[], size_t layers, Matrix in, Matrix out) : layers(layers), train_out(out) {
    as.emplace_back(in);

    for (size_t i = 1; i < layers; i++) {
        ws.emplace_back(Matrix(arch[i - 1], arch[i]));
        bs.emplace_back(Matrix(as.at(i - 1).get_rows(), arch[i]));
        as.emplace_back(Matrix(as.at(i - 1).get_rows(), arch[i]));

        ws[i - 1].rand();
        bs[i - 1].rand();
    }
}

void NeuralNetwork::forward() {
    for (size_t i = 1; i < layers; i++) {
        as[i] = (as[i - 1] * ws[i - 1]) + bs[i - 1];
    }
}

float NeuralNetwork::cost() {
    float c = 0;

    Matrix out = as.back();

    for (size_t i = 0; i < out.get_rows(); i++) {
        c += pow((train_out.get(i, 0) - out.get(i, 0)), 2);
    }

    c /= out.get_rows();

    return c;
}

void NeuralNetwork::backprop() {

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
    // Generating random floats
    srand(69);

    Matrix in(4, 4);
    Matrix out(4, 1);
    in.fill(3.14f);

    size_t arch[] = { 4, 4 };
    NeuralNetwork nn(arch, 2, in, out);

    nn.print();
    std::cout << std::endl << "cost = " << nn.cost() << std::endl;
    nn.forward();
    std::cout << std::endl << "---------- FOWARDING ----------" << std::endl << std::endl;
    nn.print();
    std::cout << std::endl << "cost = " << nn.cost() << std::endl;

    return 0;
}
