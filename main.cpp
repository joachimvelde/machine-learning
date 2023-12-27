#include "matrix.h"

#include <iostream>
#include <vector>

class NeuralNetwork {
    private:
        size_t layers;
        Matrix train_out;

        std::vector<Matrix> ws;
        std::vector<Matrix> bs;
        std::vector<Matrix> as;

    public:
        NeuralNetwork(size_t arch[], size_t l, Matrix in, Matrix out);
        void print();
};

NeuralNetwork::NeuralNetwork(size_t arch[], size_t layers, Matrix in, Matrix out) : layers(layers), train_out(out) {
    as.emplace_back(in);

    for (size_t i = 1; i < layers; i++) {
        ws.emplace_back(Matrix(arch[i - 1], arch[i]));
        bs.emplace_back(Matrix(as.at(i - 1).get_rows(), arch[i]));
        as.emplace_back(Matrix(as.at(i - 1).get_rows(), arch[i]));
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
    Matrix in(3, 3);
    Matrix out(10, 1);

    size_t arch[] = { 3, 3, 3 };
    NeuralNetwork nn(arch, 3, in, out);

    nn.print();

    return 0;
}
