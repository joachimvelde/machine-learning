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
        void forward();
};

NeuralNetwork::NeuralNetwork(size_t arch[], size_t layers, Matrix in, Matrix out) : layers(layers), train_out(out) {
    as.emplace_back(in);

    for (size_t i = 1; i < layers; i++) {
        ws.emplace_back(Matrix(arch[i - 1], arch[i]));
        bs.emplace_back(Matrix(as.at(i - 1).get_rows(), arch[i]));
        as.emplace_back(Matrix(as.at(i - 1).get_rows(), arch[i]));

        ws[i - 1].fill(3.0f);
        bs[i - 1].fill(5.0f);
    }
}

void NeuralNetwork::forward() {
    for (size_t i = 1; i < layers; i++) {
        std::cout << "multiplying " << as[i - 1].get_rows() << "x" << as[i - 1].get_cols()
            << " with " << ws[i - 1].get_rows() << "x" << ws[i - 1].get_cols() << std::endl;
        as[i] = (as[i - 1] * ws[i - 1]) + bs[i - 1];
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
    Matrix out(2, 2);
    in.fill(3.14f);

    size_t arch[] = { 3, 4, 2 };
    NeuralNetwork nn(arch, 3, in, out);

    // in.print();
    // out.print();
    // (in * out).print();

    nn.print();
    nn.forward();
    nn.print();

    return 0;
}
