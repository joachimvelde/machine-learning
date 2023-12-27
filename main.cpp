#include "matrix.h"

#include <vector>

class NeuralNetwork {
    private:
        std::vector<Matrix> ws;
        std::vector<Matrix> bs;
        std::vector<Matrix> as;

    public:
        NeuralNetwork(size_t arch[], size_t layers, Matrix train_in, Matrix train_out);
};

NeuralNetwork::NeuralNetwork(size_t arch[], size_t layers, Matrix train_in, Matrix train_out) {
    as.emplace_back(train_in);

    for (size_t i = 1; i < layers; i++) {
        ws.emplace_back(arch[i], as.at(i - 1).get_rows());
        bs.emplace_back(arch[i], as.at(i - 1).get_rows());
        as.emplace_back(arch[i], as.at(i - 1).get_rows());
    }
}

int main()
{
    Matrix in(256, 256);
    Matrix out(10, 1);

    size_t arch[] = {1, 3, 2};
    NeuralNetwork nn(arch, 3, in, out);

    return 0;
}
