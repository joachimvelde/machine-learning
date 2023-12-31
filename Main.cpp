#include "NeuralNetwork.h"

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
