#include "ml.h"

// Training and testing example for the MNIST dataset
// This particular set's header is stored in big-endian format.
// For little-endian computer the header must be flipped.

int swap_endian(int x)
{
    return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) | ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
}

Matrix *read_labels(char *path, size_t N)
{
    FILE *f = fopen(path, "rb");

    Matrix *labels = malloc(N*sizeof(Matrix));

    // Read the magic number
    int magic = 0;
    fread(&magic, sizeof(int), 1, f);
    magic = swap_endian(magic);

    // Read the number of items
    int num_items = 0;
    fread(&num_items, sizeof(int), 1, f);
    num_items = swap_endian(num_items);

    // Convert the labels into matrices
    for (size_t i = 0; i < N && i < (size_t) num_items; i++) {
        unsigned char label;
        fread(&label, sizeof(char), 1, f);

        Matrix l_matrix = mat_alloc(10, 1);
        MAT_AT(l_matrix, (size_t) label, 0) = 1.0;

        labels[i] = l_matrix;
    }

    fclose(f);

    return labels;
}

Matrix *read_inputs(char *path, size_t N)
{
    FILE *f = fopen(path, "rb");

    Matrix *inputs = malloc(N*sizeof(Matrix));

    // Read magic number

    // Read number of images

    // Read number of rows

    // Read number of columns

    // Read the images

    fclose(f);

    return inputs;
}

void free_data(Matrix *data, size_t N)
{
    for (size_t i = 0; i < N; i++) {
        mat_free(data[i]);
    }
    free(data);
}

int main()
{
    size_t N = 60000;

    // One array for storing the labels
    Matrix *labels = read_labels("datasets/train-labels-idx1-ubyte/train-labels.idx1-ubyte", N);

    // One array for storing the image matrices

    free_data(labels, N);
    
    return 0;
}
