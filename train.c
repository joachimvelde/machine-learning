#include "ml.h"

// Training and testing example for the MNIST dataset
// This particular set's header is stored in big-endian format.
// For little-endian computer the header must be flipped.

int swap_endian(int x)
{
    return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) | ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
}

Mat *read_labels(char *path, size_t N)
{
    FILE *f = fopen(path, "rb"); // Should probably check this return value, though
    size_t ret = 0; // Just to get fewer warnings, these calls work 99% of the time

    Mat *labels = malloc(N*sizeof(Mat));

    // Read the magic number
    int magic = 0;
    ret = fread(&magic, sizeof(int), 1, f);
    magic = swap_endian(magic);

    // Read the number of items
    int num_items = 0;
    ret = fread(&num_items, sizeof(int), 1, f);
    num_items = swap_endian(num_items);

    // Convert the labels into matrices
    for (size_t i = 0; i < N && i < (size_t) num_items; i++) {
        unsigned char label;
        ret = fread(&label, sizeof(char), 1, f);

        Mat l_matrix = mat_alloc(10, 1);
        MAT_AT(l_matrix, (size_t) label, 0) = 1.0;

        labels[i] = l_matrix;
    }

    fclose(f);

    return labels;
}

Mat *read_inputs(char *path, size_t N)
{
    FILE *f = fopen(path, "rb");
    size_t ret = 0;

    Mat *inputs = malloc(N*sizeof(Mat));

    // Read magic number
    int magic = 0;
    ret = fread(&magic, sizeof(int), 1, f);
    magic = swap_endian(magic);

    // Read number of images
    int num_items = 0;
    ret = fread(&num_items, sizeof(int), 1, f);
    num_items = swap_endian(num_items);

    // Read number of rows
    int rows = 0;
    ret = fread(&rows, sizeof(int), 1, f);
    rows = swap_endian(rows);

    // Read number of columns
    int cols = 0;
    ret = fread(&cols, sizeof(int), 1, f);
    cols = swap_endian(cols);

    // Read the images
    for (size_t i = 0; i < N && i < (size_t) num_items; i++) {
        Mat image = mat_alloc(rows, cols);
        for (size_t i = 0; i < (size_t) rows; i++) {
            for (size_t j = 0; j < (size_t) cols; j++) {
                unsigned char pixel = 0;
                ret = fread(&pixel, sizeof(char), 1, f);
                MAT_AT(image, i, j) = (double) pixel / 255.0; // Normalising the data improved accuracy a lot
            }
        }
        mat_flatten(&image);
        inputs[i] = image;
    }

    fclose(f);

    return inputs;
}

void free_data(Mat *data, size_t N)
{
    for (size_t i = 0; i < N; i++) {
        mat_free(data[i]);
    }
    free(data);
}

int mat_to_label(Mat m)
{
    int label = 0;
    double max = 0.0;

    for (size_t i = 0; i < m.rows; i++) {
        if (MAT_AT(m, i, 0) > max) {
            max = MAT_AT(m, i, 0);
            label = (int) i;
        }
    }

    return label;
}

int main()
{
    srand(time(NULL));

    size_t N = 60000;

    // Read the labels from the training set
    Mat *labels = read_labels("datasets/train-labels-idx1-ubyte/train-labels.idx1-ubyte", N);

    // Read the images from the training set
    Mat *images = read_inputs("datasets/train-images-idx3-ubyte/train-images.idx3-ubyte", N);

    // Create the network and train it
    double learning_rate = 0.01;
    size_t epochs = 20;
    size_t arch[] = { 28*28, 1000, 100, 10 };
    Network n = net_alloc(sizeof(arch)/sizeof(size_t), arch);
    for (size_t i = 0; i < epochs; i++) {
        for (size_t j = 0; j < N; j++) {
            net_train(n, images[j], labels[j], learning_rate);
            printf("\rEpoch %zu of %zu", i+1, epochs); fflush(stdout); // Print the progress
        }
    }



    size_t C = 10000;

    // Read the labels from the test set - these dont need to be matrices, but I'm not writing another function
    Mat *test_labels = read_labels("datasets/t10k-labels-idx1-ubyte", C);

    // Read the images from the test set
    Mat *test_images = read_inputs("datasets/t10k-images-idx3-ubyte", C);

    int correct_guesses = 0;
    for (size_t i = 0; i < C; i++) {
        // Set input
        mat_copy(NET_IN(n), test_images[i]);
        // Forward
        net_forward(n);
        // Compare output
        int guess = mat_to_label(NET_OUT(n));
        int answer = mat_to_label(test_labels[i]);
        if (guess == answer) {
            correct_guesses++;
        }
    }

    printf("\nThe network guessed correctly %d out of %zu times. With an accuracy of %.2f percent\n.",
           correct_guesses, C, (double) correct_guesses / (double) C * 100.0);

    // Save the weights and biases
    net_save(n, "weights_and_biases");
    

    // Free everything
    net_free(n);
    free_data(labels, N);
    free_data(images, N);
    free_data(test_labels, C);
    free_data(test_images, C);
    
    return 0;
}
