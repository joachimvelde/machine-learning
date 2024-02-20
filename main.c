#include "ml.h"
#include "raylib.h"

// These functions are just copied from train.c - Make a header file instead
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

int mat_to_label(Mat m, double *confidence)
{
    int label = 0;
    double max = 0.0;

    for (size_t i = 0; i < m.rows; i++) {
        if (MAT_AT(m, i, 0) > max) {
            max = MAT_AT(m, i, 0);
            label = (int) i;
        }
    }

    *confidence = max;
    return label;
}

// Downscale to 28x28
Mat downscale(Mat m)
{
    Mat d = mat_alloc(28, 28);

    for (int i = 0; i < 280; i += 10) {
        for (int j = 0; j < 280; j += 10) {
            double total = 0;

            for (int x = i; x < i + 10; x++) {
                for (int y = j; y < j + 10; y++) {
                    total += MAT_AT(m, x, y);
                }
            }

            MAT_AT(d, i/10, j/10) = total/100 / 255.0; // Normalise as well
        }
    }

    return d;
}

void classify_drawing(Network n, Mat image)
{
    Mat input = downscale(image);
    mat_flatten(&input);
    mat_copy(NET_IN(n), input);
    net_forward(n);

    double confidence = 0.0;
    int guess = mat_to_label(NET_OUT(n), &confidence);

    printf("The neural network guessed: %d, with a confidence of %.2f percent.\n", guess, confidence * 100.0);

    mat_free(input);
}

int main()
{
    const size_t WIDTH = 280;
    const size_t HEIGHT = 280;


    // Initialize the network
    size_t arch[] = { 28*28, 1000, 100, 10 };
    Network n = net_alloc(sizeof(arch)/sizeof(size_t), arch);

    // Load the weights and biases
    net_load(n, "weights_and_biases");

    Mat frame = mat_alloc(WIDTH, HEIGHT);


    // Create a window for the user to draw on
    InitWindow(WIDTH, HEIGHT, "Digit classifier");
    
    RenderTexture2D target = LoadRenderTexture(WIDTH, HEIGHT);

    // High for smoother lines while drawing
    SetTargetFPS(240);

    while (!WindowShouldClose())
    {
        Vector2 mouse = GetMousePosition();
        
        BeginTextureMode(target);

        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            ClearBackground(BLACK);
        }

        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            mouse.y = HEIGHT - mouse.y;
            DrawCircleV(mouse, 10, WHITE);
        }

        EndTextureMode();

        if (IsKeyPressed(KEY_SPACE)) {
            Image image = LoadImageFromTexture(target.texture);

            // Read the pixels of the current frame into the frame matrix
            for (size_t y = 0; y < HEIGHT; y++) {
                for (size_t x = 0; x < WIDTH; x++) {
                    Color color = GetImageColor(image, x, y);
                    int grayscale = (int) (0.3 * color.r + 0.59 * color.g + 0.11 * color.b);
                    MAT_AT(frame, y, x) = (double) grayscale;
                }
            }

            UnloadImage(image);

            classify_drawing(n, frame);
        }


        BeginDrawing();
            ClearBackground(BLACK);
            DrawTexture(target.texture, 0, 0, WHITE);
        EndDrawing();
    }

    UnloadRenderTexture(target);
    CloseWindow();

    net_free(n);
    mat_free(frame);

    return 0;
}
