#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <random>

class BOLDLayerMNIST {
private:
    int num_inputs = 784; 
    int num_classes = 10; 
    
    std::vector<std::vector<bool>> weights;
    std::vector<std::vector<int>> weight_accumulators;
    
    // Lower threshold allows faster adaptation to correct the argmax errors
    int threshold = 64; 

    int compute_xnor_sum(const std::vector<bool>& inputs, const std::vector<bool>& w) const {
        int sum = 0;
        for (int i = 0; i < num_inputs; ++i) {
            bool xnor = !(inputs[i] ^ w[i]);
            sum += xnor ? 1 : -1;
        }
        return sum;
    }

public:
    BOLDLayerMNIST() {
        weights.resize(num_classes, std::vector<bool>(num_inputs, false));
        weight_accumulators.resize(num_classes, std::vector<int>(num_inputs, 0));

        // Break symmetry by initializing weights randomly (50% True / 50% False)
        std::mt19937 gen(42); 
        std::uniform_int_distribution<> dis(0, 1);
        
        for (int c = 0; c < num_classes; ++c) {
            for (int i = 0; i < num_inputs; ++i) {
                weights[c][i] = (dis(gen) == 1);
            }
        }
    }

    int predict(const std::vector<bool>& binarized_image) const {
        int best_class = 0;
        int max_sum = -9999;

        for (int c = 0; c < num_classes; ++c) {
            int current_sum = compute_xnor_sum(binarized_image, weights[c]);
            if (current_sum > max_sum) {
                max_sum = current_sum;
                best_class = c;
            }
        }
        return best_class;
    }

    // ---------------------------------------------------------
    // COMPETITIVE MULTI-CLASS UPDATE
    // ---------------------------------------------------------
    void train_step(const std::vector<bool>& binarized_image, int target_class) {
        // Find what the network currently guesses based on argmax
        int predicted_class = predict(binarized_image);
        
        // If the network is already correct, no Boolean variation/error exists. Skip.
        if (predicted_class == target_class) return; 

        // 1. Update Target Class: Push its weights to MATCH the input image (Increase score)
        for (int i = 0; i < num_inputs; ++i) {
            int push = binarized_image[i] ? 1 : -1;
            weight_accumulators[target_class][i] += push;

            if (weight_accumulators[target_class][i] >= threshold) {
                weights[target_class][i] = true;
                weight_accumulators[target_class][i] = threshold;
            } else if (weight_accumulators[target_class][i] <= -threshold) {
                weights[target_class][i] = false;
                weight_accumulators[target_class][i] = -threshold;
            }
        }

        // 2. Update Wrong Winner: Push its weights to MISMATCH the input image (Decrease score)
        for (int i = 0; i < num_inputs; ++i) {
            int push = binarized_image[i] ? -1 : 1;
            weight_accumulators[predicted_class][i] += push;

            if (weight_accumulators[predicted_class][i] >= threshold) {
                weights[predicted_class][i] = true;
                weight_accumulators[predicted_class][i] = threshold;
            } else if (weight_accumulators[predicted_class][i] <= -threshold) {
                weights[predicted_class][i] = false;
                weight_accumulators[predicted_class][i] = -threshold;
            }
        }
    }
};


// MNIST files are Big-Endian. This reverses the bytes for Little-Endian systems.
uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xff000000) |
           ((val <<  8) & 0x00ff0000) |
           ((val >>  8) & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}

// ---------------------------------------------------------
// DATA LOADERS
// ---------------------------------------------------------

// Reads MNIST label files and returns a vector of integers (0-9)
std::vector<int> load_mnist_labels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    uint32_t magic_number = 0, num_items = 0;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_items), 4);

    magic_number = swap_endian(magic_number);
    num_items = swap_endian(num_items);

    if (magic_number != 2049) throw std::runtime_error("Invalid MNIST label file magic number.");

    std::vector<int> labels(num_items);
    for (uint32_t i = 0; i < num_items; ++i) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i] = static_cast<int>(label);
    }
    return labels;
}

// Reads MNIST image files and binarizes them on the fly for the BOLD layer
std::vector<std::vector<bool>> load_mnist_images(const std::string& filename, uint8_t binarize_threshold = 127) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    uint32_t magic_number = 0, num_images = 0, rows = 0, cols = 0;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    magic_number = swap_endian(magic_number);
    num_images = swap_endian(num_images);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    if (magic_number != 2051) throw std::runtime_error("Invalid MNIST image file magic number.");

    int image_size = rows * cols; // Should be 784
    std::vector<std::vector<bool>> images(num_images, std::vector<bool>(image_size));

    for (uint32_t i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            uint8_t pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            // Binarization: True if pixel > threshold, False otherwise
            images[i][j] = (pixel > binarize_threshold);
        }
    }
    return images;
}

// ---------------------------------------------------------
// MAIN TRAINING LOOP
// ---------------------------------------------------------

int main() {
    try {
        std::cout << "Loading MNIST dataset..." << std::endl;

        // Replace these with the actual paths to your downloaded unzipped files
        std::string train_images_path = "C:/Users/cshep/MLinC/mnistdataset/train-images.idx3-ubyte";
        std::string train_labels_path = "C:/Users/cshep/MLinC/mnistdataset/trainlabels.idx1-ubyte";
        std::string test_images_path  = "C:/Users/cshep/MLinC/mnistdataset/testimages.idx3-ubyte";
        std::string test_labels_path  = "C:/Users/cshep/MLinC/mnistdataset/testlabels.idx1-ubyte";

        // Load and automatically binarize the data
        auto X_train = load_mnist_images(train_images_path);
        auto y_train = load_mnist_labels(train_labels_path);
        auto X_test  = load_mnist_images(test_images_path);
        auto y_test  = load_mnist_labels(test_labels_path);

        std::cout << "Loaded " << X_train.size() << " training images and " 
                  << X_test.size() << " test images." << std::endl;

        BOLDLayerMNIST network;
        int epochs = 7; // BOLD converges exceptionally fast due to exact discrete gradients
        int testsamples = X_test.size();
        int train_iters = X_train.size();
        std::cout << "\nStarting Native Boolean Training Loop..." << std::endl;

        for (int epoch = 1; epoch <= epochs; ++epoch) {
            std::cout << "--- Epoch " << epoch << " ---" << std::endl;

            // 1. Train Iteration
            for (size_t i = 0; i < train_iters; ++i) {
                network.train_step(X_train[i], y_train[i]);
                
                // Optional: Print progress every 10,000 images
                if ((i + 1) % train_iters == 0) {
                    std::cout << "  Processed " << (i + 1) << " images..." << std::endl;
                }
            }

            // 2. Evaluation Iteration
            int correct = 0;
            for (size_t i = 0; i < testsamples; ++i) {
                int prediction = network.predict(X_test[i]);
                if (prediction == y_test[i]) {
                    correct++;
                }
            }

            double accuracy = (static_cast<double>(correct) / testsamples) * 100.0;
            std::cout << "Epoch " << epoch << " Test Accuracy: " << accuracy << "% (" 
                      << correct << "/" << testsamples << ")\n\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\nCheck your file paths inside the quotes." << std::endl;
        return 1;
    }

    return 0;
}