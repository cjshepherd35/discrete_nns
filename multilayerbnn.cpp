
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <random>
#include <cmath>



class BoldLayer {
private:
    int num_inputs;
    int num_outputs;
    int threshold;
    
    std::vector<std::vector<bool>> weights;
    std::vector<std::vector<int>> accumulators;

public:
    BoldLayer(int inputs, int outputs, int thresh, std::mt19937& gen)
        : num_inputs(inputs), num_outputs(outputs), threshold(thresh) {
        
        weights.resize(num_outputs, std::vector<bool>(num_inputs));
        accumulators.resize(num_outputs, std::vector<int>(num_inputs, 0));
        
        std::uniform_int_distribution<> dis(0, 1);
        for (int o = 0; o < num_outputs; ++o) {
            for (int i = 0; i < num_inputs; ++i) {
                weights[o][i] = (dis(gen) == 1);
            }
        }
    }

    std::vector<bool> forward(const std::vector<bool>& inputs) const {
        std::vector<bool> outputs(num_outputs);
        for (int o = 0; o < num_outputs; ++o) {
            int popcount_sum = 0;
            for (int i = 0; i < num_inputs; ++i) {
                popcount_sum += (!(inputs[i] ^ weights[o][i])) ? 1 : -1;
            }
            outputs[o] = (popcount_sum >= 0);
        }
        return outputs;
    }

    std::vector<int> compute_scores(const std::vector<bool>& inputs) const {
        std::vector<int> scores(num_outputs, 0);
        for (int o = 0; o < num_outputs; ++o) {
            for (int i = 0; i < num_inputs; ++i) {
                scores[o] += (!(inputs[i] ^ weights[o][i])) ? 1 : -1;
            }
        }
        return scores;
    }

    // Returns the raw weight profile matching backprop rules
    bool get_weight(int o, int i) const { return weights[o][i]; }
    int get_num_inputs() const { return num_inputs; }
    int get_num_outputs() const { return num_outputs; }

    // Direct, un-attenuated weight updates to keep convergence clean
    void update_weights(int o, const std::vector<bool>& inputs, bool target_state_match) {
        for (int i = 0; i < num_inputs; ++i) {
            int push = (inputs[i] == target_state_match) ? 1 : -1;
            accumulators[o][i] += push;

            if (accumulators[o][i] >= threshold) {
                weights[o][i] = true;
                accumulators[o][i] = threshold;
            } else if (accumulators[o][i] <= -threshold) {
                weights[o][i] = false;
                accumulators[o][i] = -threshold;
            }
        }
    }
};

class DeepBoldNetwork {
private:
    std::vector<BoldLayer> hidden_layers;
    BoldLayer output_layer;

public:
    DeepBoldNetwork(int input_dim = 784, int hidden_dim = 128, int num_hidden_layers = 1, int output_dim = 10, int thresh = 128)
        : hidden_layers(),
          output_layer(hidden_dim, output_dim, thresh, global_gen) {
        if (num_hidden_layers < 1) {
            num_hidden_layers = 1;
        }

        int layer_input_dim = input_dim;
        hidden_layers.reserve(num_hidden_layers);
        for (int i = 0; i < num_hidden_layers; ++i) {
            hidden_layers.emplace_back(layer_input_dim, hidden_dim, thresh, global_gen);
            layer_input_dim = hidden_dim;
        }
    }

    static std::mt19937 global_gen;

    int predict(const std::vector<bool>& image) const {
        std::vector<bool> activation = image;
        for (const auto& layer : hidden_layers) {
            activation = layer.forward(activation);
        }
        std::vector<int> scores = output_layer.compute_scores(activation);

        int best_class = 0;
        int max_score = -99999;
        for (size_t c = 0; c < scores.size(); ++c) {
            if (scores[c] > max_score) {
                max_score = scores[c];
                best_class = c;
            }
        }
        return best_class;
    }

    void train_step(const std::vector<bool>& image, int target_class) {
        // 1. Forward pass through all hidden layers and store activations.
        std::vector<std::vector<bool>> hidden_activations;
        hidden_activations.reserve(hidden_layers.size());

        std::vector<bool> current_act = image;
        for (const auto& layer : hidden_layers) {
            current_act = layer.forward(current_act);
            hidden_activations.push_back(current_act);
        }

        std::vector<int> scores = output_layer.compute_scores(current_act);
        int predicted_class = 0;
        int max_score = -99999;
        for (size_t c = 0; c < scores.size(); ++c) {
            if (scores[c] > max_score) {
                max_score = scores[c];
                predicted_class = c;
            }
        }

        if (predicted_class == target_class) return;

        // 2. Build the hidden-layer signal from the output weights *before* updating them.
        std::vector<int> next_layer_signal(hidden_layers.back().get_num_outputs(), 0);
        for (int h = 0; h < hidden_layers.back().get_num_outputs(); ++h) {
            bool wants_target = output_layer.get_weight(target_class, h);
            bool wants_pred = output_layer.get_weight(predicted_class, h);
            if (wants_target && !wants_pred) {
                next_layer_signal[h] = 1;
            } else if (!wants_target && wants_pred) {
                next_layer_signal[h] = -1;
            }
        }

        // 3. Output layer updates based on the last hidden activation.
        output_layer.update_weights(target_class, current_act, true);
        output_layer.update_weights(predicted_class, current_act, false);

        // The top-layer signal is exactly +/-1; propagated signals are sums, so lower
        // layers only act when the sum clears a noise floor of ~2 standard deviations.
        int sig_threshold = 1;

        for (int layer_index = (int)hidden_layers.size() - 1; layer_index >= 0; --layer_index) {
            BoldLayer& current_layer = hidden_layers[layer_index];
            const std::vector<bool>& prev_act = (layer_index == 0) ? image : hidden_activations[layer_index - 1];

            // Only touch units whose current output disagrees with the signal; units that
            // already agree are left alone to avoid churning the whole layer on every error.
            const std::vector<bool>& layer_act = hidden_activations[layer_index];
            for (int h = 0; h < current_layer.get_num_outputs(); ++h) {
                if (next_layer_signal[h] >= sig_threshold && !layer_act[h]) {
                    current_layer.update_weights(h, prev_act, true);
                } else if (next_layer_signal[h] <= -sig_threshold && layer_act[h]) {
                    current_layer.update_weights(h, prev_act, false);
                }
            }

            if (layer_index == 0) {
                break;
            }

            // Push the signal through this layer's weights to score its inputs
            // (which are the outputs of the layer below).
            std::vector<int> current_layer_signal(current_layer.get_num_inputs(), 0);
            int total_magnitude = 0;
            for (int h = 0; h < current_layer.get_num_outputs(); ++h) {
                int signal = next_layer_signal[h];
                if (signal == 0) continue;
                total_magnitude += std::abs(signal);

                for (int in = 0; in < current_layer.get_num_inputs(); ++in) {
                    bool weight = current_layer.get_weight(h, in);
                    current_layer_signal[in] += signal * (weight ? 1 : -1);
                }
            }

            sig_threshold = std::max(1, (int)(2.5 * std::sqrt((double)total_magnitude)));
            next_layer_signal = std::move(current_layer_signal);
        }
    }
};

std::mt19937 DeepBoldNetwork::global_gen(42);

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

int main(int argc, char* argv[]) {
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

        int num_hidden_layers = 3;
        if (argc > 1) {
            num_hidden_layers = std::max(1, std::atoi(argv[1]));
        }
        std::cout << "Using " << num_hidden_layers << " hidden layer(s)." << std::endl;

        DeepBoldNetwork network(784, 128, num_hidden_layers, 10, 128);
        int epochs = 4; // BOLD converges exceptionally fast due to exact discrete gradients
        size_t testsamples = X_test.size();
        size_t train_iters = X_train.size();
        std::cout << "\nStarting Native Boolean Training Loop..." << std::endl;

        for (int epoch = 1; epoch <= epochs; ++epoch) {
            std::cout << "--- Epoch " << epoch << " ---" << std::endl;

            // 1. Train Iteration
            for (size_t i = 0; i < train_iters; ++i) {
                network.train_step(X_train[i], y_train[i]);
                
                // Optional: Print progress every 10,000 images
                if ((i + 1) % 10000 == 0) {
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
