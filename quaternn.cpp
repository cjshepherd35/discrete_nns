#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <random>

// ---------------------------------------------------------
// QUATERNARY (2-BIT) SCORING
//
// Values 0..3. 0 and 1 are the "false side", 2 and 3 the "true side".
//   - Same side:      +1, or +2 for an exact match at an extreme (0-0, 3-3)
//   - Opposite sides: minus the distance (1-2 -> -1, 0-2/1-3 -> -2, 0-3 -> -3)
// ---------------------------------------------------------
inline int qscore(int input, int weight) {
    bool input_hi = (input >= 2);
    bool weight_hi = (weight >= 2);
    if (input_hi == weight_hi) {
        return (input == weight && (input == 0 || input == 3)) ? 2 : 1;
    }
    return -std::abs(input - weight);
}

// Graded influence of a weight on the score direction: how strongly this
// weight pulls the neuron toward firing when the input is high.
//   0 -> -3, 1 -> -1, 2 -> +1, 3 -> +3
inline int qpolarity(int weight) {
    return 2 * weight - 3;
}

class QuatLayer {
private:
    int num_inputs;
    int num_outputs;
    int threshold;      // accumulator clip bound, same role as in multilayerbnn.cpp

    std::vector<std::vector<uint8_t>> weights;      // values 0..3
    std::vector<std::vector<int>> accumulators;

    // Per-neuron integer running statistics (a batch-norm analog). The
    // quaternary scoring table is negatively biased (penalties reach -3 but
    // rewards cap at +2), so raw sums sit far below zero and fixed
    // zero-centered activation bands collapse to all-0 outputs. Each neuron
    // therefore tracks a running mean of its own sum (and a running mean
    // absolute deviation for the band width) and fires based on how far the
    // current sum deviates from ITS OWN typical value.
    mutable std::vector<int> run_mean;
    mutable std::vector<int> run_dev;

    // The weight is a 2-bit quantization of the latent accumulator, which
    // lives in [-threshold, +threshold] exactly as in multilayerbnn.cpp.
    // Binary read off the sign; quaternary reads off the quarter-band:
    //   [ T/2,  T]   -> 3
    //   [ 0,    T/2) -> 2
    //   [-T/2,  0)   -> 1
    //   [-T,   -T/2) -> 0
    uint8_t quantize(int acc) const {
        if (acc >= threshold / 2)  return 3;
        if (acc >= 0)              return 2;
        if (acc >= -threshold / 2) return 1;
        return 0;
    }

public:
    QuatLayer(int inputs, int outputs, int thresh, std::mt19937& gen)
        : num_inputs(inputs), num_outputs(outputs), threshold(thresh),
          run_mean(outputs, 0), run_dev(outputs, std::max(1, inputs / 8)) {

        weights.resize(num_outputs, std::vector<uint8_t>(num_inputs));
        accumulators.resize(num_outputs, std::vector<int>(num_inputs, 0));

        // Break symmetry: random latent accumulator, weight derived from it
        std::uniform_int_distribution<> dis(-thresh, thresh);
        for (int o = 0; o < num_outputs; ++o) {
            for (int i = 0; i < num_inputs; ++i) {
                accumulators[o][i] = dis(gen);
                weights[o][i] = quantize(accumulators[o][i]);
            }
        }
    }

    // Forward: sum the quaternary alignment scores, center by the neuron's
    // running mean, and quantize the deviation into a quaternary activation:
    //   c >= dev   -> 3
    //   c >= 0     -> 2
    //   c >= -dev  -> 1
    //   otherwise  -> 0
    // The running stats update on every forward call (integer EMA, 1/64 rate).
    std::vector<uint8_t> forward(const std::vector<uint8_t>& inputs) const {
        std::vector<uint8_t> outputs(num_outputs);
        for (int o = 0; o < num_outputs; ++o) {
            int sum = 0;
            for (int i = 0; i < num_inputs; ++i) {
                sum += qscore(inputs[i], weights[o][i]);
            }
            run_mean[o] += (sum - run_mean[o]) / 64;
            int c = sum - run_mean[o];
            run_dev[o] += (std::abs(c) - run_dev[o]) / 64;
            int d = std::max(1, run_dev[o]);

            if (c >= d)       outputs[o] = 3;
            else if (c >= 0)  outputs[o] = 2;
            else if (c >= -d) outputs[o] = 1;
            else              outputs[o] = 0;
        }
        return outputs;
    }

    std::vector<int> compute_scores(const std::vector<uint8_t>& inputs) const {
        std::vector<int> scores(num_outputs, 0);
        for (int o = 0; o < num_outputs; ++o) {
            for (int i = 0; i < num_inputs; ++i) {
                scores[o] += qscore(inputs[i], weights[o][i]);
            }
        }
        return scores;
    }

    int get_weight(int o, int i) const { return weights[o][i]; }
    int get_num_inputs() const { return num_inputs; }
    int get_num_outputs() const { return num_outputs; }

    // Same accumulator dynamics as multilayerbnn.cpp: clip at +/-threshold,
    // never reset. The weight is re-derived from the accumulator's band after
    // every push. Push magnitude mirrors the scoring table (|qscore|), and the
    // direction is the input's side: toward +threshold for inputs 2/3, toward
    // -threshold for inputs 0/1.
    //   match=true  (target class): extreme match reinforces by 2, same-side
    //                by 1, and a k-apart mismatch corrects by k (up to 3).
    //   match=false (wrong winner): the exact opposite push.
    void update_weights(int o, const std::vector<uint8_t>& inputs, bool match) {
        for (int i = 0; i < num_inputs; ++i) {
            int w = weights[o][i];
            int in = inputs[i];

            int push = std::abs(qscore(in, w)) * ((in >= 2) ? 1 : -1);
            if (!match) push = -push;

            accumulators[o][i] += push;

            if (accumulators[o][i] > threshold) {
                accumulators[o][i] = threshold;
            } else if (accumulators[o][i] < -threshold) {
                accumulators[o][i] = -threshold;
            }

            weights[o][i] = quantize(accumulators[o][i]);
        }
    }
};

class DeepQuatNetwork {
private:
    std::vector<QuatLayer> hidden_layers;
    QuatLayer output_layer;

public:
    DeepQuatNetwork(int input_dim = 784, int hidden_dim = 128, int num_hidden_layers = 1, int output_dim = 10, int thresh = 64)
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

    int predict(const std::vector<uint8_t>& image) const {
        std::vector<uint8_t> activation = image;
        for (const auto& layer : hidden_layers) {
            activation = layer.forward(activation);
        }
        std::vector<int> scores = output_layer.compute_scores(activation);

        int best_class = 0;
        int max_score = -99999999;
        for (size_t c = 0; c < scores.size(); ++c) {
            if (scores[c] > max_score) {
                max_score = scores[c];
                best_class = c;
            }
        }
        return best_class;
    }

    void train_step(const std::vector<uint8_t>& image, int target_class) {
        // 1. Forward pass through all hidden layers and store activations.
        std::vector<std::vector<uint8_t>> hidden_activations;
        hidden_activations.reserve(hidden_layers.size());

        std::vector<uint8_t> current_act = image;
        for (const auto& layer : hidden_layers) {
            current_act = layer.forward(current_act);
            hidden_activations.push_back(current_act);
        }

        std::vector<int> scores = output_layer.compute_scores(current_act);
        int predicted_class = 0;
        int max_score = -99999999;
        for (size_t c = 0; c < scores.size(); ++c) {
            if (scores[c] > max_score) {
                max_score = scores[c];
                predicted_class = c;
            }
        }

        if (predicted_class == target_class) return;

        // 2. Build the graded hidden-layer signal from the output weights
        // BEFORE they are updated. A hidden unit's signal is the difference
        // between how much the target class wants it high and how much the
        // wrongly predicted class wants it high; weight polarity is graded
        // (0 -> -3, 1 -> -1, 2 -> +1, 3 -> +3) instead of binary +/-1.
        std::vector<int> next_layer_signal(hidden_layers.back().get_num_outputs(), 0);
        for (int h = 0; h < hidden_layers.back().get_num_outputs(); ++h) {
            next_layer_signal[h] = output_layer.get_weight(target_class, h)
                                 - output_layer.get_weight(predicted_class, h);
        }

        // 3. Output layer updates based on the last hidden activation.
        output_layer.update_weights(target_class, current_act, true);
        output_layer.update_weights(predicted_class, current_act, false);

        // 4. Backpropagate the graded signals through the hidden layers.
        for (int layer_index = (int)hidden_layers.size() - 1; layer_index >= 0; --layer_index) {
            QuatLayer& current_layer = hidden_layers[layer_index];
            const std::vector<uint8_t>& prev_act = (layer_index == 0) ? image : hidden_activations[layer_index - 1];

            // Zero-center the signal across the layer to remove systematic
            // bias; without this, mismatch updates dominate and the weights
            // run away to one extreme (verified experimentally: layer-0
            // weights saturate ~90% at value 3 and activations collapse).
            int n_out = current_layer.get_num_outputs();
            long mean_sig = 0;
            for (int h = 0; h < n_out; ++h) mean_sig += next_layer_signal[h];
            mean_sig /= n_out;
            long abs_sum = 0;
            for (int h = 0; h < n_out; ++h) {
                next_layer_signal[h] -= (int)mean_sig;
                abs_sum += std::abs(next_layer_signal[h]);
            }
            long cutoff = std::max(1L, abs_sum / n_out);

            const std::vector<uint8_t>& own_act = hidden_activations[layer_index];
            for (int h = 0; h < n_out; ++h) {
                // Saturation gate - the activation-derivative part of the
                // Boolean chain rule. Units in the extreme bands (0 or 3) have
                // no Boolean variation: a small change in their sum cannot
                // change their output, so they receive no weight update. Only
                // units near the center boundary (activations 1 and 2) learn.
                // This single gate is the difference between ~14% and ~76%
                // test accuracy at 3 hidden layers.
                if (own_act[h] == 0 || own_act[h] == 3) continue;
                // Magnitude gate: only above-average |signal| units update,
                // so weak/noisy signals don't swamp the strong ones.
                if (std::abs(next_layer_signal[h]) < cutoff) continue;
                if (next_layer_signal[h] > 0) {
                    current_layer.update_weights(h, prev_act, true);
                } else {
                    current_layer.update_weights(h, prev_act, false);
                }
            }

            if (layer_index == 0) {
                break;
            }

            // Propagate one layer further back through current_layer's OWN
            // weights, sized by its input count.
            std::vector<int> prev_layer_signal(current_layer.get_num_inputs(), 0);
            for (int h = 0; h < n_out; ++h) {
                int signal = next_layer_signal[h];
                if (signal == 0) continue;
                // Saturated units also block signal flow to earlier layers
                if (own_act[h] == 0 || own_act[h] == 3) continue;

                for (int i = 0; i < current_layer.get_num_inputs(); ++i) {
                    prev_layer_signal[i] += signal * qpolarity(current_layer.get_weight(h, i));
                }
            }

            next_layer_signal = std::move(prev_layer_signal);
        }
    }
};

std::mt19937 DeepQuatNetwork::global_gen(42);

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

// Reads MNIST image files and quantizes each pixel to a quaternary value:
// 0-63 -> 0, 64-127 -> 1, 128-191 -> 2, 192-255 -> 3
std::vector<std::vector<uint8_t>> load_mnist_images(const std::string& filename) {
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
    std::vector<std::vector<uint8_t>> images(num_images, std::vector<uint8_t>(image_size));

    for (uint32_t i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            uint8_t pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            images[i][j] = static_cast<uint8_t>(pixel >> 6);
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

        // Load and automatically quantize the data to quaternary values
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

        // Accumulator threshold 256: at 64 the graded pushes (up to +/-3 per
        // sample) let weights churn past the optimum and accuracy collapses
        // after epoch 1 (71% -> 55%); 128 reaches ~86%, 256 ~89% and stable.
        DeepQuatNetwork network(784, 128, num_hidden_layers, 10, 256);
        int epochs = 4;
        size_t testsamples = X_test.size();
        size_t train_iters = X_train.size();
        std::cout << "\nStarting Quaternary Training Loop..." << std::endl;

        for (int epoch = 1; epoch <= epochs; ++epoch) {
            std::cout << "--- Epoch " << epoch << " ---" << std::endl;

            // 1. Train Iteration
            for (size_t i = 0; i < train_iters; ++i) {
                network.train_step(X_train[i], y_train[i]);
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
