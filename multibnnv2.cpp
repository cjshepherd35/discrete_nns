#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <random>


// class BoldLayer {
// private:
//     int num_inputs;
//     int num_outputs;
//     int threshold;
    
//     std::vector<std::vector<bool>> weights;
//     std::vector<std::vector<int>> accumulators;

// public:
//     BoldLayer(int inputs, int outputs, int thresh, std::mt19937& gen)
//         : num_inputs(inputs), num_outputs(outputs), threshold(thresh) {
        
//         weights.resize(num_outputs, std::vector<bool>(num_inputs));
//         accumulators.resize(num_outputs, std::vector<int>(num_inputs, 0));
        
//         std::uniform_int_distribution<> dis(0, 1);
//         for (int o = 0; o < num_outputs; ++o) {
//             for (int i = 0; i < num_inputs; ++i) {
//                 weights[o][i] = (dis(gen) == 1);
//             }
//         }
//     }

//     // Forward Step: Evaluates pure Boolean activation based on XNOR-Popcount logic
//     std::vector<bool> forward(const std::vector<bool>& inputs) const {
//         std::vector<bool> outputs(num_outputs);
//         for (int o = 0; o < num_outputs; ++o) {
//             int popcount_sum = 0;
//             for (int i = 0; i < num_inputs; ++i) {
//                 bool xnor = !(inputs[i] ^ weights[o][i]);
//                 popcount_sum += xnor ? 1 : -1;
//             }
//             outputs[o] = (popcount_sum >= 0);
//         }
//         return outputs;
//     }

//     // Computes raw scalar matching score (used strictly by the output layer for argmax classification)
//     std::vector<int> compute_scores(const std::vector<bool>& inputs) const {
//         std::vector<int> scores(num_outputs, 0);
//         for (int o = 0; o < num_outputs; ++o) {
//             for (int i = 0; i < num_inputs; ++i) {
//                 scores[o] += (!(inputs[i] ^ weights[o][i])) ? 1 : -1;
//             }
//         }
//         return scores;
//     }

//     // Natively computes the Boolean Variation (Derivative) with respect to a given input index.
//     // Returns true if flipping input 'i' would flip the state of output 'o'.
//     bool compute_boolean_variation(int o, int i, const std::vector<bool>& inputs) const {
//         int base_sum = 0;
//         for (int idx = 0; idx < num_inputs; ++idx) {
//             base_sum += (!(inputs[idx] ^ weights[o][idx])) ? 1 : -1;
//         }
//         bool base_out = (base_sum >= 0);

//         int mutated_sum = base_sum;
//         bool original_xnor = !(inputs[i] ^ weights[o][i]);
//         mutated_sum -= original_xnor ? 1 : -1; 
//         mutated_sum += (!original_xnor) ? 1 : -1; 
//         bool mutated_out = (mutated_sum >= 0);

//         return (base_out ^ mutated_out);
//     }

//     // Native Boolean weight update applying accumulator clipping
//     void update_weights(int o, const std::vector<bool>& inputs, bool target_state_match) {
//         for (int i = 0; i < num_inputs; ++i) {
//             int push = (inputs[i] == target_state_match) ? 1 : -1;
//             accumulators[o][i] += push;

//             if (accumulators[o][i] >= threshold) {
//                 weights[o][i] = true;
//                 accumulators[o][i] = threshold;
//             } else if (accumulators[o][i] <= -threshold) {
//                 weights[o][i] = false;
//                 accumulators[o][i] = -threshold;
//             }
//         }
//     }

//     bool get_weight(int o, int i) const { return weights[o][i]; }
//     int get_num_inputs() const { return num_inputs; }
//     int get_num_outputs() const { return num_outputs; }
// };

// class DeepBoldNetwork {
// private:
//     std::vector<BoldLayer> hidden_layers;
//     BoldLayer output_layer;

// public:
//     DeepBoldNetwork(int input_dim = 784, const std::vector<int>& hidden_dims = {128, 64}, int output_dim = 10, int thresh = 16)
//         : output_layer(hidden_dims.empty() ? input_dim : hidden_dims.back(), output_dim, thresh, global_gen) {
        
//         int current_input_dim = input_dim;
//         for (int dim : hidden_dims) {
//             hidden_layers.emplace_back(current_input_dim, dim, thresh, global_gen);
//             current_input_dim = dim;
//         }
//     }

//     static std::mt19937 global_gen;

//     int predict(const std::vector<bool>& image) const {
//         std::vector<bool> current_act = image;
//         for (const auto& layer : hidden_layers) {
//             current_act = layer.forward(current_act);
//         }
//         std::vector<int> scores = output_layer.compute_scores(current_act);
        
//         int best_class = 0;
//         int max_score = -99999;
//         for (size_t c = 0; c < scores.size(); ++c) {
//             if (scores[c] > max_score) {
//                 max_score = scores[c];
//                 best_class = c;
//             }
//         }
//         return best_class;
//     }

//     // The Deep B^O_LD Backpropagation Pipeline using Deep Boolean Chain Rule
//     void train_step(const std::vector<bool>& image, int target_class) {
//         // 1. Forward Pass: Track activations at each layer boundary
//         std::vector<std::vector<bool>> activations;
//         activations.push_back(image); // activations[0] is the input layer
        
//         for (size_t i = 0; i < hidden_layers.size(); ++i) {
//             activations.push_back(hidden_layers[i].forward(activations.back()));
//         }
        
//         // 2. Optimize Prediction: Compute scores directly from cached final hidden layer
//         std::vector<int> scores = output_layer.compute_scores(activations.back());
//         int predicted_class = 0;
//         int max_score = -99999;
//         for (size_t c = 0; c < scores.size(); ++c) {
//             if (scores[c] > max_score) {
//                 max_score = scores[c];
//                 predicted_class = c;
//             }
//         }
        
//         if (predicted_class == target_class) return; // Accurate classification; skip updates.

//         // 3. Output Layer Error Assessment
//         output_layer.update_weights(target_class, activations.back(), true);
//         output_layer.update_weights(predicted_class, activations.back(), false);

//         // 4. Deep Backward Pass: Backpropagating structural variance masks
//         std::vector<std::vector<bool>> target_masks(hidden_layers.size());
//         std::vector<std::vector<bool>> predicted_masks(hidden_layers.size());

//         // Handle the final hidden layer directly connected to the output layer
//         int last_idx = static_cast<int>(hidden_layers.size()) - 1;
//         if (last_idx < 0) return; // Guard clause against 0 hidden layers

//         target_masks[last_idx].resize(hidden_layers[last_idx].get_num_outputs(), false);
//         predicted_masks[last_idx].resize(hidden_layers[last_idx].get_num_outputs(), false);

//         for (int h = 0; h < hidden_layers[last_idx].get_num_outputs(); ++h) {
//             target_masks[last_idx][h] = output_layer.compute_boolean_variation(target_class, h, activations[last_idx + 1]);
//             predicted_masks[last_idx][h] = output_layer.compute_boolean_variation(predicted_class, h, activations[last_idx + 1]);
//         }

//         // Deep propagation through preceding hidden layers
//         for (int i = last_idx - 1; i >= 0; --i) {
//             target_masks[i].resize(hidden_layers[i].get_num_outputs(), false);
//             predicted_masks[i].resize(hidden_layers[i].get_num_outputs(), false);

//             for (int j = 0; j < hidden_layers[i].get_num_outputs(); ++j) {
//                 // Target propagation
//                 for (int h = 0; h < hidden_layers[i + 1].get_num_outputs(); ++h) {
//                     if (target_masks[i + 1][h] && hidden_layers[i + 1].compute_boolean_variation(h, j, activations[i + 1])) {
//                         target_masks[i][j] = true;
//                         break; 
//                     }
//                 }
//                 // Prediction propagation
//                 for (int h = 0; h < hidden_layers[i + 1].get_num_outputs(); ++h) {
//                     if (predicted_masks[i + 1][h] && hidden_layers[i + 1].compute_boolean_variation(h, j, activations[i + 1])) {
//                         predicted_masks[i][j] = true;
//                         break;
//                     }
//                 }
//             }
//         }

//         // 5. Update Hidden Weights using exact matching against parent execution contexts
//         for (size_t i = 0; i < hidden_layers.size(); ++i) {
//             for (int h = 0; h < hidden_layers[i].get_num_outputs(); ++h) {
//                 if (target_masks[i][h] && !predicted_masks[i][h]) {
//                     bool preferred_state = !activations[i + 1][h]; 
//                     hidden_layers[i].update_weights(h, activations[i], preferred_state);
//                 }
//                 else if (!target_masks[i][h] && predicted_masks[i][h]) {
//                     bool preferred_state = !activations[i + 1][h];
//                     hidden_layers[i].update_weights(h, activations[i], preferred_state);
//                 }
//             }
//         }
//     }
// };




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

    // Forward Step: Evaluates pure Boolean activation based on XNOR-Popcount logic
    std::vector<bool> forward(const std::vector<bool>& inputs) const {
        std::vector<bool> outputs(num_outputs);
        for (int o = 0; o < num_outputs; ++o) {
            int popcount_sum = 0;
            for (int i = 0; i < num_inputs; ++i) {
                bool xnor = !(inputs[i] ^ weights[o][i]);
                popcount_sum += xnor ? 1 : -1;
            }
            outputs[o] = (popcount_sum >= 0);
        }
        return outputs;
    }

    // Computes raw scalar matching score (used strictly by the output layer for argmax classification)
    std::vector<int> compute_scores(const std::vector<bool>& inputs) const {
        std::vector<int> scores(num_outputs, 0);
        for (int o = 0; o < num_outputs; ++o) {
            for (int i = 0; i < num_inputs; ++i) {
                scores[o] += (!(inputs[i] ^ weights[o][i])) ? 1 : -1;
            }
        }
        return scores;
    }

    // Natively computes the Boolean Variation (Derivative) with respect to a given input index.
    // Returns true if flipping input 'i' would flip the state of output 'o'.
    bool compute_boolean_variation(int o, int i, const std::vector<bool>& inputs) const {
        // Evaluate native activation
        int base_sum = 0;
        for (int idx = 0; idx < num_inputs; ++idx) {
            base_sum += (!(inputs[idx] ^ weights[o][idx])) ? 1 : -1;
        }
        bool base_out = (base_sum >= 0);

        // Evaluate activation with mutated/flipped target bit (Discrete calculus alternative to dx)
        int mutated_sum = base_sum;
        bool original_xnor = !(inputs[i] ^ weights[o][i]);
        mutated_sum -= original_xnor ? 1 : -1; // Remove original impact
        mutated_sum += (!original_xnor) ? 1 : -1; // Inject inverted impact
        bool mutated_out = (mutated_sum >= 0);

        // Boolean derivative rule: delta = f(x ^ 1) ^ f(x)
        return (base_out ^ mutated_out);
    }

    // Native Boolean weight update applying accumulator clipping
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

    bool get_weight(int o, int i) const { return weights[o][i]; }
    int get_num_inputs() const { return num_inputs; }
    int get_num_outputs() const { return num_outputs; }
};

class DeepBoldNetwork {
private:
    BoldLayer hidden;
    BoldLayer output_layer;

public:
    DeepBoldNetwork(int input_dim = 784, int hidden_dim = 128, int output_dim = 10, int thresh = 16)
        : hidden(input_dim, hidden_dim, thresh, global_gen),
          output_layer(hidden_dim, output_dim, thresh, global_gen) {}

    static std::mt19937 global_gen;

    int predict(const std::vector<bool>& image) const {
        std::vector<bool> h_act = hidden.forward(image);
        std::vector<int> scores = output_layer.compute_scores(h_act);
        
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

    // // The B^O_LD Backpropagation Pipeline using the Boolean Chain Rule
    // void train_step(const std::vector<bool>& image, int target_class) {
    //     // 1. Forward Pass
    //     std::vector<bool> h_act = hidden.forward(image);
    //     std::vector<int> scores = output_layer.compute_scores(h_act);
        
    //     int predicted_class = 0;
    //     int max_score = -99999;
    //     for (size_t c = 0; c < scores.size(); ++c) {
    //         if (scores[c] > max_score) {
    //             max_score = scores[c];
    //             predicted_class = c;
    //         }
    //     }
    //     if (predicted_class == target_class) return; // Accurate classification; no logical variance required.

    //     // 2. Output Layer Error Assessment
    //     // Correct node wants to align with activations; erroneous node wants to invert
    //     output_layer.update_weights(target_class, h_act, true);
    //     output_layer.update_weights(predicted_class, h_act, false);

    //     // 3. Backward Pass: Applying the Boolean Chain Rule
    //     // We compute standard error vectors and pass them back through the calculated variation bitmask.
    //     std::vector<bool> target_error_mask(hidden.get_num_outputs(), false);
    //     std::vector<bool> predicted_error_mask(hidden.get_num_outputs(), false);

    //     for (int h = 0; h < hidden.get_num_outputs(); ++h) {
    //         // Compute structural variation: Does changing hidden unit 'h' affect the output layer nodes?
    //         target_error_mask[h] = output_layer.compute_boolean_variation(target_class, h, h_act);
    //         predicted_error_mask[h] = output_layer.compute_boolean_variation(predicted_class, h, h_act);
    //     }

    //     // 4. Update the Hidden Layer using exact discrete variants
    //     for (int h = 0; h < hidden.get_num_outputs(); ++h) {
    //         // The paper implies optimization along the shortest Hamming distance path.
    //         // If flipping hidden unit 'h' directly helps the correct class while not aiding the error class:
    //         if (target_error_mask[h] && !predicted_error_mask[h]) {
    //             // Determine what target boolean state the hidden neuron *should* have emitted
    //             bool preferred_state = !h_act[h]; 
    //             hidden.update_weights(h, image, preferred_state);
    //         }
    //         // If changing hidden unit 'h' hurts the incorrect winner while keeping the target safe:
    //         else if (!target_error_mask[h] && predicted_error_mask[h]) {
    //             bool preferred_state = !h_act[h];
    //             hidden.update_weights(h, image, preferred_state);
    //         }
    //     }
    // }

    // Score-sensitivity training step.
    //
    // The previous version derived the hidden signal from compute_boolean_variation
    // against each output node's thresholded state (popcount >= 0). Flipping one
    // hidden unit only moves an output score by +/-2, so that variation is nonzero
    // only when a score sits within 2 of zero -- which almost never happens with
    // 128 inputs. The hidden layer was therefore gradient-starved and stayed at its
    // random initialization. Classification is also argmax over raw scores, not the
    // >=0 threshold, so the relevant derivative is d(score_target - score_pred)/dh,
    // which reads directly off the two output weight rows.
    void train_step(const std::vector<bool>& image, int target_class) {
        // 1. Single forward pass; classify from the raw scores.
        std::vector<bool> h_act = hidden.forward(image);
        std::vector<int> scores = output_layer.compute_scores(h_act);

        int predicted_class = 0;
        int max_score = -99999;
        for (size_t c = 0; c < scores.size(); ++c) {
            if (scores[c] > max_score) {
                max_score = scores[c];
                predicted_class = c;
            }
        }
        if (predicted_class == target_class) return;

        // 2. Hidden updates, using the output weights *before* they are updated.
        //    h_act[h] = true adds (w[c][h] ? +1 : -1) to class c's score, so a unit
        //    helps the margin when it agrees with the target row and disagrees with
        //    the predicted row. Only units whose current output disagrees with the
        //    desired state are touched, to avoid churning the whole layer.
        for (int h = 0; h < hidden.get_num_outputs(); ++h) {
            bool wants_target = output_layer.get_weight(target_class, h);
            bool wants_pred = output_layer.get_weight(predicted_class, h);

            if (wants_target && !wants_pred && !h_act[h]) {
                hidden.update_weights(h, image, true);
            } else if (!wants_target && wants_pred && h_act[h]) {
                hidden.update_weights(h, image, false);
            }
        }

        // 3. Output layer: pull the target row toward the hidden activation,
        //    push the wrongly-predicted row away from it.
        output_layer.update_weights(target_class, h_act, true);
        output_layer.update_weights(predicted_class, h_act, false);
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

        
        DeepBoldNetwork network(784, 128, 10, 128);
        //     std::vector<int> hidden_topology = {128, 64, 32};
        // DeepBoldNetwork network(784, hidden_topology, 10, 16);
    
        int epochs = 5; // BOLD converges exceptionally fast due to exact discrete gradients
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