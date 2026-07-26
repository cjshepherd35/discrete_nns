// qllm.cpp -- a quaternary (2-bit) LANGUAGE MODEL, built in the style of
// qnn2.cpp but doing the job of timedwikiv2karpathy.py: predict the next
// token on WikiText-2.
//
// WHY AN MLP AND NOT A TRANSFORMER
// --------------------------------
// qnn2.cpp is a feed-forward classifier over quaternary inputs: it maps a
// fixed-size vector of 2-bit values to one of K classes using integer
// accumulators and an alignment score. A transformer's attention/softmax/
// layernorm have no clean analog in this discrete-accumulator world. But
// Karpathy's neural LM reduces, at its core, to exactly what qnn2 already
// does once you add one missing piece: instead of classifying a 784-pixel
// image into 10 digits, you classify a *window of previous tokens* into one
// of `vocab_size` next-tokens. That is the Bengio / makemore-MLP recipe:
//
//     [ tok_{t-C} ... tok_{t-1} ] --embed--> quaternary vector --MLP--> tok_t
//
// So the whole model is qnn2's DeepQuatNetwork with a learnable quaternary
// EMBEDDING TABLE bolted onto the front as the (now trainable) input layer.
//
// THE EMBEDDING TABLE (the interesting part)
// ------------------------------------------
// Every token id needs to become a vector of quaternary digits (each 0..3)
// to feed the net. We store one row of `n_embed` quaternary values per token.
// Crucially the row is LEARNABLE using the exact same machinery qnn2 uses for
// a weight: each embedding cell is a 2-bit quantization of a latent integer
// accumulator living in [-threshold, +threshold]. Forward = look the row up.
// Backward = the graded error signal, after flowing back through the hidden
// layers to the input, tells each cell "you should have been higher / lower";
// we push its accumulator that way and re-quantize. qnn2 stopped backprop at
// layer 0 because its input (raw pixels) was fixed; here layer 0's input is
// the embedding lookup, so we let the signal reach it and the table learns.
//
// Position is encoded by CONCATENATION: the C context slots occupy disjoint
// input ranges, and the first hidden layer has independent weights per input
// index, so "token X in slot 0" and "token X in slot 3" act differently --
// the makemore trick, no separate positional table needed.
//
// DATASET: same as timedwikiv2karpathy.py -- WikiText-2 (Salesforce/wikitext,
// wikitext-2-v1) with a byte-level BPE tokenizer (vocab 1000). qnn2.cpp reads
// pre-exported MNIST idx files rather than downloading; by the same
// convention this reads a pre-exported "wikitext2_train.txt" (the train split
// joined with spaces, exactly as the Python does) and runs its own BPE on it.
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cctype>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <random>

// =========================================================================
// QUATERNARY (2-BIT) SCORING  -- identical semantics to qnn2.cpp
// =========================================================================
// Values 0..3. 0/1 are the "false side", 2/3 the "true side".
inline int qscore(int input, int weight) {
    bool input_hi = (input >= 2);
    bool weight_hi = (weight >= 2);
    if (input_hi == weight_hi) {
        return (input == weight && (input == 0 || input == 3)) ? 2 : 1;
    }
    return -std::abs(input - weight);
}

// Graded polarity of a value: 0 -> -3, 1 -> -1, 2 -> +1, 3 -> +3.
inline int qpolarity(int v) { return 2 * v - 3; }

// =========================================================================
// BPE TOKENIZER  -- a C++ port of the BPETokenizer in timedwikiv2karpathy.py
// =========================================================================
// Byte-level BPE. Same algorithm (merge the most frequent adjacent pair,
// repeat vocab_size-256 times) and the same GPT-2-ish pre-tokenization split,
// approximated over ASCII (\w = [A-Za-z0-9_]). Non-ASCII bytes fall into the
// "symbol run" bucket and are handled at the byte level, so any input is
// representable. Chunks are de-duplicated with multiplicities for speed --
// WikiText repeats words heavily, so training merges over unique chunks is far
// faster than the Python's per-occurrence loop while producing the same merges.
class BPETokenizer {
public:
    std::unordered_map<uint64_t, int> merges;   // (a<<32|b) -> new id
    std::vector<std::string> vocab;             // id -> raw bytes

    BPETokenizer() {
        vocab.resize(256);
        for (int i = 0; i < 256; ++i) vocab[i] = std::string(1, (char)i);
    }

    static uint64_t key(int a, int b) { return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; }

    // GPT-2-ish pre-tokenizer (ASCII approximation of the Python regex
    //   'contractions | ?\w+ | ?[^\s\w]+ | \s+ ).
    static std::vector<std::string> pretokenize(const std::string& text) {
        std::vector<std::string> chunks;
        size_t i = 0, n = text.size();
        auto is_word  = [](unsigned char c){ return std::isalnum(c) || c == '_'; };
        auto is_space = [](unsigned char c){ return std::isspace(c) != 0; };
        static const char* sfx[] = {"re", "ve", "ll", "s", "t", "m", "d"}; // longest-first within groups
        while (i < n) {
            unsigned char c = (unsigned char)text[i];
            // contractions: 's 't 're 've 'm 'll 'd
            if (c == '\'' && i + 1 < n) {
                bool matched = false;
                for (const char* s : sfx) {
                    size_t L = std::strlen(s);
                    if (i + 1 + L <= n && text.compare(i + 1, L, s) == 0) {
                        chunks.push_back(text.substr(i, 1 + L));
                        i += 1 + L; matched = true; break;
                    }
                }
                if (matched) continue;
            }
            size_t j = i;
            // optional single leading space then a word-run or a symbol-run
            if (c == ' ' && i + 1 < n) {
                unsigned char d = (unsigned char)text[i + 1];
                if (is_word(d)) {
                    j = i + 1; while (j < n && is_word((unsigned char)text[j])) ++j;
                    chunks.push_back(text.substr(i, j - i)); i = j; continue;
                }
                if (!is_space(d)) {
                    j = i + 1; while (j < n && !is_space((unsigned char)text[j]) && !is_word((unsigned char)text[j])) ++j;
                    chunks.push_back(text.substr(i, j - i)); i = j; continue;
                }
            }
            if (is_word(c)) {
                while (j < n && is_word((unsigned char)text[j])) ++j;
                chunks.push_back(text.substr(i, j - i)); i = j; continue;
            }
            if (!is_space(c)) {
                while (j < n && !is_space((unsigned char)text[j]) && !is_word((unsigned char)text[j])) ++j;
                chunks.push_back(text.substr(i, j - i)); i = j; continue;
            }
            while (j < n && is_space((unsigned char)text[j])) ++j;      // whitespace run
            chunks.push_back(text.substr(i, j - i)); i = j;
        }
        return chunks;
    }

    static std::vector<int> merge_seq(const std::vector<int>& ids, uint64_t pk, int idx) {
        std::vector<int> out;
        out.reserve(ids.size());
        size_t i = 0;
        while (i < ids.size()) {
            if (i + 1 < ids.size() && key(ids[i], ids[i + 1]) == pk) {
                out.push_back(idx); i += 2;
            } else {
                out.push_back(ids[i]); ++i;
            }
        }
        return out;
    }

    void train(const std::string& text, int vocab_size, bool verbose = false) {
        std::vector<std::string> chunks = pretokenize(text);
        // de-duplicate chunks, keep multiplicity
        std::unordered_map<std::string, long> counts;
        counts.reserve(chunks.size() * 2);
        for (auto& ch : chunks) counts[ch]++;

        std::vector<std::vector<int>> seqs;
        std::vector<long> mult;
        seqs.reserve(counts.size()); mult.reserve(counts.size());
        for (auto& kv : counts) {
            std::vector<int> ids(kv.first.size());
            for (size_t k = 0; k < kv.first.size(); ++k) ids[k] = (unsigned char)kv.first[k];
            seqs.push_back(std::move(ids));
            mult.push_back(kv.second);
        }

        int num_merges = vocab_size - 256;
        vocab.resize(256);
        for (int i = 0; i < 256; ++i) vocab[i] = std::string(1, (char)i);
        merges.clear();

        for (int m = 0; m < num_merges; ++m) {
            std::unordered_map<uint64_t, long> stats;
            stats.reserve(1 << 16);
            for (size_t s = 0; s < seqs.size(); ++s) {
                const std::vector<int>& v = seqs[s];
                long w = mult[s];
                for (size_t t = 0; t + 1 < v.size(); ++t) stats[key(v[t], v[t + 1])] += w;
            }
            if (stats.empty()) break;
            uint64_t best = 0; long bestc = -1;
            for (auto& kv : stats) {
                if (kv.second > bestc || (kv.second == bestc && kv.first < best)) {
                    bestc = kv.second; best = kv.first;
                }
            }
            int a = (int)(best >> 32), b = (int)(best & 0xffffffff);
            int idx = 256 + m;
            merges[best] = idx;
            vocab.push_back(vocab[a] + vocab[b]);
            for (auto& v : seqs) v = merge_seq(v, best, idx);
            if (verbose && (m + 1) % 100 == 0)
                std::cout << "  merge " << (m + 1) << "/" << num_merges
                          << ": (" << a << "," << b << ") -> " << idx << "\n";
        }
    }

    std::vector<int> encode_chunk(const std::string& chunk) const {
        std::vector<int> ids(chunk.size());
        for (size_t k = 0; k < chunk.size(); ++k) ids[k] = (unsigned char)chunk[k];
        while (ids.size() >= 2) {
            // find adjacent pair with the lowest merge id (earliest merge wins)
            uint64_t bestpair = 0; int bestrank = INT32_MAX;
            for (size_t t = 0; t + 1 < ids.size(); ++t) {
                auto it = merges.find(key(ids[t], ids[t + 1]));
                if (it != merges.end() && it->second < bestrank) {
                    bestrank = it->second; bestpair = key(ids[t], ids[t + 1]);
                }
            }
            if (bestrank == INT32_MAX) break;
            ids = merge_seq(ids, bestpair, bestrank);
        }
        return ids;
    }

    // Encode a whole document, caching per unique chunk.
    std::vector<int> encode(const std::string& text) const {
        std::vector<std::string> chunks = pretokenize(text);
        std::unordered_map<std::string, std::vector<int>> cache;
        std::vector<int> out;
        out.reserve(text.size() / 3 + 16);
        for (auto& ch : chunks) {
            auto it = cache.find(ch);
            if (it == cache.end()) it = cache.emplace(ch, encode_chunk(ch)).first;
            out.insert(out.end(), it->second.begin(), it->second.end());
        }
        return out;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string s;
        for (int id : ids) if (id >= 0 && id < (int)vocab.size()) s += vocab[id];
        return s;
    }
    int vocab_size() const { return (int)vocab.size(); }
};

// =========================================================================
// QuatLayer -- carried over from qnn2.cpp essentially unchanged.
// =========================================================================
class QuatLayer {
private:
    int num_inputs;
    int num_outputs;
    int threshold;
    std::vector<std::vector<uint8_t>> weights;
    std::vector<std::vector<int>> accumulators;
    mutable std::vector<int> run_mean;
    mutable std::vector<int> run_dev;

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
        std::uniform_int_distribution<> dis(-thresh, thresh);
        for (int o = 0; o < num_outputs; ++o)
            for (int i = 0; i < num_inputs; ++i) {
                accumulators[o][i] = dis(gen);
                weights[o][i] = quantize(accumulators[o][i]);
            }
    }

    std::vector<uint8_t> forward(const std::vector<uint8_t>& inputs) const {
        std::vector<uint8_t> outputs(num_outputs);
        for (int o = 0; o < num_outputs; ++o) {
            int sum = 0;
            for (int i = 0; i < num_inputs; ++i) sum += qscore(inputs[i], weights[o][i]);
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
        for (int o = 0; o < num_outputs; ++o)
            for (int i = 0; i < num_inputs; ++i)
                scores[o] += qscore(inputs[i], weights[o][i]);
        return scores;
    }

    int get_weight(int o, int i) const { return weights[o][i]; }
    int get_num_inputs() const { return num_inputs; }
    int get_num_outputs() const { return num_outputs; }

    void update_weights(int o, const std::vector<uint8_t>& inputs, bool match) {
        for (int i = 0; i < num_inputs; ++i) {
            int w = weights[o][i];
            int in = inputs[i];
            int push = std::abs(qscore(in, w)) * ((in >= 2) ? 1 : -1);
            if (!match) push = -push;
            accumulators[o][i] += push;
            if (accumulators[o][i] > threshold) accumulators[o][i] = threshold;
            else if (accumulators[o][i] < -threshold) accumulators[o][i] = -threshold;
            weights[o][i] = quantize(accumulators[o][i]);
        }
    }
};

// =========================================================================
// EmbeddingTable -- the learnable quaternary input layer.
// =========================================================================
// One row of `dim` quaternary digits per token, each a 2-bit quantization of a
// latent accumulator in [-threshold, threshold] -- the same representation
// QuatLayer uses for a weight. Forward is a lookup; the row for a context is
// built by concatenating the rows of its tokens (position encoded by slot).
class EmbeddingTable {
private:
    int vocab;
    int dim;
    int threshold;
    std::vector<std::vector<uint8_t>> emb;         // [vocab][dim] quaternary
    std::vector<std::vector<int>> accumulators;    // [vocab][dim] latent

    uint8_t quantize(int acc) const {
        if (acc >= threshold / 2)  return 3;
        if (acc >= 0)              return 2;
        if (acc >= -threshold / 2) return 1;
        return 0;
    }

public:
    EmbeddingTable(int vocab_size, int embed_dim, int thresh, std::mt19937& gen)
        : vocab(vocab_size), dim(embed_dim), threshold(thresh) {
        emb.resize(vocab, std::vector<uint8_t>(dim));
        accumulators.resize(vocab, std::vector<int>(dim, 0));
        std::uniform_int_distribution<> dis(-thresh, thresh);
        for (int v = 0; v < vocab; ++v)
            for (int d = 0; d < dim; ++d) {
                accumulators[v][d] = dis(gen);
                emb[v][d] = quantize(accumulators[v][d]);
            }
    }

    int embed_dim() const { return dim; }

    // Concatenate the rows of the context tokens into one quaternary vector.
    std::vector<uint8_t> lookup_context(const std::vector<int>& ctx) const {
        std::vector<uint8_t> out;
        out.reserve(ctx.size() * dim);
        for (int tok : ctx)
            for (int d = 0; d < dim; ++d) out.push_back(emb[tok][d]);
        return out;
    }

    // Backward: `signal` is the graded per-input desirability that flowed back
    // to the concatenated embedding vector. Slot s (dim d) at index s*dim+d
    // belongs to token ctx[s]. Where the signal is meaningfully positive we
    // want that cell higher, so push its accumulator toward +threshold, and
    // toward -threshold where it is negative -- a pure gradient-sign step, gated
    // like qnn2's hidden updates so weak/noisy signals are ignored. Re-quantize.
    void update(const std::vector<int>& ctx, const std::vector<int>& signal) {
        long abs_sum = 0;
        for (int s : signal) abs_sum += std::abs(s);
        long cutoff = std::max(1L, abs_sum / (long)std::max<size_t>(1, signal.size()));
        for (size_t s = 0; s < ctx.size(); ++s) {
            int tok = ctx[s];
            for (int d = 0; d < dim; ++d) {
                int sig = signal[s * dim + d];
                if (std::abs(sig) < cutoff) continue;      // magnitude gate
                int push = (sig > 0) ? 2 : -2;             // fixed-size step
                accumulators[tok][d] += push;
                if (accumulators[tok][d] > threshold) accumulators[tok][d] = threshold;
                else if (accumulators[tok][d] < -threshold) accumulators[tok][d] = -threshold;
                emb[tok][d] = quantize(accumulators[tok][d]);
            }
        }
    }
};

// =========================================================================
// QuatLM -- the full model: embedding table -> hidden QuatLayers -> vocab head
// =========================================================================
class QuatLM {
private:
    int context_len;
    EmbeddingTable embedding;
    std::vector<QuatLayer> hidden_layers;
    QuatLayer output_layer;

public:
    static std::mt19937 global_gen;

    QuatLM(int vocab_size, int context, int n_embed, int hidden_dim,
           int num_hidden_layers, int thresh)
        : context_len(context),
          embedding(vocab_size, n_embed, thresh, global_gen),
          hidden_layers(),
          output_layer(hidden_dim, vocab_size, thresh, global_gen) {
        if (num_hidden_layers < 1) num_hidden_layers = 1;
        int layer_input_dim = context * n_embed;
        hidden_layers.reserve(num_hidden_layers);
        for (int i = 0; i < num_hidden_layers; ++i) {
            hidden_layers.emplace_back(layer_input_dim, hidden_dim, thresh, global_gen);
            layer_input_dim = hidden_dim;
        }
    }

    int context_length() const { return context_len; }

    std::vector<int> logits(const std::vector<int>& ctx) const {
        std::vector<uint8_t> act = embedding.lookup_context(ctx);
        for (const auto& layer : hidden_layers) act = layer.forward(act);
        return output_layer.compute_scores(act);
    }

    int predict(const std::vector<int>& ctx) const {
        std::vector<int> scores = logits(ctx);
        int best = 0, mx = -2147483647;
        for (size_t c = 0; c < scores.size(); ++c)
            if (scores[c] > mx) { mx = scores[c]; best = (int)c; }
        return best;
    }

    void train_step(const std::vector<int>& ctx, int target) {
        // 1. Forward, keeping every activation (embedding output + each hidden).
        std::vector<uint8_t> emb_act = embedding.lookup_context(ctx);
        std::vector<std::vector<uint8_t>> hidden_activations;
        hidden_activations.reserve(hidden_layers.size());
        std::vector<uint8_t> current_act = emb_act;
        for (const auto& layer : hidden_layers) {
            current_act = layer.forward(current_act);
            hidden_activations.push_back(current_act);
        }

        std::vector<int> scores = output_layer.compute_scores(current_act);
        int predicted = 0, mx = -2147483647;
        for (size_t c = 0; c < scores.size(); ++c)
            if (scores[c] > mx) { mx = scores[c]; predicted = (int)c; }
        if (predicted == target) return;

        // 2. Graded hidden signal from the output weights BEFORE they update
        //    (the qnn2 ordering): target's desire minus wrong winner's desire.
        std::vector<int> next_layer_signal(hidden_layers.back().get_num_outputs(), 0);
        for (int h = 0; h < hidden_layers.back().get_num_outputs(); ++h)
            next_layer_signal[h] = output_layer.get_weight(target, h)
                                 - output_layer.get_weight(predicted, h);

        // 3. Output head update.
        output_layer.update_weights(target, current_act, true);
        output_layer.update_weights(predicted, current_act, false);

        // 4. Backprop the graded signal through the hidden layers, then into
        //    the embedding table (qnn2 stopped at layer 0; we continue).
        for (int li = (int)hidden_layers.size() - 1; li >= 0; --li) {
            QuatLayer& layer = hidden_layers[li];
            const std::vector<uint8_t>& prev_act = (li == 0) ? emb_act : hidden_activations[li - 1];

            int n_out = layer.get_num_outputs();
            long mean_sig = 0;
            for (int h = 0; h < n_out; ++h) mean_sig += next_layer_signal[h];
            mean_sig /= n_out;
            long abs_sum = 0;
            for (int h = 0; h < n_out; ++h) {
                next_layer_signal[h] -= (int)mean_sig;
                abs_sum += std::abs(next_layer_signal[h]);
            }
            long cutoff = std::max(1L, abs_sum / n_out);

            const std::vector<uint8_t>& own_act = hidden_activations[li];
            for (int h = 0; h < n_out; ++h) {
                if (own_act[h] == 0 || own_act[h] == 3) continue;         // saturation gate
                if (std::abs(next_layer_signal[h]) < cutoff) continue;    // magnitude gate
                if (next_layer_signal[h] > 0) layer.update_weights(h, prev_act, true);
                else                          layer.update_weights(h, prev_act, false);
            }

            // Propagate one step further back through this layer's own weights.
            std::vector<int> prev_layer_signal(layer.get_num_inputs(), 0);
            for (int h = 0; h < n_out; ++h) {
                int signal = next_layer_signal[h];
                if (signal == 0) continue;
                if (own_act[h] == 0 || own_act[h] == 3) continue;
                for (int i = 0; i < layer.get_num_inputs(); ++i)
                    prev_layer_signal[i] += signal * qpolarity(layer.get_weight(h, i));
            }

            if (li == 0) {
                // prev_layer_signal is now the signal on the concatenated
                // embedding vector -> update the embedding table.
                embedding.update(ctx, prev_layer_signal);
                break;
            }
            next_layer_signal = std::move(prev_layer_signal);
        }
    }
};

std::mt19937 QuatLM::global_gen(1337);

// =========================================================================
// MAIN
// =========================================================================
static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char* argv[]) {
    try {
        // ---- hyperparameters (scaled for discrete integer training) ----
        const int vocab_size = 1000;   // same target vocab as the Python BPE
        const int context_len = 16;    // previous tokens fed to the MLP
        const int n_embed = 16;        // quaternary digits per token
        const int hidden_dim = 128;
        const int num_hidden_layers = 2;
        const int threshold = 256;
        long train_iters = 400000;     // random-position samples (argv[1] overrides)
        if (argc > 1) train_iters = std::max(1L, std::atol(argv[1]));

        std::string text_path = "wikitext2_train.txt";
        if (argc > 2) text_path = argv[2];

        std::cout << "Loading WikiText-2 text from " << text_path << " ...\n";
        std::string text = read_file(text_path);
        std::cout << "  " << text.size() << " chars.\n";

        std::cout << "Training BPE tokenizer (vocab " << vocab_size << ") ...\n";
        BPETokenizer tok;
        tok.train(text, vocab_size, true);
        std::cout << "  vocab size " << tok.vocab_size() << "\n";

        std::cout << "Encoding dataset ...\n";
        std::vector<int> data = tok.encode(text);
        std::cout << "  " << data.size() << " tokens.\n";

        // 90/10 split, same as the Python.
        size_t n = (size_t)(0.9 * data.size());
        std::vector<int> train_data(data.begin(), data.begin() + n);
        std::vector<int> test_data(data.begin() + n, data.end());
        std::cout << "  train tokens " << train_data.size()
                  << ", test tokens " << test_data.size() << "\n";

        QuatLM model(vocab_size, context_len, n_embed, hidden_dim,
                     num_hidden_layers, threshold);

        std::mt19937 sampler(1337);
        std::uniform_int_distribution<size_t> train_pos(
            context_len, train_data.size() - 1);
        std::uniform_int_distribution<size_t> test_pos(
            context_len, test_data.size() - 1);

        auto next_token_accuracy = [&](const std::vector<int>& d, int samples) {
            std::uniform_int_distribution<size_t> pos(context_len, d.size() - 1);
            int correct = 0;
            std::vector<int> ctx(context_len);
            for (int s = 0; s < samples; ++s) {
                size_t i = pos(sampler);
                for (int c = 0; c < context_len; ++c) ctx[c] = d[i - context_len + c];
                if (model.predict(ctx) == d[i]) ++correct;
            }
            return 100.0 * correct / samples;
        };

        std::cout << "\nStarting Quaternary LM training (" << train_iters
                  << " iters) ...\n";
        const long eval_interval = std::max(1L, train_iters / 10);
        std::vector<int> ctx(context_len);
        for (long it = 0; it < train_iters; ++it) {
            if (it % eval_interval == 0) {
                double tr = next_token_accuracy(train_data, 4000);
                double te = next_token_accuracy(test_data, 4000);
                std::cout << "  iter " << it
                          << "  train next-token acc " << tr << "%"
                          << "  test " << te << "%\n";
            }
            size_t i = train_pos(sampler);
            for (int c = 0; c < context_len; ++c) ctx[c] = train_data[i - context_len + c];
            model.train_step(ctx, train_data[i]);
        }

        double final_tr = next_token_accuracy(train_data, 10000);
        double final_te = next_token_accuracy(test_data, 10000);
        std::cout << "\nFinal next-token accuracy  train " << final_tr
                  << "%  test " << final_te << "%\n";

        // ---- generation, like the Python's model.generate ----
        // Sample from the integer scores with a softmax-with-temperature so the
        // output isn't a deterministic argmax loop.
        std::cout << "\n--- Sample generation ---\n";
        std::vector<int> generated;
        std::vector<int> gctx(context_len, 0);   // seed with token 0, like the Python
        const int max_new_tokens = 300;
        const double temperature = 1.0;
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        for (int step = 0; step < max_new_tokens; ++step) {
            std::vector<int> sc = model.logits(gctx);
            int mx = *std::max_element(sc.begin(), sc.end());
            double Z = 0.0;
            std::vector<double> probs(sc.size());
            for (size_t c = 0; c < sc.size(); ++c) {
                probs[c] = std::exp((sc[c] - mx) / temperature);
                Z += probs[c];
            }
            double r = uni(sampler) * Z, acc = 0.0;
            int nxt = (int)sc.size() - 1;
            for (size_t c = 0; c < probs.size(); ++c) {
                acc += probs[c];
                if (acc >= r) { nxt = (int)c; break; }
            }
            generated.push_back(nxt);
            for (int c = 0; c < context_len - 1; ++c) gctx[c] = gctx[c + 1];
            gctx[context_len - 1] = nxt;
        }
        std::cout << tok.decode(generated) << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
