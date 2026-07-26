// qllm_attn_xf.cpp -- INTERLEAVED transformer-style architecture.
//
// The original design ran ALL attention blocks first, THEN a single separate
// per-position MLP head. This file interleaves them: the model is a stack of
// transformer BLOCKS, each = causal self-attention + a per-position feed-forward
// MLP (E->ff->E), like a normal transformer (minus residuals -- the discrete
// quaternary reps have no add). After the block stack, a plain LM head projects
// E->vocab. So the compute path is attn/MLP/attn/MLP/.../head instead of
// attn/attn/.../MLP/MLP/head.
//
// Keeps the project's best recipe: the head-only ASYMMETRIC negative flip-
// threshold (K=1; the LM head's single push-away against the top wrong class
// accumulates at ACC_SCALE/N, everything else -- target pull, every block's
// attention + MLP, embedding -- full strength). x16 fixed-point accumulators.
//
// BASELINE run: 2 blocks, each with a 2-layer FFN. n_embed 64, N=4.
//
//   argv[3] = num transformer blocks (default 2)
//   argv[4] = N, LM-head neg-threshold multiplier   (default 4 = winner)
//   argv[5] = n_embed                               (default 64; mult of n_head)
//   argv[6] = FFN layers per block                  (default 2 => 1:2 attn:ffn;
//                                                    1 => 1:1 attn:ffn)
//   argv[7] = sampler seed                          (default 1337; eval stays 4242)
//   argv[8] = context length                        (default 16)
//   argv[9]  = base flip threshold                   (default 512)
//   argv[10] = threshold-ramp iters                  (0 = off; effective thr
//              ramps threshold/boost_max -> threshold over these iters)
//   argv[11] = boost_max                             (default 8)
//   argv[12] = positions trained per step            (default 4 of context_len)
//
// ===== original qllm_attn.cpp header follows =====
//
// qllm_attn.cpp -- qllm.cpp + a trainable QUATERNARY SELF-ATTENTION block.
//
// Same quaternary language model as qllm.cpp (embedding table -> MLP -> next
// token on WikiText-2), but a causal self-attention block is inserted between
// the embedding lookup and the MLP head. Each context token's embedding is
// projected to quaternary Q/K/V, the "dot products" are qnn2's qscore
// accumulation, and the mixed (context-aware) embeddings then flow into the
// existing hidden layers. Kept separate from qllm.cpp on purpose so the two
// can be A/B-compared: does discrete attention beat the plain concat MLP?
//
// PIPELINE  (per-position, stacked)
//   ctx (C tokens)
//     -> embedding.lookup_rows           C x E quaternary  (per-token rows)
//     -> [ attention block ] x num_blocks  C x E each      (deep context mixing)
//     -> per-position: attn_out[t] (dim E) -> hidden QuatLayers -> vocab head
//        predicts token t+1  (a shared head runs at every position)
//
// BACKWARD (the part the value projection needed)
//   The graded error flows head -> hidden layers -> the flattened attention
//   output. From there attention.backward:
//     1. back through out_proj  (a QuatLayer, standard qnn2 backprop),
//     2. into the VALUE projections: the signal on output position t is spread
//        to the value of every key j it attended to, weighted by the stored
//        attention weight wgt[t][j], then v_proj.update_weights pushes those
//        value neurons -- the exact accumulator push qnn2 uses,
//     3. on through v_proj's weights (via qpolarity) to the embedding rows,
//        so the embedding table keeps learning.
//     4. the Q/K ROUTER learns via a Hebbian surrogate (there is no softmax
//        derivative): for each key a query attended to, if attending more would
//        have helped the output, Q[t] and K[j] are nudged to agree (raise their
//        qscore); if it would have hurt, they are nudged apart -- same
//        accumulator push, gated to above-average routes. So the model learns
//        *what* to attend to, not just how to mix a fixed random pattern.
//
// DATASET: WikiText-2, read from pre-exported wikitext2_train.txt (see qllm.cpp).
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
#include <climits>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <random>

// =========================================================================
// QUATERNARY (2-BIT) SCORING  -- identical to qnn2.cpp
// =========================================================================
inline int qscore(int input, int weight) {
    bool input_hi = (input >= 2);
    bool weight_hi = (weight >= 2);
    if (input_hi == weight_hi) {
        return (input == weight && (input == 0 || input == 3)) ? 2 : 1;
    }
    return -std::abs(input - weight);
}
inline int qpolarity(int v) { return 2 * v - 3; }
inline uint8_t quantize_band(int c, int d) {
    if (c >= d)   return 3;
    if (c >= 0)   return 2;
    if (c >= -d)  return 1;
    return 0;
}

// QuatLayer accumulators run in x384 fixed point so the head's negative push can
// be scaled down (ACC_SCALE/N) losslessly => an N x higher effective negative
// flip-threshold. ACC_SCALE = full-strength push scale (== the positive push).
// 384 (= 2^7 * 3) so ACC_SCALE/N is EXACT for N up to 128 (incl. 32, 64): with
// smaller ACC_SCALE the integer division collapsed high N together (e.g. at 48,
// 48/32==48/64==1). Any N that divides the scale evenly is unchanged vs before,
// so all earlier N=4/8/16 results are bit-identical; only high N become genuine.
static const int ACC_SCALE = 384;

// Optional THRESHOLD-RAMP schedule: raising the base threshold mid-run would move
// the quantize() boundaries and mass-reinterpret every weight (a re-quantization
// shock), so instead we scale EVERY accumulator push by a global boost that
// decays over time. boost > 1 = pushes bigger = weights flip on less evidence =
// LOWER effective threshold (fast, plastic); boost -> 1 = base threshold (stable).
// Applied to all QuatLayer + embedding updates, so the positive/negative ratio N
// is preserved throughout. g_boost_num/g_boost_den = boost (default 1/1 = off).
static int g_boost_num = 1;
static int g_boost_den = 1;

// PER-NEURON TRUST REGION (argv[14]). The bang-bang collapse at large n_embed is
// a CORRELATED sqrt(d) overshoot: many of a neuron's d incoming weights flip the
// same way in one step, so its pre-activation s_o = sum_i qscore(in_i, w_i) jumps
// by ~alpha*d while the level-flip scale is only ~run_dev[o] ~ sigma*sqrt(d) --
// the overshoot ratio grows as sqrt(d). This caps the NET change to s_o per step
// to g_trust_frac * run_dev[o] "noise units": pending level-flips are committed
// most-confident-first until the cap is hit; the rest stay pending (no evidence
// lost) and can win the budget on a later step. Kills the overshoot WITHOUT
// stiffening per-weight plasticity (unlike raising threshold). 0 = OFF, and the
// update path is then bit-identical to the reference.
static double g_trust_frac = 0.0;

// ---- TWO ANTI-RAILING MECHANISMS for the wide-model burn-in (argv[17], [18]) ----
// Telemetry localized the n128 collapse to the attn out_proj + FFN accumulators
// progressively PINNING at +/-threshold (railing). Two fixes to test:
//  (A) LEAK (g_leak_div): each update, bleed every accumulator toward 0 by
//      acc -= acc/g_leak_div. Un-pins railed weights before they trap. Latent-
//      weight decay. 0 = off. Applies to QuatLayer only (embedding does not rail).
//  (B) RESET-ON-FLIP (g_reset_mode): integrate-and-fire. The accumulator measures
//      EVIDENCE toward the next single-notch move: when it reaches +/-threshold,
//      move the weight one level and RESET the accumulator to 0. It never lives
//      outside (-threshold, +threshold), so it CANNOT rail by construction. This
//      replaces the absolute-position quantize() mapping for the layer weights.
static int g_leak_div   = 0;
static int g_reset_mode = 0;
// RESET fire point = threshold / g_reset_fire_div (1 = full rail, 2 = flip boundary
// T/2 = more plastic). CLAMP cap: accumulators clamp at g_clamp_pct% of threshold
// (100 = default/off). Lowering it caps burial depth (anti-rail) with NO mid-range
// drag -- weights below the cap are untouched, unlike the proportional leak.
static int g_reset_fire_div = 1;
static int g_clamp_pct      = 100;
static int g_satgate        = 0;
// MULTI-WINDOW MINIBATCH (argv[22]=g_batch_B). Accumulate the pushes from B
// INDEPENDENT context windows into the accumulators with the WEIGHTS FROZEN
// (g_defer_commit skips re-quantization during the batch), then commit once. All
// B windows' gradients are computed at the same weights and summed, so noise from
// independent windows averages out (true variance reduction) and boundary weights
// stop flip-flopping -- unlike positions_per_step, which adds CORRELATED pushes
// from one window. B=1 = original per-window training (bit-identical).
static int  g_batch_B      = 1;
static bool g_defer_commit = false;
// SATURATION-GATE mode (argv[21]). The update gates skip a unit when its output
// activation is railed at 0 or 3. 0 = original (skip always); 1 = one-sided (skip
// only if the push would DEEPEN saturation, allow pushes that relieve it -- un-
// freezes railed units); 2 = open (never skip for saturation). `up` = push raises
// output (match flag). Returns true if this update should be SKIPPED.
static inline bool satgate_skip(int act, bool up) {
    if (act != 0 && act != 3) return false;               // not saturated: never skip
    if (g_satgate == 2) return false;                     // open
    if (g_satgate == 1) return (act == 3 && up) || (act == 0 && !up);  // deepen-only
    return true;                                          // original: always skip
}

// ---- PER-COMPONENT TELEMETRY (argv[15]; READ-ONLY, does NOT affect dynamics) --
// Localizes the large-n_embed collapse. For each component, over each eval
// window, we log three numbers:
//   flip = level-changing weights / weights touched      (how much is churning)
//   coh  = |sum d_qscore| / sum |d_qscore|  in [0,1]     (1 = fully CORRELATED
//          flips = the cross-unit cascade; ~0 = random churn)
//   sat  = touched accumulators pinned at +/-threshold / touched  (rail/burn-in)
// Each update call site sets g_tele_tag to its component; the update methods
// accumulate into these counters, which tele_dump() prints and resets per eval.
enum { T_EMB = 0, T_Q, T_K, T_V, T_OUT, T_FFN, T_HEAD, T_NCOMP };
static const char* g_tele_names[T_NCOMP] = {"emb","q","k","v","out","ffn","head"};
static long long g_tele_touch[T_NCOMP] = {0};
static long long g_tele_flip [T_NCOMP] = {0};
static long long g_tele_dsig [T_NCOMP] = {0};
static long long g_tele_dabs [T_NCOMP] = {0};
static long long g_tele_sat  [T_NCOMP] = {0};
static bool g_tele_on  = false;
static int  g_tele_tag = -1;
static void tele_dump(long iter) {
    std::cout << "    [tele it" << iter << "]";
    for (int t = 0; t < T_NCOMP; ++t) {
        double fr  = g_tele_touch[t] ? (double)g_tele_flip[t] / g_tele_touch[t] : 0.0;
        double coh = g_tele_dabs[t]  ? (double)std::llabs(g_tele_dsig[t]) / g_tele_dabs[t] : 0.0;
        double sat = g_tele_touch[t] ? (double)g_tele_sat[t]  / g_tele_touch[t] : 0.0;
        std::cout << "  " << g_tele_names[t]
                  << " f" << fr << " c" << coh << " s" << sat;
    }
    std::cout << "\n";
    for (int t = 0; t < T_NCOMP; ++t)
        g_tele_touch[t] = g_tele_flip[t] = g_tele_dsig[t] = g_tele_dabs[t] = g_tele_sat[t] = 0;
}

// WEIGHT INITIALIZATION mode. Normally weights start random across all four
// levels {0,1,2,3}; the extremes (0,3) have strong polarity and add init noise.
// A quieter, lower-noise start may help a big model bootstrap. Applies to the
// LAYER weights only (attention / FFN / head) -- the embedding always inits
// random so token rows stay distinguishable. init_draw returns a value in the
// ORIGINAL threshold range (each layer scales it as usual); its quantized level:
// draw>=thresh/2 ->3, >=0 ->2, >=-thresh/2 ->1, else 0.
//   0 = random {0,1,2,3}   (default)
//   1 = two middle values {1,2}  (diverse but no strong extremes)
//   2 = one value: all layer weights -> level 2
static int g_init_mode = 0;
static int init_draw(int thresh, std::mt19937& gen){
    if (g_init_mode == 2) return 0;                       // -> level 2 everywhere
    int hi = (g_init_mode == 1) ? std::max(1, thresh/2 - 1) : thresh;
    std::uniform_int_distribution<> d(-hi, hi);
    return d(gen);
}

// =========================================================================
// BPE TOKENIZER  -- a C++ port of BPETokenizer in timedwikiv2karpathy.py
// (identical to qllm.cpp)
// =========================================================================
class BPETokenizer {
public:
    std::unordered_map<uint64_t, int> merges;
    std::vector<std::string> vocab;

    BPETokenizer() {
        vocab.resize(256);
        for (int i = 0; i < 256; ++i) vocab[i] = std::string(1, (char)i);
    }
    static uint64_t key(int a, int b) { return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; }

    static std::vector<std::string> pretokenize(const std::string& text) {
        std::vector<std::string> chunks;
        size_t i = 0, n = text.size();
        auto is_word  = [](unsigned char c){ return std::isalnum(c) || c == '_'; };
        auto is_space = [](unsigned char c){ return std::isspace(c) != 0; };
        static const char* sfx[] = {"re", "ve", "ll", "s", "t", "m", "d"};
        while (i < n) {
            unsigned char c = (unsigned char)text[i];
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
            while (j < n && is_space((unsigned char)text[j])) ++j;
            chunks.push_back(text.substr(i, j - i)); i = j;
        }
        return chunks;
    }

    static std::vector<int> merge_seq(const std::vector<int>& ids, uint64_t pk, int idx) {
        std::vector<int> out; out.reserve(ids.size());
        size_t i = 0;
        while (i < ids.size()) {
            if (i + 1 < ids.size() && key(ids[i], ids[i + 1]) == pk) { out.push_back(idx); i += 2; }
            else { out.push_back(ids[i]); ++i; }
        }
        return out;
    }

    void train(const std::string& text, int vocab_size, bool verbose = false) {
        std::vector<std::string> chunks = pretokenize(text);
        std::unordered_map<std::string, long> counts;
        counts.reserve(chunks.size() * 2);
        for (auto& ch : chunks) counts[ch]++;
        std::vector<std::vector<int>> seqs; std::vector<long> mult;
        seqs.reserve(counts.size()); mult.reserve(counts.size());
        for (auto& kv : counts) {
            std::vector<int> ids(kv.first.size());
            for (size_t k = 0; k < kv.first.size(); ++k) ids[k] = (unsigned char)kv.first[k];
            seqs.push_back(std::move(ids)); mult.push_back(kv.second);
        }
        int num_merges = vocab_size - 256;
        vocab.resize(256);
        for (int i = 0; i < 256; ++i) vocab[i] = std::string(1, (char)i);
        merges.clear();
        for (int m = 0; m < num_merges; ++m) {
            std::unordered_map<uint64_t, long> stats; stats.reserve(1 << 16);
            for (size_t s = 0; s < seqs.size(); ++s) {
                const std::vector<int>& v = seqs[s]; long w = mult[s];
                for (size_t t = 0; t + 1 < v.size(); ++t) stats[key(v[t], v[t + 1])] += w;
            }
            if (stats.empty()) break;
            uint64_t best = 0; long bestc = -1;
            for (auto& kv : stats)
                if (kv.second > bestc || (kv.second == bestc && kv.first < best)) { bestc = kv.second; best = kv.first; }
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
            uint64_t bestpair = 0; int bestrank = INT32_MAX;
            for (size_t t = 0; t + 1 < ids.size(); ++t) {
                auto it = merges.find(key(ids[t], ids[t + 1]));
                if (it != merges.end() && it->second < bestrank) { bestrank = it->second; bestpair = key(ids[t], ids[t + 1]); }
            }
            if (bestrank == INT32_MAX) break;
            ids = merge_seq(ids, bestpair, bestrank);
        }
        return ids;
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<std::string> chunks = pretokenize(text);
        std::unordered_map<std::string, std::vector<int>> cache;
        std::vector<int> out; out.reserve(text.size() / 3 + 16);
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
// QuatLayer -- from qnn2.cpp, used for hidden layers, the vocab head, and all
// attention projections (Q/K/V/out).
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
        : num_inputs(inputs), num_outputs(outputs), threshold(thresh * ACC_SCALE),
          run_mean(outputs, 0), run_dev(outputs, std::max(1, inputs / 8)) {
        weights.resize(num_outputs, std::vector<uint8_t>(num_inputs));
        accumulators.resize(num_outputs, std::vector<int>(num_inputs, 0));
        // Draw in the ORIGINAL range then scale by ACC_SCALE: this keeps the RNG
        // stream and the quantized init weights bit-identical to the reference
        // (quantize boundary is threshold/2 = thresh*8, so draw*16 >= thresh*8
        // iff draw >= thresh/2 -- same decision as the un-scaled version).
        for (int o = 0; o < num_outputs; ++o)
            for (int i = 0; i < num_inputs; ++i) {
                accumulators[o][i] = init_draw(thresh, gen) * ACC_SCALE;
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
            outputs[o] = quantize_band(c, d);
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
    // Re-quantize every weight from its accumulator (minibatch commit).
    void commit_all() {
        for (int o = 0; o < num_outputs; ++o)
            for (int i = 0; i < num_inputs; ++i)
                weights[o][i] = quantize(accumulators[o][i]);
    }
    // push_scale defaults to full strength (ACC_SCALE). The head passes a
    // smaller scale for its negative-class update => higher effective threshold.
    void update_weights(int o, const std::vector<uint8_t>& inputs, bool match,
                        int push_scale = ACC_SCALE) {
        static std::vector<uint8_t> tele_old;               // telemetry scratch
        if (g_tele_on) tele_old.assign(weights[o].begin(), weights[o].end());
        // MODE B: RESET-ON-FLIP (integrate-and-fire). Accumulator = evidence toward
        // the next single-notch move; at +/-threshold, step the weight one level and
        // reset to 0. Cannot rail (acc never leaves (-threshold,+threshold)).
        if (g_reset_mode) {
            int fire = threshold / std::max(1, g_reset_fire_div);
            for (int i = 0; i < num_inputs; ++i) {
                int w = weights[o][i];
                int push = std::abs(qscore(inputs[i], w)) * ((inputs[i] >= 2) ? 1 : -1);
                if (!match) push = -push;
                accumulators[o][i] += (int)((long)push * push_scale * g_boost_num / g_boost_den);
                if (accumulators[o][i] >= fire) {
                    if (w < 3) weights[o][i] = (uint8_t)(w + 1);
                    accumulators[o][i] = 0;                      // fire + reset
                } else if (accumulators[o][i] <= -fire) {
                    if (w > 0) weights[o][i] = (uint8_t)(w - 1);
                    accumulators[o][i] = 0;
                }
            }
            tele_record(o, inputs, tele_old);
            return;
        }
        // 1. Push the graded signal into the accumulators (clamped). Unchanged.
        for (int i = 0; i < num_inputs; ++i) {
            int w = weights[o][i];
            int in = inputs[i];
            int push = std::abs(qscore(in, w)) * ((in >= 2) ? 1 : -1);
            if (!match) push = -push;
            accumulators[o][i] += (int)((long)push * push_scale * g_boost_num / g_boost_den);
            int clamp = (g_clamp_pct >= 100) ? threshold
                                             : (int)((long)threshold * g_clamp_pct / 100);
            if (accumulators[o][i] > clamp) accumulators[o][i] = clamp;
            else if (accumulators[o][i] < -clamp) accumulators[o][i] = -clamp;
            // MODE A: LEAK -- bleed the accumulator toward 0 to un-pin railed weights.
            if (g_leak_div > 0) accumulators[o][i] -= accumulators[o][i] / g_leak_div;
        }
        if (g_trust_frac <= 0.0) {
            // ORIGINAL: commit every pending flip immediately (bit-identical).
            // In minibatch mode weights stay frozen until commit_all() at batch end.
            if (!g_defer_commit)
                for (int i = 0; i < num_inputs; ++i)
                    weights[o][i] = quantize(accumulators[o][i]);
            tele_record(o, inputs, tele_old);
            return;
        }
        // 2. TRUST REGION: bound the net change to s_o to g_trust_frac*run_dev[o].
        //    weights[] holds the COMMITTED levels; accumulators[] may hold PENDING
        //    flips (weights[o][i] != quantize(acc)). forward()/compute_scores use
        //    weights[], so the committed state is what the network sees; a pending
        //    flip re-competes for the budget every step until it wins.
        long cap = (long)(g_trust_frac * (double)std::max(1, run_dev[o]));
        static std::vector<int> cand;           // single-threaded scratch
        cand.clear();
        for (int i = 0; i < num_inputs; ++i)
            if (quantize(accumulators[o][i]) != weights[o][i]) cand.push_back(i);
        if (cand.empty()) { tele_record(o, inputs, tele_old); return; }
        // most-earned first: largest |accumulator| (deepest past the boundary)
        std::sort(cand.begin(), cand.end(), [&](int a, int b) {
            return std::abs(accumulators[o][a]) > std::abs(accumulators[o][b]);
        });
        long running = 0;
        for (int idx : cand) {
            int old_lvl = weights[o][idx];
            int new_lvl = quantize(accumulators[o][idx]);
            long dscore = (long)qscore(inputs[idx], new_lvl)
                        - (long)qscore(inputs[idx], old_lvl);
            if (std::labs(running + dscore) <= cap) {
                weights[o][idx] = (uint8_t)new_lvl;   // commit this flip
                running += dscore;
            }
            // else: leave pending -- weight keeps old_lvl, accumulator retained.
        }
        tele_record(o, inputs, tele_old);
    }
private:
    // READ-ONLY: accumulate per-component telemetry for row o vs its pre-update
    // levels old_w. Never mutates dynamics. Tagged by the global g_tele_tag.
    void tele_record(int o, const std::vector<uint8_t>& inputs,
                     const std::vector<uint8_t>& old_w) {
        if (!g_tele_on || g_tele_tag < 0) return;
        int tg = g_tele_tag;
        for (int i = 0; i < num_inputs; ++i) {
            g_tele_touch[tg]++;
            if (std::abs(accumulators[o][i]) >= threshold) g_tele_sat[tg]++;
            int nw = weights[o][i], ow = old_w[i];
            if (nw != ow) {
                g_tele_flip[tg]++;
                long dqs = (long)qscore(inputs[i], nw) - (long)qscore(inputs[i], ow);
                g_tele_dsig[tg] += dqs;
                g_tele_dabs[tg] += std::labs(dqs);
            }
        }
    }
};

// =========================================================================
// EmbeddingTable -- learnable quaternary rows (same as qllm.cpp) plus a
// lookup_rows() that returns the per-token rows as a sequence for attention.
// =========================================================================
class EmbeddingTable {
private:
    int vocab;
    int dim;
    int threshold;
    std::vector<std::vector<uint8_t>> emb;
    std::vector<std::vector<int>> accumulators;
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
        // The embedding always inits RANDOM across all levels -- the token rows
        // need full diversity to stay distinguishable; the init-mode restriction
        // applies only to the layer weights (attention / FFN / head).
        std::uniform_int_distribution<> dis(-thresh, thresh);
        for (int v = 0; v < vocab; ++v)
            for (int d = 0; d < dim; ++d) {
                accumulators[v][d] = dis(gen);
                emb[v][d] = quantize(accumulators[v][d]);
            }
    }
    int embed_dim() const { return dim; }
    void commit_all() {
        for (int v = 0; v < vocab; ++v)
            for (int d = 0; d < dim; ++d)
                emb[v][d] = quantize(accumulators[v][d]);
    }

    // Per-token rows, as a sequence of C quaternary vectors (for attention).
    std::vector<std::vector<uint8_t>> lookup_rows(const std::vector<int>& ctx) const {
        std::vector<std::vector<uint8_t>> rows(ctx.size(), std::vector<uint8_t>(dim));
        for (size_t s = 0; s < ctx.size(); ++s)
            for (int d = 0; d < dim; ++d) rows[s][d] = emb[ctx[s]][d];
        return rows;
    }

    // Push each cell toward +/-threshold along the graded input signal
    // (index s*dim+d belongs to token ctx[s]), gated like qnn2's updates.
    void update(const std::vector<int>& ctx, const std::vector<int>& signal) {
        long abs_sum = 0;
        for (int s : signal) abs_sum += std::abs(s);
        long cutoff = std::max(1L, abs_sum / (long)std::max<size_t>(1, signal.size()));
        for (size_t s = 0; s < ctx.size(); ++s) {
            int tok = ctx[s];
            for (int d = 0; d < dim; ++d) {
                int sig = signal[s * dim + d];
                if (std::abs(sig) < cutoff) continue;
                int oe = emb[tok][d];                          // telemetry: pre-update
                int push = (sig > 0) ? 2 : -2;
                accumulators[tok][d] += (int)((long)push * g_boost_num / g_boost_den);
                if (accumulators[tok][d] > threshold) accumulators[tok][d] = threshold;
                else if (accumulators[tok][d] < -threshold) accumulators[tok][d] = -threshold;
                if (!g_defer_commit) emb[tok][d] = quantize(accumulators[tok][d]);
                if (g_tele_on) {                               // READ-ONLY telemetry
                    g_tele_touch[T_EMB]++;
                    if (std::abs(accumulators[tok][d]) >= threshold) g_tele_sat[T_EMB]++;
                    int ne = emb[tok][d];
                    if (ne != oe) {
                        g_tele_flip[T_EMB]++;
                        long dq = (long)qpolarity(ne) - (long)qpolarity(oe);
                        g_tele_dsig[T_EMB] += dq;
                        g_tele_dabs[T_EMB] += std::labs(dq);
                    }
                }
            }
        }
    }
};

// =========================================================================
// QuatSelfAttention -- causal, multi-head, trainable through the value path.
// Stores the last forward pass (single training example) so backward can reuse
// the attention weights and value vectors.
// =========================================================================
class QuatSelfAttention {
private:
    int n_embed;
    int n_head;
    int head_size;
    int threshold;
    std::vector<QuatLayer> q_proj, k_proj, v_proj;   // per head
    QuatLayer out_proj;                              // concat(E) -> E
    mutable std::vector<std::vector<int>> agg_mean, agg_dev;  // [head][dim] EMA

    // --- forward state kept for backward (batch of 1) ---
    std::vector<std::vector<uint8_t>> seq_;                          // C x E
    std::vector<std::vector<std::vector<uint8_t>>> Q_, K_, V_;       // [head][C][hs]
    std::vector<std::vector<std::vector<int>>> wgt_;                 // [head][t][j]
    std::vector<std::vector<uint8_t>> concat_;                       // C x E
    std::vector<std::vector<uint8_t>> attn_out_;                     // C x E

    static int qk_dot(const std::vector<uint8_t>& q, const std::vector<uint8_t>& k) {
        int s = 0;
        for (size_t d = 0; d < q.size(); ++d) s += qscore(q[d], k[d]);
        return s;
    }

public:
    QuatSelfAttention(int E, int H, int thresh, std::mt19937& gen)
        : n_embed(E), n_head(H), head_size(E / H), threshold(thresh),
          out_proj(E, E, thresh, gen),
          agg_mean(H, std::vector<int>(E / H, 0)),
          agg_dev(H, std::vector<int>(E / H, 1)) {
        for (int h = 0; h < H; ++h) {
            q_proj.emplace_back(E, head_size, thresh, gen);
            k_proj.emplace_back(E, head_size, thresh, gen);
            v_proj.emplace_back(E, head_size, thresh, gen);
        }
    }

    // Causal forward. seq is C quaternary vectors of dim n_embed. Returns the
    // C context-mixed quaternary vectors (dim n_embed). Stores state.
    std::vector<std::vector<uint8_t>> forward(const std::vector<std::vector<uint8_t>>& seq) {
        int C = (int)seq.size();
        seq_ = seq;
        Q_.assign(n_head, std::vector<std::vector<uint8_t>>(C));
        K_.assign(n_head, std::vector<std::vector<uint8_t>>(C));
        V_.assign(n_head, std::vector<std::vector<uint8_t>>(C));
        wgt_.assign(n_head, std::vector<std::vector<int>>(C));
        concat_.assign(C, std::vector<uint8_t>(n_embed));

        for (int h = 0; h < n_head; ++h) {
            std::vector<std::vector<uint8_t>> Q(C), K(C), V(C);
            for (int t = 0; t < C; ++t) {
                Q[t] = q_proj[h].forward(seq[t]);
                K[t] = k_proj[h].forward(seq[t]);
                V[t] = v_proj[h].forward(seq[t]);
            }
            Q_[h] = Q; K_[h] = K; V_[h] = V;
            std::vector<int> scores(C);
            for (int t = 0; t < C; ++t) {
                int last = t;                                   // causal
                long mean = 0;
                for (int j = 0; j <= last; ++j) { scores[j] = qk_dot(Q[t], K[j]); mean += scores[j]; }
                mean /= (last + 1);
                std::vector<int> w(last + 1);
                long sumw = 0; int best = 0, bestscore = INT_MIN;
                for (int j = 0; j <= last; ++j) {
                    int val = scores[j] - (int)mean;
                    w[j] = (val > 0) ? val : 0;
                    sumw += w[j];
                    if (scores[j] > bestscore) { bestscore = scores[j]; best = j; }
                }
                if (sumw == 0) { std::fill(w.begin(), w.end(), 0); w[best] = 1; }
                wgt_[h][t] = w;
                for (int d = 0; d < head_size; ++d) {
                    long acc = 0;
                    for (int j = 0; j <= last; ++j) if (w[j]) acc += (long)w[j] * qpolarity(V[j][d]);
                    agg_mean[h][d] += (int)((acc - agg_mean[h][d]) / 64);
                    int c = (int)acc - agg_mean[h][d];
                    agg_dev[h][d] += (std::abs(c) - agg_dev[h][d]) / 64;
                    int band = std::max(1, agg_dev[h][d]);
                    concat_[t][h * head_size + d] = quantize_band(c, band);
                }
            }
        }
        attn_out_.assign(C, std::vector<uint8_t>(n_embed));
        for (int t = 0; t < C; ++t) attn_out_[t] = out_proj.forward(concat_[t]);
        return attn_out_;
    }

    // Backward. out_sig is the graded signal on the attention output (C x E).
    // Updates out_proj and the value projections; returns the signal on the
    // embedding rows (C x E) to hand to EmbeddingTable::update.
    std::vector<std::vector<int>> backward(const std::vector<std::vector<int>>& out_sig) {
        int C = (int)out_sig.size();

        // ---- 1. back through out_proj, per position ----
        g_tele_tag = T_OUT;
        std::vector<std::vector<int>> concat_sig(C, std::vector<int>(n_embed, 0));
        for (int t = 0; t < C; ++t) {
            std::vector<int> sig = out_sig[t];
            long mean = 0; for (int o = 0; o < n_embed; ++o) mean += sig[o];
            mean /= n_embed;
            long abs_sum = 0;
            for (int o = 0; o < n_embed; ++o) { sig[o] -= (int)mean; abs_sum += std::abs(sig[o]); }
            long cutoff = std::max(1L, abs_sum / n_embed);
            for (int o = 0; o < n_embed; ++o) {
                if (satgate_skip(attn_out_[t][o], sig[o] > 0)) continue;     // saturation gate
                if (std::abs(sig[o]) < cutoff) continue;                     // magnitude gate
                out_proj.update_weights(o, concat_[t], sig[o] > 0);
            }
            for (int i = 0; i < n_embed; ++i) {
                long acc = 0;
                for (int o = 0; o < n_embed; ++o) acc += (long)sig[o] * qpolarity(out_proj.get_weight(o, i));
                concat_sig[t][i] = (int)acc;
            }
        }

        // ---- 2 & 3. per head: value-projection update + signal to embeddings ----
        std::vector<std::vector<int>> seq_sig(C, std::vector<int>(n_embed, 0));
        for (int h = 0; h < n_head; ++h) {
            // Spread each query t's head signal onto the values it attended to.
            std::vector<std::vector<int>> val_sig(C, std::vector<int>(head_size, 0));
            for (int t = 0; t < C; ++t) {
                const std::vector<int>& w = wgt_[h][t];         // over j = 0..t
                for (int j = 0; j < (int)w.size(); ++j) {
                    if (!w[j]) continue;
                    for (int d = 0; d < head_size; ++d)
                        val_sig[j][d] += concat_sig[t][h * head_size + d] * w[j];
                }
            }
            // Update v_proj and propagate to the embedding rows.
            g_tele_tag = T_V;
            for (int j = 0; j < C; ++j) {
                long abs_sum = 0; for (int d = 0; d < head_size; ++d) abs_sum += std::abs(val_sig[j][d]);
                long cutoff = std::max(1L, abs_sum / head_size);
                for (int d = 0; d < head_size; ++d) {
                    if (satgate_skip(V_[h][j][d], val_sig[j][d] > 0)) continue;  // saturation gate
                    if (std::abs(val_sig[j][d]) < cutoff) continue;       // magnitude gate
                    v_proj[h].update_weights(d, seq_[j], val_sig[j][d] > 0);
                }
                // Propagate to the block's input signal. NOTE: unlike the
                // v_proj update above, this path does NOT apply the saturation
                // gate. Gating propagation on V==0/3 starves lower blocks in a
                // deep stack (each block would drop ~half the signal), so we let
                // it flow through and only gate the weight *updates*.
                for (int i = 0; i < n_embed; ++i) {
                    long acc = 0;
                    for (int d = 0; d < head_size; ++d)
                        acc += (long)val_sig[j][d] * qpolarity(v_proj[h].get_weight(d, i));
                    seq_sig[j][i] += (int)acc;
                }
            }

            // ---- learnable Q/K router (Hebbian surrogate for the softmax
            // gradient) ----
            // For each query t and each key j it attended to, ask: would
            // attending MORE to j have moved the output the right way?
            //   route(t,j) = sum_d out_signal[t][d] * qpolarity(V[j][d])
            // (out_signal is this head's output signal = concat_sig slice).
            // route>0  -> that key helped -> raise score(t,j): push Q[t] and
            //            K[j] toward the same side (agree more).
            // route<0  -> it hurt -> lower score(t,j): push them apart.
            // The nudge is the same accumulator push update_weights uses; only
            // above-average |route| keys update, so weak routes stay put.
            for (int t = 0; t < C; ++t) {
                const std::vector<int>& w = wgt_[h][t];
                std::vector<int> route(w.size(), 0);
                long abs_sum = 0; int cnt = 0;
                for (int j = 0; j < (int)w.size(); ++j) {
                    if (!w[j]) continue;
                    int r = 0;
                    for (int d = 0; d < head_size; ++d)
                        r += concat_sig[t][h * head_size + d] * qpolarity(V_[h][j][d]);
                    route[j] = r; abs_sum += std::abs(r); ++cnt;
                }
                if (cnt == 0) continue;
                long cutoff = std::max(1L, abs_sum / cnt);
                for (int j = 0; j < (int)w.size(); ++j) {
                    if (!w[j] || std::abs(route[j]) < cutoff) continue;
                    bool more = (route[j] > 0);
                    for (int d = 0; d < head_size; ++d) {
                        bool k_hi = (K_[h][j][d] >= 2);
                        bool q_hi = (Q_[h][t][d] >= 2);
                        // move Q[t][d] toward K's side (more) or away (less)
                        g_tele_tag = T_Q;
                        q_proj[h].update_weights(d, seq_[t], more ? k_hi : !k_hi);
                        // move K[j][d] toward Q's side (more) or away (less)
                        g_tele_tag = T_K;
                        k_proj[h].update_weights(d, seq_[j], more ? q_hi : !q_hi);
                    }
                }
            }
        }
        return seq_sig;
    }
    void commit_all() {
        for (int h = 0; h < n_head; ++h) {
            q_proj[h].commit_all(); k_proj[h].commit_all(); v_proj[h].commit_all();
        }
        out_proj.commit_all();
    }
};

// =========================================================================
// QuatLM -- embedding -> attention -> PER-POSITION (shared) hidden+head.
// =========================================================================
// Decoder-style: every context position t predicts token t+1 from attn_out[t]
// ALONE (dim n_embed, not the whole concatenated context). Two consequences:
//   * attention is now the ONLY path by which position t can see earlier
//     tokens -- so it can finally matter (in the flatten-then-MLP design the
//     MLP already saw every position, making attention redundant + lossy);
//   * each training step yields C predictions / C targets instead of 1, ~C x
//     more learning signal per forward pass.
// The hidden layers and vocab head are SHARED across positions; the graded
// signals from all positions accumulate back into attention and the embedding.
// =========================================================================
// TransformerBlock -- ONE transformer-style block: causal self-attention
// followed by a PER-POSITION feed-forward MLP (E -> ff -> ... -> E). Unlike the
// original design (ALL attention blocks, THEN a separate MLP head), here each
// block interleaves attention + MLP so the stack is attn/MLP/attn/MLP/... like a
// normal transformer. No residuals (the discrete reps have no add). The MLP maps
// E -> E so blocks stack; it is applied independently to every position.
// =========================================================================
class TransformerBlock {
private:
    int n_embed;
    QuatSelfAttention attn;
    std::vector<QuatLayer> mlp;                   // E -> ff -> ... -> E
    // forward state for backward (whole sequence):
    std::vector<std::vector<uint8_t>> attn_out_;                       // C x E
    std::vector<std::vector<std::vector<uint8_t>>> mlp_acts_;          // C x [layer] x dim
public:
    TransformerBlock(int E, int n_head, int ff_dim, int mlp_layers, int thresh,
                     std::mt19937& gen)
        : n_embed(E), attn(E, n_head, thresh, gen) {
        if (mlp_layers < 1) mlp_layers = 1;
        // dims: E->ff, ff->ff, ..., ff->E  (last layer outputs E so blocks stack)
        int in = E;
        for (int i = 0; i < mlp_layers; ++i) {
            int out = (i == mlp_layers - 1) ? E : ff_dim;
            mlp.emplace_back(in, out, thresh, gen);
            in = out;
        }
    }

    // seq: C quaternary vectors dim E -> C quaternary vectors dim E.
    std::vector<std::vector<uint8_t>> forward(const std::vector<std::vector<uint8_t>>& seq) {
        attn_out_ = attn.forward(seq);                     // C x E
        int C = (int)attn_out_.size();
        mlp_acts_.assign(C, {});
        std::vector<std::vector<uint8_t>> out(C);
        for (int t = 0; t < C; ++t) {
            std::vector<uint8_t> cur = attn_out_[t];
            std::vector<std::vector<uint8_t>> acts;
            acts.reserve(mlp.size());
            for (auto& layer : mlp) { cur = layer.forward(cur); acts.push_back(cur); }
            mlp_acts_[t] = std::move(acts);
            out[t] = std::move(cur);                       // dim E
        }
        return out;
    }

    // out_sig: signal on this block's output (C x E). Updates the MLP + attention;
    // returns signal on this block's INPUT (C x E). Mirrors the per-layer gated
    // backprop used by the shared head. MLP updates run at full strength (only the
    // LM head's negative class is softened, per the head-only asymmetry recipe).
    std::vector<std::vector<int>> backward(const std::vector<std::vector<int>>& out_sig) {
        int C = (int)out_sig.size();
        std::vector<std::vector<int>> attn_sig(C, std::vector<int>(n_embed, 0));
        for (int t = 0; t < C; ++t) {
            std::vector<int> sig = out_sig[t];             // on last MLP layer's output
            for (int li = (int)mlp.size() - 1; li >= 0; --li) {
                QuatLayer& layer = mlp[li];
                const std::vector<uint8_t>& prev_act =
                    (li == 0) ? attn_out_[t] : mlp_acts_[t][li - 1];
                const std::vector<uint8_t>& own_act = mlp_acts_[t][li];
                int n_out = layer.get_num_outputs();
                long mean_sig = 0;
                for (int h = 0; h < n_out; ++h) mean_sig += sig[h];
                mean_sig /= n_out;
                long abs_sum = 0;
                for (int h = 0; h < n_out; ++h) { sig[h] -= (int)mean_sig; abs_sum += std::abs(sig[h]); }
                long cutoff = std::max(1L, abs_sum / n_out);
                g_tele_tag = T_FFN;
                for (int h = 0; h < n_out; ++h) {
                    if (satgate_skip(own_act[h], sig[h] > 0)) continue;
                    if (std::abs(sig[h]) < cutoff) continue;
                    layer.update_weights(h, prev_act, sig[h] > 0);
                }
                std::vector<int> prev_sig(layer.get_num_inputs(), 0);
                for (int h = 0; h < n_out; ++h) {
                    int s = sig[h];
                    if (s == 0 || own_act[h] == 0 || own_act[h] == 3) continue;
                    for (int i = 0; i < layer.get_num_inputs(); ++i)
                        prev_sig[i] += s * qpolarity(layer.get_weight(h, i));
                }
                sig = std::move(prev_sig);
            }
            attn_sig[t] = std::move(sig);                  // signal on attn_out[t]
        }
        return attn.backward(attn_sig);                    // -> signal on block input
    }
    void commit_all() { attn.commit_all(); for (auto& l : mlp) l.commit_all(); }
};

// =========================================================================
// QuatLM (INTERLEAVED) -- embedding -> [attn+MLP block] x N -> per-position LM
// head (E -> vocab, with the head-only negative asymmetry). Same per-position
// training as before, but each MLP now lives INSIDE its block instead of as a
// single separate post-attention stack.
// =========================================================================
class QuatLM {
private:
    int context_len;
    int n_embed;
    int neg_push_scale;               // LM-head negative-class push scale (<= ACC_SCALE)
    EmbeddingTable embedding;
    std::vector<TransformerBlock> blocks;   // stacked attn+MLP transformer blocks
    QuatLayer output_layer;                 // LM head: E -> vocab
    std::mt19937 train_rng;
    std::vector<int> pos_order;

    // Shared LM head on one position's rep (dim E). No hidden layers here -- the
    // MLPs are inside the blocks -- so this is a plain E->vocab projection with
    // the head-only asymmetric target/negative update.
    int run_head(const std::vector<uint8_t>& rep, int target, bool train,
                 std::vector<int>* rep_sig) {
        std::vector<int> scores = output_layer.compute_scores(rep);
        int predicted = 0, mx = -2147483647;
        for (size_t c = 0; c < scores.size(); ++c)
            if (scores[c] > mx) { mx = scores[c]; predicted = (int)c; }
        if (!train || predicted == target) return predicted;

        // signal on rep (= output_layer's input, dim E) from target/pred rows
        if (rep_sig) {
            rep_sig->assign(n_embed, 0);
            for (int i = 0; i < n_embed; ++i)
                (*rep_sig)[i] = output_layer.get_weight(target, i)
                              - output_layer.get_weight(predicted, i);
        }
        g_tele_tag = T_HEAD;
        output_layer.update_weights(target, rep, true);                     // full strength
        output_layer.update_weights(predicted, rep, false, neg_push_scale); // softened
        return predicted;
    }

public:
    static std::mt19937 global_gen;

    QuatLM(int vocab_size, int context, int embed, int n_head, int num_blocks,
           int ff_dim, int mlp_layers, int thresh, int neg_push_scale_ = ACC_SCALE)
        : context_len(context), n_embed(embed),
          neg_push_scale(std::max(1, neg_push_scale_)),
          embedding(vocab_size, embed, thresh, global_gen),
          blocks(),
          output_layer(embed, vocab_size, thresh, global_gen),
          train_rng(2024), pos_order(context) {
        if (num_blocks < 1) num_blocks = 1;
        for (int i = 0; i < context; ++i) pos_order[i] = i;
        blocks.reserve(num_blocks);
        for (int i = 0; i < num_blocks; ++i)
            blocks.emplace_back(embed, n_head, ff_dim, mlp_layers, thresh, global_gen);
    }

    int context_length() const { return context_len; }

    std::vector<int> logits(const std::vector<int>& ctx) {
        std::vector<std::vector<uint8_t>> seq = embedding.lookup_rows(ctx);
        for (auto& blk : blocks) seq = blk.forward(seq);
        return output_layer.compute_scores(seq.back());
    }

    int predict(const std::vector<int>& ctx) {
        std::vector<int> scores = logits(ctx);
        int best = 0, mx = -2147483647;
        for (size_t c = 0; c < scores.size(); ++c)
            if (scores[c] > mx) { mx = scores[c]; best = (int)c; }
        return best;
    }

    void train_step(const std::vector<int>& input_tokens, const std::vector<int>& targets,
                    int positions_per_step) {
        std::vector<std::vector<uint8_t>> seq = embedding.lookup_rows(input_tokens);
        for (auto& blk : blocks) seq = blk.forward(seq);   // top block output, C x E

        int k = std::min(positions_per_step, context_len);
        std::shuffle(pos_order.begin(), pos_order.end(), train_rng);

        std::vector<std::vector<int>> sig(context_len, std::vector<int>(n_embed, 0));
        for (int m = 0; m < k; ++m) {
            int t = pos_order[m];
            std::vector<int> rep_sig;
            run_head(seq[t], targets[t], true, &rep_sig);
            if (!rep_sig.empty()) sig[t] = std::move(rep_sig);
        }

        for (int b = (int)blocks.size() - 1; b >= 0; --b)
            sig = blocks[b].backward(sig);

        std::vector<int> emb_signal(context_len * n_embed);
        for (int t = 0; t < context_len; ++t)
            for (int d = 0; d < n_embed; ++d)
                emb_signal[t * n_embed + d] = sig[t][d];
        g_tele_tag = T_EMB;
        embedding.update(input_tokens, emb_signal);
    }

    // Minibatch commit: re-quantize every weight from its (summed) accumulator.
    void flush_batch() {
        embedding.commit_all();
        for (auto& blk : blocks) blk.commit_all();
        output_layer.commit_all();
    }
};

std::mt19937 QuatLM::global_gen(1337);

// =========================================================================
// MAIN
// =========================================================================
static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char* argv[]) {
    try {
        const int vocab_size = 1000;
        const int n_head = 4;          // head_size = n_embed / n_head
        // BASELINE for the interleaved architecture: 2 transformer blocks, each
        // = causal attention + a 2-layer per-position FFN (E->ff->E). argv[3]
        // overrides the block count.
        int num_blocks = 2;
        if (argc > 3) num_blocks = std::max(1, std::atoi(argv[3]));
        const int ff_dim = 128;        // per-block FFN hidden width
        long train_iters = 40000;
        if (argc > 1) train_iters = std::max(1L, std::atol(argv[1]));
        std::string text_path = "wikitext2_train.txt";
        if (argc > 2) text_path = argv[2];
        // argv[4] = N, LM-head negative-threshold multiplier (default 4 = winner).
        int neg_thresh_mult = 4;
        if (argc > 4) neg_thresh_mult = std::max(1, std::atoi(argv[4]));
        int neg_push_scale = std::max(1, ACC_SCALE / neg_thresh_mult);
        // argv[5] = n_embed, per-position rep width (default 64; multiple of n_head).
        int n_embed = 64;
        if (argc > 5) n_embed = std::max(n_head, std::atoi(argv[5]));
        if (n_embed % n_head != 0) n_embed = ((n_embed / n_head) + 1) * n_head;
        // argv[6] = FFN layers per block (default 2 => 1:2 attn:ffn; 1 => 1:1).
        int mlp_layers = 2;
        if (argc > 6) mlp_layers = std::max(1, std::atoi(argv[6]));
        // argv[7] = training-sample-stream seed (default 1337). Eval stays 4242.
        unsigned sampler_seed = 1337;
        if (argc > 7) sampler_seed = (unsigned)std::strtoul(argv[7], nullptr, 10);
        // argv[8] = context length (default 16).
        int context_len = 16;
        if (argc > 8) context_len = std::max(2, std::atoi(argv[8]));
        // argv[9] = base flip threshold (default 512). Was raised 256->512 for
        // inertia vs the negative-driven collapse; the head asymmetry now handles
        // negative stability, so a lower threshold (e.g. 64) => faster learning.
        int threshold = 512;
        if (argc > 9) threshold = std::max(1, std::atoi(argv[9]));
        // argv[10] = threshold-ramp iters (0 = off). Over these iters the EFFECTIVE
        // threshold ramps from threshold/boost_max UP to threshold (base), by
        // decaying a global push boost from boost_max to 1: fast/plastic early,
        // stable late. argv[11] = boost_max (default 8 => starts at threshold/8).
        long thr_ramp = (argc > 10) ? std::max(0L, std::atol(argv[10])) : 0;
        int boost_max = (argc > 11) ? std::max(1, std::atoi(argv[11])) : 8;
        // argv[12] = positions trained per step (default 4 of context_len). More
        // = more update signal/step (helps a big model bootstrap) but MORE
        // correlated updates (the 4/16 subsample was to DECORRELATE / reduce churn).
        int positions_per_step = 4;
        if (argc > 12) positions_per_step = std::max(1, std::atoi(argv[12]));
        // argv[13] = weight init mode (0 random / 1 two middle {1,2} / 2 all level 2).
        if (argc > 13) g_init_mode = std::max(0, std::min(2, std::atoi(argv[13])));
        // argv[14] = per-neuron trust-region fraction (default 0 = off). Caps the
        // net change to each hidden neuron's pre-activation to this many run_dev
        // units per step -- kills the correlated sqrt(d) overshoot behind the
        // large-n_embed collapse WITHOUT stiffening plasticity (unlike threshold).
        if (argc > 14) g_trust_frac = std::max(0.0, std::atof(argv[14]));
        // argv[15] = per-component telemetry (0 = off, 1 = on). Prints, per eval,
        // a [tele] line: for each component (emb/q/k/v/out/ffn/head) the flip-rate
        // f, flip-coherence c (1 = correlated cascade), and saturation s. READ-ONLY.
        if (argc > 15) g_tele_on = (std::atoi(argv[15]) != 0);
        // argv[16] = eval-interval override (0 = default). Small value (e.g. 500)
        // gives fine time resolution through the collapse window.
        long eval_override = (argc > 16) ? std::max(0L, std::atol(argv[16])) : 0;
        // argv[17] = accumulator LEAK divisor (0 = off). Each update, acc -= acc/div;
        // bigger div = gentler. Un-pins railed weights (MODE A).
        if (argc > 17) g_leak_div = std::max(0, std::atoi(argv[17]));
        // argv[18] = RESET-ON-FLIP mode (0 = off, 1 = on). Integrate-and-fire weight
        // update that cannot rail (MODE B).
        if (argc > 18) g_reset_mode = (std::atoi(argv[18]) != 0);
        // argv[19] = reset fire divisor (1 = fire at full threshold, 2 = at T/2 = more
        // plastic). argv[20] = accumulator clamp as % of threshold (100 = off; e.g. 70
        // caps burial depth to un-stick rails with no mid-range drag).
        if (argc > 19) g_reset_fire_div = std::max(1, std::atoi(argv[19]));
        if (argc > 20) g_clamp_pct = std::max(1, std::min(100, std::atoi(argv[20])));
        // argv[21] = saturation-gate mode (0 orig / 1 one-sided / 2 open).
        if (argc > 21) g_satgate = std::max(0, std::min(2, std::atoi(argv[21])));
        // argv[22] = multi-window minibatch size B (1 = off). B independent windows
        // trained at frozen weights, committed once -> variance reduction.
        if (argc > 22) g_batch_B = std::max(1, std::atoi(argv[22]));

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

        size_t n = (size_t)(0.9 * data.size());
        std::vector<int> train_data(data.begin(), data.begin() + n);
        std::vector<int> test_data(data.begin() + n, data.end());
        std::cout << "  train tokens " << train_data.size()
                  << ", test tokens " << test_data.size() << "\n";

        std::cout << "  INTERLEAVED transformer blocks: " << num_blocks
                  << " (each = attention + " << mlp_layers << "-layer FFN E->"
                  << ff_dim << "->E)\n";
        std::cout << "  n_embed: " << n_embed << " (head_size " << n_embed / n_head
                  << "),  context length: " << context_len << "\n";
        std::cout << "  base flip threshold: " << threshold;
        if (thr_ramp > 0)
            std::cout << " (ramp from " << threshold / boost_max << " over "
                      << thr_ramp << " iters, boost_max " << boost_max << "x)";
        std::cout << "\n";
        std::cout << "  negative flip-threshold multiplier: " << neg_thresh_mult
                  << "x (neg push scale " << neg_push_scale << "/" << ACC_SCALE << ")\n";
        std::cout << "  positions trained per step: " << positions_per_step
                  << " / " << context_len << "\n";
        std::cout << "  weight init: " << (g_init_mode==0?"random {0,1,2,3}":
                     g_init_mode==1?"two middle {1,2}":"one value {2}") << "\n";
        std::cout << "  sampler seed: " << sampler_seed << "\n";
        std::cout << "  trust-region frac: " << g_trust_frac
                  << (g_trust_frac > 0 ? " (per-neuron dS cap)" : " (off)") << "\n";
        std::cout << "  telemetry: " << (g_tele_on ? "ON (per-component f/c/s)" : "off")
                  << (eval_override > 0 ? "  eval-override " : "")
                  << (eval_override > 0 ? std::to_string(eval_override) : std::string()) << "\n";
        std::cout << "  anti-rail: leak_div " << g_leak_div
                  << (g_leak_div > 0 ? " (MODE A on)" : " (off)")
                  << ",  reset-on-flip " << (g_reset_mode ? "ON (MODE B)" : "off")
                  << (g_reset_mode ? " fire@T/" : "") << (g_reset_mode ? std::to_string(g_reset_fire_div) : std::string())
                  << ",  clamp% " << g_clamp_pct
                  << ",  satgate " << (g_satgate==0?"orig":g_satgate==1?"one-sided":"open")
                  << ",  minibatch B=" << g_batch_B << "\n";
        QuatLM model(vocab_size, context_len, n_embed, n_head, num_blocks,
                     ff_dim, mlp_layers, threshold, neg_push_scale);

        std::mt19937 sampler(sampler_seed);
        // window start p: need p+context_len in range (targets go one past input)
        std::uniform_int_distribution<size_t> train_pos(0, train_data.size() - context_len - 1);

        // FIXED eval sets (same positions every eval) so the accuracy curve is
        // comparable across checkpoints -- required for meaningful best-model
        // selection. A dedicated RNG keeps them independent of the train stream.
        auto make_eval_set = [&](const std::vector<int>& d, int samples, std::mt19937& g) {
            std::uniform_int_distribution<size_t> pos(context_len, d.size() - 1);
            std::vector<size_t> idx(samples);
            for (auto& x : idx) x = pos(g);
            return idx;
        };
        std::mt19937 eval_rng(4242);
        std::vector<size_t> train_eval = make_eval_set(train_data, 2000, eval_rng);
        std::vector<size_t> test_eval  = make_eval_set(test_data, 3000, eval_rng);

        // Evaluate next-token accuracy AND a cross-entropy loss. The model's
        // outputs are integer alignment-scores, not calibrated logits, so a raw
        // softmax would be ~one-hot and give a meaningless CE. We instead treat
        // the scores as logits and TEMPERATURE-SCALE them: find the single T
        // that minimizes mean CE (Platt/temperature calibration). That is the
        // fair "best-case" CE for comparing to a normal LM's val loss (nats),
        // and perplexity = exp(CE). Random baseline = ln(vocab) = 6.908 nats.
        struct Metrics { double acc, ce, ppl, T; };
        auto eval_metrics = [&](const std::vector<int>& d, const std::vector<size_t>& idx) -> Metrics {
            int N = (int)idx.size();
            std::vector<std::vector<int>> S(N);
            std::vector<int> tgt(N);
            std::vector<int> ctx(context_len);
            int correct = 0;
            for (int s = 0; s < N; ++s) {
                size_t i = idx[s];
                for (int c = 0; c < context_len; ++c) ctx[c] = d[i - context_len + c];
                S[s] = model.logits(ctx);
                tgt[s] = d[i];
                int am = 0, mx = -2147483647;
                for (int v = 0; v < (int)S[s].size(); ++v) if (S[s][v] > mx) { mx = S[s][v]; am = v; }
                if (am == tgt[s]) ++correct;
            }
            double bestCE = 1e18, bestT = 1.0;
            for (double T = 1.0; T <= 2048.0; T *= 1.3) {
                double ce = 0.0;
                for (int s = 0; s < N; ++s) {
                    const std::vector<int>& sc = S[s];
                    int mx = -2147483647;
                    for (int v : sc) if (v > mx) mx = v;
                    double Z = 0.0;
                    for (int v : sc) Z += std::exp((v - mx) / T);
                    ce += -((sc[tgt[s]] - mx) / T - std::log(Z));
                }
                ce /= N;
                if (ce < bestCE) { bestCE = ce; bestT = T; }
            }
            return { 100.0 * correct / N, bestCE, std::exp(bestCE), bestT };
        };

        std::cout << "\nStarting Quaternary-Attention LM training (" << train_iters
                  << " iters) ...\n";
        std::cout << "  (random-guess cross-entropy = ln(" << vocab_size
                  << ") = " << std::log((double)vocab_size) << " nats, perplexity "
                  << vocab_size << ")\n";
        // Best-checkpoint tracking: snapshot the model whenever fixed-set test
        // accuracy improves, so a later collapse can't cost us the peak.
        QuatLM best = model;
        double best_te = -1.0; long best_iter = 0;
        // Fixed, frequent interval: the model oscillates on a few-thousand-step
        // scale, so a train_iters-proportional interval undersamples the peaks
        // and biases best-checkpoint selection low. Cap at 2500.
        const long eval_interval = eval_override > 0 ? eval_override
                                 : std::max(1L, std::min<long>(2500, train_iters / 10));
        std::vector<int> input(context_len), targets(context_len);
        for (long it = 0; it <= train_iters; ++it) {
            if (it % eval_interval == 0 || it == train_iters) {
                Metrics tr = eval_metrics(train_data, train_eval);
                Metrics te = eval_metrics(test_data, test_eval);
                bool improved = te.acc > best_te;
                if (improved) { best = model; best_te = te.acc; best_iter = it; }
                std::cout << "  iter " << it
                          << "  acc(tr/te) " << tr.acc << "/" << te.acc << "%"
                          << "  CE(tr/te) " << tr.ce << "/" << te.ce << " nats"
                          << "  ppl(te) " << te.ppl
                          << (improved ? "   <- best" : "") << std::endl;  // flush to monitor
                if (g_tele_on) tele_dump(it);
            }
            if (it == train_iters) break;
            // threshold ramp: boost decays linearly boost_max -> 1 over thr_ramp
            // iters (effective threshold  threshold/boost_max -> threshold).
            if (thr_ramp > 0) {
                long rp = std::min(it, thr_ramp);
                g_boost_num = (int)(boost_max * thr_ramp - (long)(boost_max - 1) * rp);
                g_boost_den = (int)thr_ramp;
            }
            // MULTI-WINDOW MINIBATCH: B independent windows at frozen weights, then
            // one commit. B=1 = original single-window step.
            int B = std::max(1, g_batch_B);
            g_defer_commit = (B > 1);
            for (int b = 0; b < B; ++b) {
                size_t p = train_pos(sampler);
                for (int c = 0; c < context_len; ++c) {
                    input[c]   = train_data[p + c];        // token at position c
                    targets[c] = train_data[p + c + 1];    // ...predicts the next one
                }
                model.train_step(input, targets, positions_per_step);
            }
            if (B > 1) { g_defer_commit = false; model.flush_batch(); }
        }

        // Restore the best checkpoint for the final report and generation.
        model = best;
        Metrics bm = eval_metrics(test_data, test_eval);
        std::cout << "\nBest checkpoint: iter " << best_iter
                  << ", test next-token acc " << best_te << "%\n"
                  << "  test cross-entropy " << bm.ce << " nats  (perplexity "
                  << bm.ppl << ", calibration T=" << bm.T << ")\n"
                  << "  vs random-guess CE " << std::log((double)vocab_size)
                  << " nats (perplexity " << vocab_size << ")\n";

        // ---- sampled generation ----
        std::cout << "\n--- Sample generation ---\n";
        std::vector<int> generated;
        std::vector<int> gctx(context_len, 0);
        const int max_new_tokens = 300;
        const double temperature = 1.0;
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        for (int step = 0; step < max_new_tokens; ++step) {
            std::vector<int> sc = model.logits(gctx);
            int mx = *std::max_element(sc.begin(), sc.end());
            double Z = 0.0;
            std::vector<double> probs(sc.size());
            for (size_t c = 0; c < sc.size(); ++c) { probs[c] = std::exp((sc[c] - mx) / temperature); Z += probs[c]; }
            double r = uni(sampler) * Z, acc = 0.0;
            int nxt = (int)sc.size() - 1;
            for (size_t c = 0; c < probs.size(); ++c) { acc += probs[c]; if (acc >= r) { nxt = (int)c; break; } }
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
