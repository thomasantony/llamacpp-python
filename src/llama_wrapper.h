#ifndef LLAMA_WRAPPER_H
#define LLAMA_WRAPPER_H

#include "llama.h"
#include <vector>
#include <random>
#include <thread>
#include <functional>

/* High level wrapper for the C-style LLAMA API */
using std::vector;
using Callback = std::function<void(double)>;

struct InferenceParams {
    // model parameters
    std::string path_model = "";
    int32_t seed          = -1;   // RNG seed
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_predict     = 128;  // new tokens to predict
    int32_t repeat_last_n = 64;   // last n tokens to penalize
    int32_t n_batch       = 8;    // batch size for prompt processing
    int32_t n_keep        = 0;    // number of tokens to keep from initial prompt

    // sampling parameters
    int32_t top_k = 40;
    float   top_p = 0.95f;
    float   temp  = 0.80f;
    float   repeat_penalty  = 1.10f;

    bool use_mlock = false;
    bool memory_f16 = false;

    int n_ctx = 512;  // context size

    llama_context_params ctx_params = llama_context_default_params();

    Callback callback{};
};

class LlamaWrapper {
    public:
        // LLAMA API
        LlamaWrapper() = default;
        LlamaWrapper(InferenceParams inference_params) 
            : is_initialized(false), inference_params(inference_params)
        {}
        ~LlamaWrapper() {
            if (ctx)
            {
                llama_free(ctx);
            }
        };

        // Initialize the model
        bool init();
        // Check if the model is initialized
        bool is_init() const { return is_initialized; }

        // Input processing and inference
        // Tokenize text
        const vector<llama_token> tokenize_text(const std::string& text, bool add_bos = false) const;
        // Queues up a BOS token to the model input
        void add_bos();
        // Clears the model input buffer
        void clear_input();
        // Set the model input buffer
        void set_input(const std::string& text);
        // Set the model input buffer from tokens
        void set_input(const vector<llama_token>& tokens);
        // Queues up input text to the model input
        void update_input(const std::string& text);
        // Queues up input tokens to the model input
        void update_input(const vector<llama_token>& tokens);
        // Ingests input previously added using update_input()
        void ingest_input_batch();
        // Ingests all input previously added using update_input() in multiple batches
        // Batch size is determined by n_batch in InferenceParams
        bool ingest_all_pending_input();
        // Checks if the model has unconsumed input to be ingested using ingest_input_batch()
        bool has_unconsumed_input() const;

        // Evaluate the model on a batch of input. Must call llama_ingest_input_batch() first.
        bool eval();
        // Sample token from the model and add it to the model input
        llama_token sample();

        // Output processing
        // Get logits
        const float* get_logits() const;

        // Get embeddings
        const float* get_embeddings() const {
            return llama_get_embeddings(ctx);
        }

        int get_n_vocab() const { return llama_n_vocab(ctx); }
        int get_n_embd() const { return llama_n_embd(ctx); }

        // Convert token to str
        std::string token_to_str(llama_token token) const { return llama_token_to_str(ctx, token); }

        // Print timings
        void print_timings() const { llama_print_timings(ctx); }
        // Reset timings
        void reset_timings() const { llama_reset_timings(ctx); }

    private:
        std::string path_model = "";
        llama_context* ctx = nullptr;
        InferenceParams inference_params{};

        // Random number generator
        std::mt19937 rng{};

        // Tokens
        vector<llama_token> embd{};
        vector<llama_token> embd_inp{};
        vector<llama_token> last_n_tokens{};

        int n_consumed = 0;
        int remaining_tokens = 0;
        int n_past = 0;
        int n_ctx = 0;
        size_t mem_per_token = 0;

        bool is_initialized = false;
};

#endif /* LLAMA_WRAPPER_H */
