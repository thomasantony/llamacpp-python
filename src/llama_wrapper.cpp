#include "llama_wrapper.h"
#include <cassert>

static void trigger_cb(float progress, void * user_data) {
    if (user_data == nullptr) {
        return;
    }
    auto cb = static_cast<Callback*>(user_data);
    (*cb)(progress);
}

// Initialize the model
bool LlamaWrapper::init()
{
    if (is_initialized)
    {
        return true;
    }
    // update pointer to callback if needed
    if(inference_params.callback)
    {
        using raw_cb = void (*)(float, void*);
        inference_params.ctx_params.progress_callback = (raw_cb)trigger_cb;
        inference_params.ctx_params.progress_callback_user_data = &inference_params.callback;
    }else{
        inference_params.ctx_params.progress_callback = nullptr;
    }

    // update default ctx params with our user-selected overrides
    inference_params.ctx_params.n_ctx = inference_params.n_ctx;
    inference_params.ctx_params.seed = inference_params.seed;
    inference_params.ctx_params.use_mlock = inference_params.use_mlock;
    inference_params.ctx_params.f16_kv = inference_params.memory_f16;

    ctx = llama_init_from_file(inference_params.path_model.c_str(), inference_params.ctx_params);

    n_ctx = llama_n_ctx(ctx);
    last_n_tokens = std::vector<llama_token>(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    is_initialized = true;
    return true;
}
// Tokenize text
const vector<llama_token> LlamaWrapper::tokenize_text(const std::string& text, bool add_bos) const
{
    // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
    std::vector<llama_token> res(text.size() + (int)add_bos);
    int n = llama_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
    assert(n >= 0);
    res.resize(n);
    return res;
}

// Add BOS token to input
void LlamaWrapper::add_bos() {
    embd_inp.push_back(llama_token_bos());
}

// Clear the model input buffer
void LlamaWrapper::clear_input()
{
    embd_inp.clear();
    n_consumed = 0;
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
}

// Set the model input buffer
void LlamaWrapper::set_input(const std::string& text)
{
    set_input(tokenize_text(text));
}

// Set the model input buffer from tokens
void LlamaWrapper::set_input(const vector<llama_token>& tokens)
{
    embd_inp.clear();
    update_input(tokens);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    n_consumed = 0;
    n_past = 0;
}

// Update input with text
void LlamaWrapper::update_input(const std::string& text)
{
    update_input(tokenize_text(text));
}

// Update input with tokens
void LlamaWrapper::update_input(const vector<llama_token>& tokens)
{
    embd_inp.insert(embd_inp.end(), tokens.begin(), tokens.end());
}

// Ingest one batch of input
void LlamaWrapper::ingest_input_batch()
{
    // Copy at most n_batch elements from embd_inp to embd
    size_t num_copied = std::min((size_t) inference_params.n_batch+1, embd_inp.size() - n_consumed);
    std::copy(embd_inp.begin() + n_consumed,
                embd_inp.begin() + n_consumed + num_copied,
                std::back_inserter(embd));
    n_consumed += num_copied;

    // Copy the last `repeat_last_n` elements copied into embd to last_n_tokens
    size_t num_copied_last_n = std::min(num_copied, (size_t) inference_params.repeat_last_n);
    last_n_tokens.erase(last_n_tokens.begin(), last_n_tokens.begin()+num_copied_last_n);
    last_n_tokens.insert(last_n_tokens.end(), embd.end() - num_copied_last_n, embd.end());
}

// Ingest all input
bool LlamaWrapper::ingest_all_pending_input()
{
    while (has_unconsumed_input())
    {
        ingest_input_batch();
        eval();
    }
    return true;
}

// Check if there is unconsumed input
bool LlamaWrapper::has_unconsumed_input() const
{
    return n_consumed < embd_inp.size();
}

// Eval model and clear input
bool LlamaWrapper::eval()
{
    if (embd.size() > 0) {
        // infinite text generation via context swapping
        // if we run out of context:
        // - take the n_keep first tokens from the original prompt (via n_past)
        // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
        if (n_past + (int) embd.size() > n_ctx) {
            const int n_left = n_past - inference_params.n_keep;

            n_past = inference_params.n_keep;

            // insert n_left/2 tokens at the start of embd from last_n_tokens
            embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());
        }
        if (llama_eval(ctx, embd.data(), embd.size(), n_past, inference_params.n_threads) != 0) {
            fprintf(stderr, "Failed to predict\n");
            return false;
        }
    }
    n_past += embd.size();
    embd.clear();
    return true;
}

// Sample from logits
llama_token LlamaWrapper::sample()
{
    llama_token id = 0;

    {
        id = llama_sample_top_p_top_k(
                ctx,
                last_n_tokens.data() + n_ctx - inference_params.repeat_last_n,
                inference_params.repeat_last_n,
                inference_params.top_k,
                inference_params.top_p,
                inference_params.temp,
                inference_params.repeat_penalty
        );

        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
        embd.push_back(id);
    }
    return id;
}

// Get the logits for the last token
const float* LlamaWrapper::get_logits() const
{
    return llama_get_logits(ctx);
}
