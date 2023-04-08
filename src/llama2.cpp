#include "ggml.h"
#include "llama.h"
#include "llama_wrapper.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/functional.h"
#include "pybind11/numpy.h"
#include <iostream>
namespace py = pybind11;
using Callback = std::function<void(double)>;


class LlamaInference;
/* Tokenizer for use with text-ui project */
class Tokenizer {
    const LlamaInference& llama;
public:
    Tokenizer(const LlamaInference& llama): llama(llama) {}
    std::vector<llama_token> tokenize(const std::string & text, bool bos);
    std::string detokenize(const std::vector<llama_token>& ids);
    std::string detokenize(const llama_token& id);
};

// Lower level API that gives more direct access to llama_context
class LlamaContext
{
    llama_context* ctx;
public:
    LlamaContext(std::string path_model, const llama_context_params& params) {
        ctx = llama_init_from_file(path_model.c_str(), params);
    }
    LlamaContext(std::string path_model, const llama_context_params& params, Callback progress_cb)
    {
        llama_context_params params_with_cb = params;
        params_with_cb.progress_callback = [](float progress, void* user_data) {
            auto cb = static_cast<Callback*>(user_data);
            (*cb)(progress);
        };
        params_with_cb.progress_callback_user_data = &progress_cb;

        ctx = llama_init_from_file(path_model.c_str(), params_with_cb);
    }
    ~LlamaContext()
    {
        llama_free(ctx);
    }

    // Run the llama inference to obtain the logits and probabilities for the next token.
    // tokens + n_tokens is the provided batch of new tokens to process
    // n_past is the number of tokens to use from previous eval calls
    // Returns 0 on success
    int eval(py::buffer tokens,
            const int n_tokens,
            const int n_past,
            const int n_threads) {
        py::buffer_info tokens_info = tokens.request();
        // Check that tokens are integers and one-dimensional
        if (tokens_info.format != py::format_descriptor<llama_token>::format() ||
            tokens_info.ndim != 1) {
            throw std::runtime_error("Invalid tokens buffer format");
        }
        // Check that the number of tokens is correct
        if (tokens_info.size < n_tokens) {
            throw std::runtime_error("Invalid number of tokens");
        }
        llama_token* tokens_ptr = (llama_token*)tokens.request().ptr;
        return llama_eval(ctx, tokens_ptr, n_tokens, n_past, n_threads);
    }

    // Sample a token from the logits
    llama_token sample_top_p_top_k(py::buffer last_n_tokens_data,
                        int   top_k,
                      float   top_p,
                      float   temp,
                      float   repeat_penalty)
    {
        py::buffer_info last_n_tokens_info = last_n_tokens_data.request();
        // Check that tokens are integers and one-dimensional
        if (last_n_tokens_info.format != py::format_descriptor<llama_token>::format() ||
            last_n_tokens_info.ndim != 1) {
            throw std::runtime_error("Invalid tokens buffer format");
        }
        llama_token* last_n_tokens_ptr = (llama_token*)last_n_tokens_info.ptr;
        size_t last_n_tokens_size = last_n_tokens_info.size;
        return llama_sample_top_p_top_k(ctx, last_n_tokens_ptr, last_n_tokens_size, top_k, top_p, temp, repeat_penalty);
    }

    // Token logits obtained from the last call to eval()
    // The logits for the last token are stored in the last row
    // Can be mutated in order to change the probabilities of the next token
    // TODO: Fix this so that it returns a 2D array
    // Length: n_vocab
    py::memoryview get_logits() const
    {
        const float* logit_ptr = llama_get_logits(ctx);
        const size_t n_vocab = llama_n_vocab(ctx);
        return py::memoryview::from_memory(
            logit_ptr,                // buffer pointer
            sizeof(float) * n_vocab   // strides in bytes
        );
    }

    // Get the embeddings for the input
    // shape: [n_embd] (1-dimensional)
    py::memoryview get_embeddings() const
    {
        const float* embd_ptr = llama_get_embeddings(ctx);
        const size_t n_embd = llama_n_embd(ctx);
        return py::memoryview::from_memory(
            embd_ptr,                // buffer pointer
            sizeof(float) * n_embd   // strides in bytes
        );
    }

    // Get the number of tokens in the vocabulary
    size_t get_n_vocab() const
    {
        return llama_n_vocab(ctx);
    }

    // Get the number of dimensions in the embedding
    size_t get_n_embd() const
    {
        return llama_n_embd(ctx);
    }

    // Get the context size
    size_t get_n_ctx() const
    {
        return llama_n_ctx(ctx);
    }


    // Token Id -> String. Uses the vocabulary in the provided context
    std::string token_to_str(llama_token token) const
    {
        return llama_token_to_str(ctx, token);
    }

    // String -> Token Id. Uses the vocabulary in the provided context
    py::array str_to_token(const std::string& text, bool add_bos) const
    {
        std::vector<llama_token> res(text.size() + (int)add_bos);
        int n = llama_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
        assert(n >= 0);
        res.resize(n);
        return py::array(res.size(), res.data());
    }

    // Performance information
    void print_timings() const
    {
        llama_print_timings(ctx);
    }
    void reset_timings()
    {
        llama_reset_timings(ctx);
    }
};


// High level API that includes input management and other convenience functions
class LlamaInference {
public:
    LlamaWrapper llama{};
    InferenceParams params{};
    LlamaInference(InferenceParams params): params(params), llama(params) {
        llama.init();
    }

    // Get tokenizer for the provided context
    // Returns a Tokenizer object
    Tokenizer get_tokenizer() const
    {
        return Tokenizer(*this);
    }
    // Run the llama inference to obtain the logits and probabilities for the next token.
    // tokens + n_tokens is the provided batch of new tokens to process
    // n_past is the number of tokens to use from previous eval calls
    // Returns 0 on success
    int eval()
    {
        return llama.eval();
    }

    // Convert the provided text into tokens.
    // Duplicate of the version in examples/common.h in llama.cpp
    std::vector<llama_token> tokenize(const std::string& text, bool add_bos) const
    {
        // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
        return llama.tokenize_text(text, add_bos);
    }

    // Token logits obtained from the last call to eval()
    // The logits for the last token are stored in the last row
    // Can be mutated in order to change the probabilities of the next token
    // Rows: n_tokens
    // Cols: n_vocab
    py::memoryview get_logits() const
    {
        const float* logit_ptr = llama.get_logits();
        const size_t n_vocab = llama.get_n_vocab();
        return py::memoryview::from_memory(
            logit_ptr,                // buffer pointer
            sizeof(float) * n_vocab   // strides in bytes
        );
    }

    // Get the embeddings for the input
    // shape: [n_embd] (1-dimensional)
    py::memoryview get_embeddings() const
    {
        const float* embd_ptr = llama.get_embeddings();
        const size_t n_embd = llama.get_n_embd();
        return py::memoryview::from_memory(
            embd_ptr,                // buffer pointer
            sizeof(float) * n_embd   // strides in bytes
        );
    }

    // Token Id -> String. Uses the vocabulary in the provided context
    std::string token_to_str(llama_token token) const
    {
        return llama.token_to_str(token);
    }

    // String -> Token Id. Uses the vocabulary in the provided context
    llama_token sample()
    {
        return llama.sample();
    }
    // Add BOS token to the input
    void add_bos()
    {
        llama.add_bos();
    }
    // set input using tokens
    void set_input(const std::vector<llama_token>& tokens)
    {
        llama.set_input(tokens);
    }
    // set input using string
    void set_input(const std::string& text)
    {
        llama.set_input(text);
    }

    // set input using tokens
    void update_input(const std::vector<llama_token>& tokens)
    {
        llama.update_input(tokens);
    }
    // update input using string
    void update_input(const std::string& text)
    {
        llama.update_input(text);
    }

    bool has_unconsumed_input() const {
        return llama.has_unconsumed_input();
    }

    void ingest_all_pending_input()
    {
        llama.ingest_all_pending_input();
    }

    // Performance information
    void print_timings()
    {
        llama.print_timings();
    }
    void reset_timings()
    {
        llama.reset_timings();
    }
};

std::vector<llama_token> Tokenizer::tokenize(const std::string & text, bool bos) {
    return llama.tokenize(text, bos);
}
std::string Tokenizer::detokenize(const std::vector<llama_token>& ids) {
    std::string output = "";
    for (auto id: ids) {
        output += detokenize(id);
    }
    return output;
}
std::string Tokenizer::detokenize(const llama_token& id) {
    return llama.token_to_str(id);
}


PYBIND11_MODULE(llamacpp, m) {
    m.doc() = "Python bindings for C++ implementation of the LLaMA language model";
    /* Wrapper for llama_context_params */
    py::class_<llama_context_params>(m, "LlamaContextParams")
        .def(py::init<>(&llama_context_default_params))
        .def_readwrite("n_ctx", &llama_context_params::n_ctx)
        .def_readwrite("n_parts", &llama_context_params::n_parts)
        .def_readwrite("seed", &llama_context_params::seed)
        .def_readwrite("f16_kv", &llama_context_params::f16_kv)
        .def_readwrite("logits_all", &llama_context_params::logits_all)
        .def_readwrite("vocab_only", &llama_context_params::vocab_only)
        .def_readwrite("use_mlock", &llama_context_params::use_mlock)
        .def_readwrite("embedding", &llama_context_params::embedding);

    /* Wrapper for InferenceParams */
    py::class_<InferenceParams>(m, "InferenceParams")
        .def(py::init<>())
        .def_static("default_with_callback", [](Callback cb){
            InferenceParams params;
            params.callback = cb;
            return params;
        }, py::arg("callback"))
        .def_readwrite("path_model", &InferenceParams::path_model)
        .def_readwrite("seed", &InferenceParams::seed)
        .def_readwrite("n_threads", &InferenceParams::n_threads)
        .def_readwrite("n_predict", &InferenceParams::n_predict)
        .def_readwrite("repeat_last_n", &InferenceParams::repeat_last_n)
        .def_readwrite("n_batch", &InferenceParams::n_batch)
        .def_readwrite("top_k", &InferenceParams::top_k)
        .def_readwrite("top_p", &InferenceParams::top_p)
        .def_readwrite("temp", &InferenceParams::temp)
        .def_readwrite("repeat_penalty", &InferenceParams::repeat_penalty)
        .def_readwrite("use_mlock", &InferenceParams::use_mlock)
        .def_readwrite("memory_f16", &InferenceParams::memory_f16)
        .def_readwrite("n_ctx", &InferenceParams::n_ctx)
        .def_readwrite("callback", &InferenceParams::callback)
        .def_readwrite("n_keep", &InferenceParams::n_keep);

    /* Wrapper for LlamaContext */
    py::class_<LlamaContext>(m, "LlamaContext")
        .def(py::init<std::string, const llama_context_params&>(), py::arg("path_model"), py::arg("params")) 
        .def(py::init<std::string, const llama_context_params&, Callback>(), py::arg("path_model"), py::arg("params"), py::arg("progress_callback"))
        .def("get_n_vocab", &LlamaContext::get_n_vocab, "Get the number of tokens in the vocabulary")
        .def("get_n_embd", &LlamaContext::get_n_embd, "Get the number of dimensions in the embedding")
        .def("get_n_ctx", &LlamaContext::get_n_ctx, "Get the number of tokens in the context")
        .def("get_embeddings", &LlamaContext::get_embeddings, "Get the embeddings as a numpy array")
        .def("token_to_str", &LlamaContext::token_to_str, "Convert a token id to a string")
        .def("str_to_token", &LlamaContext::str_to_token, "Convert a string to a token id")
        .def("print_timings", &LlamaContext::print_timings, "Print the timings for the last call to eval()")
        .def("reset_timings", &LlamaContext::reset_timings, "Reset the timings for the last call to eval()")
        .def("eval", &LlamaContext::eval, "Run the llama inference to obtain the logits and probabilities for the next token",
                py::call_guard<py::gil_scoped_release>())
        .def("sample_top_p_top_k", &LlamaContext::sample_top_p_top_k, "Sample a token from the logits using top-p and top-k");

    /* Wrapper for LlamaInference methods */
    py::class_<LlamaInference>(m, "LlamaInference")
        .def(py::init<InferenceParams>(), py::arg("params"))
        .def("set_input", py::overload_cast<const std::vector<llama_token>&>(&LlamaInference::set_input), "Set the input to the provided tokens")
        .def("set_input", py::overload_cast<const std::string&>(&LlamaInference::set_input), "Set the input to the provided tokens")
        .def("update_input", py::overload_cast<const std::vector<llama_token>&>(&LlamaInference::update_input), "Update the input with the provided tokens")
        .def("update_input", py::overload_cast<const std::string&>(&LlamaInference::update_input), "Update the input with the provided text")
        .def("eval", &LlamaInference::eval, "Run the llama inference to obtain the logits and probabilities for the next token",
                py::call_guard<py::gil_scoped_release>())
        .def("add_bos", &LlamaInference::add_bos)
        .def("tokenize", &LlamaInference::tokenize, "Convert the provided text into tokens",
                py::arg("text"), py::arg("add_bos"))
        .def("has_unconsumed_input", &LlamaInference::has_unconsumed_input, "Check if there is unconsumed input")
        .def("ingest_all_pending_input", &LlamaInference::ingest_all_pending_input, "Ingest all pending input")
        .def("get_logits", &LlamaInference::get_logits, "Get the logits for the last token", py::call_guard<py::gil_scoped_release>())
        .def("get_embeddings", &LlamaInference::get_embeddings, "Get the embeddings for the last token")
        .def("token_to_str", &LlamaInference::token_to_str, "Convert a token to a string",
                py::arg("token"))
        .def_static("token_bos", &llama_token_bos, "Get the token for the beginning of a sentence")
        .def_static("token_eos", &llama_token_eos, "Get the token for the end of a sentence")
        .def("print_timings", &LlamaInference::print_timings, "Print the timings for the last call to eval()")
        .def("reset_timings", &LlamaInference::reset_timings, "Reset the timings for the last call to eval()")
        .def_static("system_info", &llama_print_system_info, "Print system information")
        .def("sample", &LlamaInference::sample, "Sample a token from the logits")
        .def("get_tokenizer", &LlamaInference::get_tokenizer, "Get the tokenizer");
        

    // /* Wrapper for Tokenizer */
    py::class_<Tokenizer>(m, "Tokenizer")
        .def("tokenize", &Tokenizer::tokenize, "Tokenize text", py::arg("text"), py::arg("add_bos") = false)
        .def("detokenize", py::overload_cast<const std::vector<llama_token>&>(&Tokenizer::detokenize), "Detokenize text")
        .def("detokenize", py::overload_cast<const llama_token&>(&Tokenizer::detokenize), "Detokenize single token");

    /* Wrapper for llama_model_quantize */
    m.def("llama_model_quantize", &llama_model_quantize, "Quantize the LLaMA model");
}
