import os
import sys
import pytest
import llamacpp


@pytest.fixture
def nominal_params():
    """Nominal InfereceParams"""
    params = llamacpp.InferenceParams()
    # Get path to current file
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    # Get path to the model
    model_path = os.path.join(current_file_path, "../models/7B/ggml-model-f16.bin")
    params.path_model = model_path
    params.seed = 19472
    params.repeat_penalty = 1.0
    return params


@pytest.fixture
def limited_context_params(nominal_params):
    """InferenceParams with limited context"""
    nominal_params.n_ctx = 32
    return nominal_params


@pytest.fixture
def llama_model(nominal_params):
    return llamacpp.LlamaInference(nominal_params)


@pytest.fixture
def limited_context_llama_model(limited_context_params):
    return llamacpp.LlamaInference(limited_context_params)


def test_update_input(llama_model):
    prompt_tokens = [1, 2, 3]
    llama_model.update_input(prompt_tokens)
    assert llama_model.has_unconsumed_input()
    llama_model.ingest_all_pending_input()
    assert not llama_model.has_unconsumed_input()


def test_tokenize(llama_model):
    prompt = "Hello World"
    prompt_tokens = llama_model.tokenize(prompt, True)
    assert prompt_tokens == [1, 10994, 2787]


def test_token_to_str(llama_model):
    tokens = [1, 10994, 2787]
    text = ''.join([llama_model.token_to_str(token) for token in tokens])
    assert text == "Hello World"


def test_eval(llama_model):
    prompt = " Llama is"
    prompt_tokens = llama_model.tokenize(prompt, True)
    llama_model.update_input(prompt_tokens)
    llama_model.ingest_all_pending_input()
    output = prompt
    for i in range(8):
        llama_model.eval()
        token = llama_model.sample()
        output += llama_model.token_to_str(token)

    assert output == " Llama is the newest member of our growing family"


def test_eval_exceed_n_ctx(limited_context_llama_model):
    # Tests context swapping feature
    llama_model = limited_context_llama_model
    prompt = " Llama is"
    prompt_tokens = llama_model.tokenize(prompt, True)
    llama_model.update_input(prompt_tokens)
    llama_model.ingest_all_pending_input()
    output = prompt
    # Generate 35 tokens with n_ctx of 32
    for i in range(35):
        llama_model.eval()
        token = llama_model.sample()
        output += llama_model.token_to_str(token)

    assert output == " Llama is the newest member of our growing family. We’re excited to welcome him to the pack!\nLlama is a male, born in 2017"


if __name__=='__main__':
    sys.exit(pytest.main(['-s', '-v', __file__]))
