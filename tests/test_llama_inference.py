import pytest
import llamacpp


@pytest.fixture(scope="session")
def llama_model():
    params = llamacpp.InferenceParams()
    params.path_model = '../models/7B/ggml-model-f16.bin'
    params.seed = 19472
    return llamacpp.LlamaInference(params)


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
    prompt = "Llama is"
    prompt_tokens = llama_model.tokenize(prompt, True)
    llama_model.update_input(prompt_tokens)
    llama_model.ingest_all_pending_input()
    output = prompt
    for i in range(9):
        llama_model.eval()
        token = llama_model.sample()
        output += llama_model.token_to_str(token)

    assert output == "Llama is the newest member of our farm family."
