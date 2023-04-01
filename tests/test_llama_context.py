import pytest
import llamacpp


@pytest.fixture(scope="session")
def llama_context():
    params = llamacpp.LlamaContextParams()
    params.seed = 19472
    return llamacpp.LlamaContext("../models/7B/ggml-model-f16.bin", params)


def test_str_to_token(llama_context):
    prompt = "Hello World"
    prompt_tokens = llama_context.str_to_token(prompt, True)
    assert prompt_tokens == [1, 10994, 2787]


def test_token_to_str(llama_context):
    tokens = [1, 10994, 2787]
    text = ''.join([llama_context.token_to_str(token) for token in tokens])
    assert text == "Hello World"


def test_eval(llama_context):
    pass
