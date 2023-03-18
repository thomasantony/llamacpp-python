## Python bindings for llama.cpp

## Building the Python bindings

### macOS

```
brew install pybind11  # Installs dependency
git submodule init && git submodule update
poetry install
```
### From PyPI

```
pip install llamacpp
```

## Get the model weights

You will need to obtain the weights for LLaMA yourself. There are a few torrents floating around as well as some huggingface repositories (e.g https://huggingface.co/nyanko7/LLaMA-7B/). Once you have them, copy them into the models folder.

```
ls ./models
65B 30B 13B 7B tokenizer_checklist.chk tokenizer.model
```

Convert the weights to GGML format using `llamacpp-convert`. Then use `llamacpp-quantize` to quantize them into INT4. For example, for the 7B parameter model, run

```
llamacpp-convert ./models/7B/ 1
llamacpp-quantize ./models/7B/
llamacpp-cli
```

## Command line interface

The package installs the command line entry point `llamacpp-cli` that points to `llamacpp/cli.py` and should provide about the same functionality as the `main` program in the original C++ repository. There is also an experimental `llamacpp-chat` that is supposed to bring up a chat interface but this is not working correctly yet.

## Demo script

See `llamacpp/cli.py` for a detailed example. The simplest demo would be something like the following:

## ToDo

- [x] Use poetry to build package
- [x] Add command line entry point for quantize script
- [x] Publish wheel to PyPI
- [ ] Add chat interface based on tinygrad
