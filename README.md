## Building the Python bindings

### macOS

`brew install pybind11`

## Install python package

### From PyPI

```
pip install llamacpp
```

### From source

```
poetry install
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
```

## Run this demo script

```
import llamacpp
import os

model_path = "./models/7B/ggml-model-q4_0.bin"
params = llamacpp.gpt_params(model_path,
"Hi, I'm a llama.",
4096,
40,
0.1,
0.7,
2.0)
model = llamacpp.PyLLAMA(model_path, params)
model.predict("Hello, I'm a llama.", 10)
```

## ToDo

- [x] Use poetry to build package
- [x] Add command line entry point for quantize script
- [x] Publish wheel to PyPI
- [ ] Add chat interface based on tinygrad
