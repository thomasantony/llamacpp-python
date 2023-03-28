## Python bindings for llama.cpp

## Install
### From PyPI

```
pip install llamacpp
```

### Build from Source

```
pip install .
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

**Note that running `llamacpp-convert` requires `torch`, `sentencepiece` and `numpy` to be installed. These packages are not installed by default when your install `llamacpp`.**

## Command line interface

The package installs the command line entry point `llamacpp-cli` that points to `llamacpp/cli.py` and should provide about the same functionality as the `main` program in the original C++ repository. There is also an experimental `llamacpp-chat` that is supposed to bring up a chat interface but this is not working correctly yet.

## API

Documentation is TBD. But the long and short of it is that there are two interfaces
* `LlamaInference` - this one is a high level interface that tries to take care of most things for you. The demo script below uses this.
* `LlamaContext` - this is a low level interface to the underlying llama.cpp API. You can use this similar to how the [main](https://github.com/ggerganov/llama.cpp/blob/master/examples/main/main.cpp) example in `llama.cpp` does uses the C API. This is a rough implementation and currently untested except for compiling successfully.

## Demo script

See `llamacpp/cli.py` for a detailed example. The simplest demo would be something like the following:

```python
import sys
import llamacpp


def progress_callback(progress):
    print("Progress: {:.2f}%".format(progress * 100))
    sys.stdout.flush()


params = llamacpp.InferenceParams.default_with_callback(progress_callback)
params.path_model = './models/7B/ggml-model-q4_0.bin'
model = llamacpp.LlamaInference(params)

prompt = "A llama is a"
prompt_tokens = model.tokenize(prompt, True)
model.update_input(prompt_tokens)

model.ingest_all_pending_input()

model.print_system_info()
for i in range(20):
    model.eval()
    token = model.sample()
    text = model.token_to_str(token)
    print(text, end="")
    
# Flush stdout
sys.stdout.flush()

model.print_timings()
```

## ToDo

- [ ] Investigate using dynamic versions using setuptools-scm (Example: https://github.com/pypa/setuptools_scm/blob/main/scm_hack_build_backend.py)
