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

**Note that running `llamacpp-convert` requires `torch`, `sentencepiece` and `numpy` to be installed. These packages are not installed by default when your install `llamacpp`.**

## Command line interface

The package installs the command line entry point `llamacpp-cli` that points to `llamacpp/cli.py` and should provide about the same functionality as the `main` program in the original C++ repository. There is also an experimental `llamacpp-chat` that is supposed to bring up a chat interface but this is not working correctly yet.

## Demo script

See `llamacpp/cli.py` for a detailed example. The simplest demo would be something like the following:

```python

params = llamacpp.gpt_params(
	'./models/7B/ggml_model_q4_0.bin', # model,
	"A llama is a ", # prompt
	"", # reverse_prompt
	512, # ctx_size
	100, # n_predict
	40, # top_k
	0.95, # top_p
	0.85, # temp
	1.30, # repeat_penalty
	-1, # seed
	8, # threads
	64, # repeat_last_n
	8, # batch_size
	False, # color
	False, # interactive or args.interactive_start
	False, # interactive_start
)
model = llamacpp.PyLLAMA(params)
model.add_bos()		# Adds "beginning of string" token
model.update_input(params.prompt)
model.print_startup_stats()
model.prepare_context()

model.ingest_all_pending_input(True)
while not model.is_finished():
	model.ingest_all_pending_input(not input_noecho)
	text, is_finished = model.infer_text()
	print(text, end="")
if is_finished:
	break
```

## ToDo

- [x] Use poetry to build package
- [x] Add command line entry point for quantize script
- [x] Publish wheel to PyPI
- [ ] Add chat interface based on tinygrad
