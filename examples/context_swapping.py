"""Demonstrates that the library now supports going over the context size limit 
(but loses "memory" of earlier text in the process)"""
import sys
import llamacpp


params = llamacpp.InferenceParams()
params.path_model = './models/7B/ggml-model-f16.bin'
params.seed = 69420
params.repeat_penalty = 1.0
params.n_ctx = 128
model = llamacpp.LlamaInference(params)

prompt = " Llama is"
prompt_tokens = model.tokenize(prompt, True)
model.update_input(prompt_tokens)

model.ingest_all_pending_input()
print(model.system_info())

print(prompt, end='')
for i in range(256):
    model.eval()
    token = model.sample()
    text = model.token_to_str(token)
    print(text, end='')
    sys.stdout.flush()

print()
model.print_timings()
