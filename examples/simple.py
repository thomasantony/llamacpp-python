import sys
import llamacpp


def progress_callback(progress):
    # print("Progress: {:.2f}%".format(progress * 100))
    # sys.stdout.flush()
    pass


params = llamacpp.InferenceParams.default_with_callback(progress_callback)
params.path_model = './models/7B/ggml-model-f16.bin'
params.seed = 19472
params.repeat_penalty = 1.0
model = llamacpp.LlamaInference(params)

prompt = " Llama is"
prompt_tokens = model.tokenize(prompt, True)
model.update_input(prompt_tokens)

model.ingest_all_pending_input()
print(model.system_info())

print(prompt, end='')
for i in range(20):
    model.eval()
    token = model.sample()
    text = model.token_to_str(token)
    print(text, end="")
    
# Flush stdout
sys.stdout.flush()

# model.print_timings()
