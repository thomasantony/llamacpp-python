import array
import llamacpp

params = llamacpp.LlamaContextParams()
params.seed = 19472
model = llamacpp.LlamaContext("./models/7B/ggml-model-f16.bin", params)

prompt = "Llama is"
# add a space in front of the first character to match OG llama tokenizer behavior
prompt = f" {prompt}"

# tokenize the prompt
embd_inp = model.str_to_token(prompt, True)

n_ctx = model.get_n_ctx()

if len(embd_inp) > n_ctx - 4:
    raise Exception("Prompt is too long")

n_past = 0
n_remain = 9
n_consumed = 0
embd = []

while n_remain:
    if len(embd):
        if model.eval(array.array('i', embd), len(embd), n_past, 1):
            raise Exception("Failed to predict\n")
    n_past += len(embd)
    embd.clear()

    if len(embd_inp) <= n_consumed:
        # sample
        top_k = 40
        top_p = 0.95
        temp = 0.8
        repeat_penalty = 0.0

        # sending an empty array for the last n tokens
        id = model.sample_top_p_top_k(array.array('i', []), top_k, top_p, temp, repeat_penalty)
        # add it to the context
        embd.append(id)
        # decrement remaining sampling budget
        n_remain -= 1
    else:
        # has unconsumed input
        while len(embd_inp) > n_consumed:
            # update_input
            embd.append(embd_inp[n_consumed])
            n_consumed += 1

    for id in embd:
        print(model.token_to_str(id), end="")
