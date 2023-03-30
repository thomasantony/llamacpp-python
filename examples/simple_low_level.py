import llamacpp

params = llamacpp.LlamaContextParams()
model = llamacpp.LlamaContext("./models/7B/ggml-model-q4_0.bin", params)

print(model.get_n_embd())
