import os
import sys


def main():
    """Pass command line arguments to llama_model_quantize"""
    import llamacpp

    # Print usage if not enough arguments are provided
    if len(sys.argv) < 2:
        print(f"Usage: llamacpp-quantize <model_path> [<bits>=0]")
        print("bits: 0 = q4_0, 1 = q4_1\n")
        print("This script assumes that you have already used convert-pth-to-ggml.py to convert")
        print("the pytorch model to a ggml model. It will then quantize the ggml model to INT4")
        print("for use with the llamacpp library.\n")
        print("llamacpp-quantize will walk through the model_path directory and quantize all")
        print("ggml-model-f16.bin.* files it finds. The output files will be named")
        print("ggml-model-q4_0.bin.* or ggml-model-q4_1.bin.* depending on the value of <bits>.\n")
        sys.exit(1)
    
    model_path = sys.argv[1]
    if len(sys.argv) < 3:
        bits = 0
    else:
        bits = int(sys.argv[2])

    # Convert "bits" to input for llama_model_quantize()
    if bits == 0:
        q_type = 2
        q_type_str = 'q4_0'
    elif bits == 1:
        q_type = 3
        q_type_str = 'q4_1'

    # Print the model path
    print(f"Quantizing model in {model_path} to {q_type_str}")

    # Walk through files in model_path matching ggml-model-q*.bin 
    # and pass them to llama_model_quantize()
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.startswith("ggml-model-f16") and file.endswith(".bin"):               
                output_file = file.replace("-f16.bin", f"-{q_type_str}.bin")
                print(f"Quantizing file: {file} to {output_file}")
                llamacpp.llama_model_quantize(os.path.join(root, file), os.path.join(root, output_file), q_type)


if __name__ == '__main__':
    main()
