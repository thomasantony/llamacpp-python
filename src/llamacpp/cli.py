"""Python version of main.cpp"""
import sys
import argparse
import llamacpp
from typing import Dict


def parse_args_into_params(argv) -> Dict[str, str]:
    """Parses arguments using argparse based on usage information above"""
    parser = argparse.ArgumentParser(description="llama.cpp CLI")
    parser.add_argument("-i", "--interactive", action="store_true", help="run in interactive mode")
    parser.add_argument(
        "-ins", "--instruct",
        action="store_true",
        help="run in 'instruct mode' where the user is prompted to enter a command",
        default=False,
    )
    parser.add_argument(
        "-r",
        "--reverse-prompt",
        type=str,
        help="in interactive mode, poll user input upon seeing PROMPT",
        default="",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="colorise output to distinguish prompt and user input from generations",
    )
    parser.add_argument("-s", "--seed", type=int, default=-1, help="RNG seed (default: -1)")
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=4,
        help="number of threads to use during computation (default: 4)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="prompt to start generation with (default: random)",
    )
    parser.add_argument(
        "-f", "--file", type=str, default="", help="prompt file to start generation."
    )
    parser.add_argument(
        "-n", "--n_predict", type=int, default=128, help="number of tokens to predict (default: 128)"
    )
    parser.add_argument("--top_k", type=int, default=40, help="top-k sampling (default: 40)")
    parser.add_argument("--top_p", type=float, default=0.95, help="top-p sampling (default: 0.1)")
    parser.add_argument(
        "--repeat_last_n",
        type=int,
        default=64,
        help="last n tokens to consider for penalize (default: 64)",
    )
    parser.add_argument(
        "--repeat_penalty",
        type=float,
        default=1.30,
        help="penalize repeat sequence of tokens (default: 1.30)",
    )
    parser.add_argument(
        "-c",
        "--ctx_size",
        type=int,
        default=512,
        help="size of the prompt context (default: 512)",
    )
    parser.add_argument("--temp", type=float, default=0.8, help="temperature (default: 0.7)")
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8,
        help="batch size for prompt processing (default: 8)",
    )
    parser.add_argument("-m", "--model", type=str, default="./models/7B/ggml-model-q4_0.bin", help="model path (default: )")
    parser.add_argument("--mlock", action="store_true", help="use mlock to lock memory")
    parser.add_argument("--memory_f16", action="store_true", help="use half-precision memory")

    args = parser.parse_args(argv[1:])

    if args.interactive or args.instruct:
        print("WARNING: interactive mode and instruct mode are currently broken")
    return args


def process_interactive_input(model: llamacpp.LlamaInference):
    """Process interactive input similar to the C++ version"""

    # Read lines as long as user is entering "\" at the end of the line
    # Pass each line to the model
    while True:
        line = input()
        if line.endswith("\\"):
            line = line[:-1]
            model.update_input(line)
        else:
            model.update_input(line)
            break


def main(args):
    """Main function"""

    # Add a space in front of the first character to match OG llama tokenizer behavior
    args.prompt = " " + args.prompt
    
    params = llamacpp.InferenceParams()
    params.path_model = args.model
    params.seed = args.seed
    params.n_threads = args.threads

    params.repeat_last_n = args.repeat_last_n
    params.n_batch = args.batch_size
    params.top_k = args.top_k
    params.top_p = args.top_p
    params.temp = args.temp
    params.repeat_penalty = args.repeat_penalty
    params.use_mlock = args.mlock
    params.memory_f16 = args.memory_f16
    params.n_ctx = args.ctx_size

    model = llamacpp.LlamaInference(params)
    model.update_input([model.token_bos()])
    model.update_input(args.prompt)
    print(model.system_info())

    inp_pfx = model.tokenize("\n\n### Instruction:\n\n", True)
    inp_sfx = model.tokenize("\n\n### Response:\n\n", False)

    if args.instruct:
        args.interactive = True
        args.reverse_prompt = "### Instruction:\n\n"

    # Set antiprompt if we are in interactive mode
    if args.reverse_prompt:
        args.interactive = True

    if args.interactive:
        print("== Running in interactive mode. ==")
        print(" - Press Ctrl+C to interject at any time.")
        print(" - Press Return to return control to LLaMa.")
        print(" - If you want to submit another line, end your input in '\\'.")
        print()
        is_interacting = True

    input_noecho = False
    is_finished = False

    print(args.prompt, end="")

    n_output = 0
    while n_output < args.n_predict:
        if model.has_unconsumed_input():
            model.ingest_all_pending_input()
            # # reset color to default if we there is no pending user input
            # if (!input_noecho && args.use_color) {
            #     printf(ANSI_COLOR_RESET);
            # }
        else:
            token = model.sample()
            text = model.token_to_str(token)
            print(text, end="")
            n_output += 1
            is_finished = token == model.token_eos()
            input_noecho = False

        if args.interactive:
            if model.is_antiprompt_present():
                # reverse prompt found
                is_interacting = True
            if is_interacting:
                if args.instruct:
                    model.update_input_tokens(inp_pfx)
                    print("\n> ", end="")

                process_interactive_input(model)

                if args.instruct:
                    model.update_input_tokens(inp_sfx)

                input_noecho = True
                is_interacting = False
        
        # end of text token was found
        if is_finished:
            if args.interactive:
                is_interacting = True
            else:
                print(" [end of text]")
                break
        
        if args.interactive and model.is_finished():
            model.reset_remaining_tokens()
            is_interacting = True

    return 0


def run():
    # Parse params into a gpt_params object
    args = parse_args_into_params(sys.argv)

    # if args.file is specified, read the file and set the prompt to the contents
    if args.file:
        with open(args.file, "r") as f:
            args.prompt = f.read().strip()

    return main(args)


if __name__ == "__main__":
    sys.exit(run())
