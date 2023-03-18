"""Python version of main.cpp"""
import sys
import argparse
import llamacpp


def parse_args_into_params(argv) -> llamacpp.gpt_params:
    """Parses arguments using argparse based on usage information above"""
    parser = argparse.ArgumentParser(description="llama.cpp CLI")
    parser.add_argument("-i", "--interactive", action="store_true", help="run in interactive mode")
    parser.add_argument(
        "--interactive-start",
        action="store_true",
        help="run in interactive mode and poll user input at startup",
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
        help="number of threads to use during computation (default: 1)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="prompt to start generation with (default: random)",
        required=True,
    )
    # parser.add_argument(
    #     "-f", "--file", type=str, default="", help="prompt file to start generation."
    # )
    parser.add_argument(
        "-n", "--n_predict", type=int, default=128, help="number of tokens to predict (default: 128)"
    )
    parser.add_argument("--top_k", type=int, default=40, help="top-k sampling (default: 40)")
    parser.add_argument("--top_p", type=float, default=0.95, help="top-p sampling (default: 0.1)")
    parser.add_argument(
        "--repeat_last_n",
        type=int,
        default=64,
        help="last n tokens to consider for penalize (default: 0)",
    )
    parser.add_argument(
        "--repeat_penalty",
        type=float,
        default=1.30,
        help="penalize repeat sequence of tokens (default: 0.0)",
    )
    parser.add_argument(
        "-c",
        "--ctx_size",
        type=int,
        default=4096,
        help="size of the prompt context (default: 4096)",
    )
    parser.add_argument("--temp", type=float, default=0.8, help="temperature (default: 0.7)")
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8,
        help="batch size for prompt processing (default: 2)",
    )
    parser.add_argument("-m", "--model", type=str, default="./models/7B/ggml-model-q4_0.bin", help="model path (default: )")
    parser.usage = parser.format_help()

    args = parser.parse_args(argv[1:])

    # Add a space in front of the first character to match OG llama tokenizer behavior
    args.prompt = " " + args.prompt

    # Initialize gpt_params object
    params = llamacpp.gpt_params(
        args.model,
        args.prompt,
        args.reverse_prompt,
        args.ctx_size,
        args.n_predict,
        args.top_k,
        args.top_p,
        args.temp,
        args.repeat_penalty,
        args.seed,
        args.threads,
        args.repeat_last_n,
        args.batch_size,
        args.color,
        args.interactive or args.interactive_start,
        args.interactive_start,
    )

    return params


def process_interactive_input(model: llamacpp.PyLLAMA):
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


def main(params):
    """Main function"""
    model = llamacpp.PyLLAMA(params)
    model.add_bos()
    model.update_input(params.prompt)
    model.print_startup_stats()
    model.prepare_context()

    # Set antiprompt if we are in interactive mode
    if params.interactive:
        model.set_antiprompt(params.antiprompt)

    if params.interactive:
        print("== Running in interactive mode. ==")
        print(" - Press Ctrl+C to interject at any time.")
        print(" - Press Return to return control to LLaMa.")
        print(" - If you want to submit another line, end your input in '\\'.")
        print()

    # prompt user immediately after the starting prompt has been loaded
    if params.interactive_start:
        is_interacting = True

    input_noecho = False
    is_finished = False

    while not model.is_finished():
        if model.has_unconsumed_input():
            model.ingest_all_pending_input(not input_noecho)
            # # reset color to default if we there is no pending user input
            # if (!input_noecho && params.use_color) {
            #     printf(ANSI_COLOR_RESET);
            # }
        else:
            text, is_finished = model.infer_text()
            print(text, end="")
            input_noecho = False

        if params.interactive:
            if model.is_antiprompt_present():
                # reverse prompt found
                is_interacting = True
            if is_interacting:
                process_interactive_input(model)
                input_noecho = True
                is_interacting = False

        if is_finished:
            break

    return 0


def run():
    # Parse params into a gpt_params object
    params = parse_args_into_params(sys.argv)
    return main(params)

if __name__ == "__main__":
    sys.exit(run())
