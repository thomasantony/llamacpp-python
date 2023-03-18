"""A modified version of llamacpp-cli and includes a good prompt for the chatbot"""
import sys
import llamacpp
import argparse

from llamacpp.cli import main as llamacpp_main

# Default prompt
prompt = """Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User: Hello, Bob.
Bob: Hello. How may I help you today?
User:"""


def parse_chat_params(argv) -> llamacpp.gpt_params:
    """Parse chat parameters"""

    parser = argparse.ArgumentParser(description="LLaMa")
    parser.add_argument("-i", "--interactive", action="store_true", help="run in interactive mode", default=True)
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
        default="User:",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="colorise output to distinguish prompt and user input from generations",
        default=True,
    )
    parser.add_argument("-s", "--seed", type=int, default=-1, help="RNG seed (default: -1)")
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=8,
        help="number of threads to use during computation (default: 4)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="prompt to start generation with (default: random)",
        default=prompt,
    )
    # parser.add_argument(
    #     "-f", "--file", type=str, default="", help="prompt file to start generation."
    # )
    parser.add_argument(
        "-n", "--n_predict", type=int, default=256, help="number of tokens to predict (default: 128)"
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


def run():
    params = parse_chat_params(sys.argv)
    return llamacpp_main(params)


if __name__ == "__main__":
    sys.exit(run())
